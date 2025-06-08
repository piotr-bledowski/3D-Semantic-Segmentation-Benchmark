import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dgcnn import get_model
import time


def load_trained_model(model_path: str, config_path: str = None, device: str = 'auto'):
    """
    Load a trained DGCNN model from saved state dict and configuration
    
    Args:
        model_path: Path to the saved model state dict
        config_path: Path to the model configuration (optional)
        device: Device to load model on ('auto', 'cuda', 'cpu')
    
    Returns:
        model: Loaded DGCNN model
        config: Model configuration dictionary
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration
    if config_path is None:
        config_path = model_path.replace('.pt', '_config.pt')
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(model_path), 'model_config.pt')
    
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location=device)
        print(f"Loaded model configuration from {config_path}")
    else:
        # Default configuration
        config = {
            'num_classes': 13,
            'use_color': True,
            'k': 20,
            'emb_dims': 1024,
            'dropout': 0.5
        }
        print("Using default configuration (config file not found)")
    
    # Create model
    model = get_model(
        num_classes=config['num_classes'],
        use_color=config['use_color'],
        k=config['k'],
        emb_dims=config['emb_dims'],
        dropout=config['dropout']
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def predict_single_scene(model, points: torch.Tensor, device: str = 'cuda', 
                        batch_size: int = 4096, overlap: int = 512):
    """
    Predict semantic labels for a single scene using sliding window approach
    to handle large point clouds that don't fit in GPU memory
    
    Args:
        model: Trained DGCNN model
        points: Point cloud tensor [N, 6] (xyz + rgb) or [N, 3] (xyz only)
        device: Device to run inference on
        batch_size: Size of sliding window
        overlap: Overlap between windows
    
    Returns:
        predictions: Predicted class indices [N]
        confidences: Prediction confidences [N]
    """
    model.eval()
    points = points.to(device)
    n_points = points.shape[0]
    n_features = points.shape[1]
    
    # If small enough, process all at once
    if n_points <= batch_size:
        with torch.no_grad():
            # Reshape for model input [1, features, n_points]
            x = points.T.unsqueeze(0)  # [1, n_features, n_points]
            logits, _, _ = model(x)
            logits = logits.squeeze(0)  # [n_points, num_classes]
            
        predictions = torch.argmax(logits, dim=1)
        confidences = torch.softmax(logits, dim=1).max(dim=1)[0]
        return predictions.cpu(), confidences.cpu()
    
    # Use sliding window for large point clouds
    step = batch_size - overlap
    all_logits = torch.zeros(n_points, model.num_classes, device=device)
    point_counts = torch.zeros(n_points, device=device)
    
    with torch.no_grad():
        for start_idx in range(0, n_points, step):
            end_idx = min(start_idx + batch_size, n_points)
            
            # Get batch
            batch_points = points[start_idx:end_idx]
            batch_size_actual = batch_points.shape[0]
            
            # Reshape for model input
            x = batch_points.T.unsqueeze(0)  # [1, n_features, batch_size_actual]
            
            # Predict
            logits, _, _ = model(x)
            logits = logits.squeeze(0)  # [batch_size_actual, num_classes]
            
            # Accumulate logits
            all_logits[start_idx:end_idx] += logits
            point_counts[start_idx:end_idx] += 1
    
    # Average overlapping predictions
    all_logits = all_logits / point_counts.unsqueeze(1)
    
    predictions = torch.argmax(all_logits, dim=1)
    confidences = torch.softmax(all_logits, dim=1).max(dim=1)[0]
    
    return predictions.cpu(), confidences.cpu()


def evaluate_model(model, test_loader, device: str = 'cuda', class_names: List[str] = None):
    """
    Evaluate DGCNN model on test dataset
    
    Args:
        model: Trained DGCNN model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names for reporting
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(model.num_classes)]
    
    # Metrics tracking
    total_correct = 0
    total_points = 0
    class_correct = torch.zeros(model.num_classes)
    class_total = torch.zeros(model.num_classes)
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Process batch (assuming batch size = 1 for now)
            points = batch['x'][0].to(device)  # [N, 6]
            labels = batch['y'][0]  # List of string labels
            
            # Convert string labels to indices
            label_indices = torch.tensor([class_names.index(label) for label in labels])
            
            # Predict
            predictions, confidences = predict_single_scene(
                model, points, device=device, batch_size=2048
            )
            
            # Calculate metrics
            correct = (predictions == label_indices).sum().item()
            total_correct += correct
            total_points += len(labels)
            
            # Per-class metrics
            for i in range(model.num_classes):
                class_mask = (label_indices == i)
                class_total[i] += class_mask.sum().item()
                class_correct[i] += ((predictions == label_indices) & class_mask).sum().item()
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(label_indices.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    eval_time = time.time() - start_time
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_points
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(model.num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            class_accuracies.append(acc.item())
        else:
            class_accuracies.append(0.0)
    
    mean_class_accuracy = np.mean(class_accuracies)
    
    # Calculate IoU (Intersection over Union)
    class_ious = []
    all_pred_np = np.array(all_predictions)
    all_target_np = np.array(all_targets)
    
    for i in range(model.num_classes):
        pred_mask = (all_pred_np == i)
        target_mask = (all_target_np == i)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union > 0:
            iou = intersection / union
            class_ious.append(iou)
        else:
            class_ious.append(0.0)
    
    mean_iou = np.mean(class_ious)
    
    # Compile results
    results = {
        'overall_accuracy': overall_accuracy,
        'mean_class_accuracy': mean_class_accuracy,
        'mean_iou': mean_iou,
        'class_accuracies': class_accuracies,
        'class_ious': class_ious,
        'class_names': class_names,
        'total_points': total_points,
        'evaluation_time': eval_time
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean Class Accuracy: {mean_class_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Total Points: {total_points:,}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    
    print("\nPer-Class Results:")
    print(f"{'Class':<12} {'Accuracy':<10} {'IoU':<10}")
    print("-" * 32)
    for i, name in enumerate(class_names):
        print(f"{name:<12} {class_accuracies[i]:<10.4f} {class_ious[i]:<10.4f}")
    
    return results


def visualize_predictions(points: np.ndarray, predictions: np.ndarray, 
                         ground_truth: np.ndarray = None, class_names: List[str] = None,
                         save_path: str = None):
    """
    Visualize point cloud predictions (requires matplotlib)
    
    Args:
        points: Point cloud coordinates [N, 3]
        predictions: Predicted class indices [N]
        ground_truth: Ground truth class indices [N] (optional)
        class_names: List of class names
        save_path: Path to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(max(predictions) + 1)]
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot predictions
    ax1 = fig.add_subplot(131, projection='3d')
    for i, class_name in enumerate(class_names):
        mask = predictions == i
        if mask.any():
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                       c=[colors[i]], label=class_name, s=1, alpha=0.7)
    ax1.set_title('Predictions')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot ground truth if available
    if ground_truth is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        for i, class_name in enumerate(class_names):
            mask = ground_truth == i
            if mask.any():
                ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                           c=[colors[i]], label=class_name, s=1, alpha=0.7)
        ax2.set_title('Ground Truth')
        
        # Plot errors
        ax3 = fig.add_subplot(133, projection='3d')
        correct_mask = predictions == ground_truth
        error_mask = ~correct_mask
        
        if correct_mask.any():
            ax3.scatter(points[correct_mask, 0], points[correct_mask, 1], points[correct_mask, 2], 
                       c='green', s=1, alpha=0.3, label='Correct')
        if error_mask.any():
            ax3.scatter(points[error_mask, 0], points[error_mask, 1], points[error_mask, 2], 
                       c='red', s=1, alpha=0.7, label='Error')
        ax3.set_title('Errors')
        ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def get_memory_usage():
    """
    Get current GPU memory usage
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print(f"  Total: {memory_total:.2f} GB")
        print(f"  Free: {memory_total - memory_reserved:.2f} GB")
        
        return {
            'allocated': memory_allocated,
            'reserved': memory_reserved,
            'total': memory_total,
            'free': memory_total - memory_reserved
        }
    else:
        print("CUDA not available")
        return None


def benchmark_model(model, input_shape: Tuple[int, int, int], device: str = 'cuda', 
                   num_runs: int = 100):
    """
    Benchmark model inference time
    
    Args:
        model: DGCNN model
        input_shape: Input tensor shape (batch_size, features, num_points)
        device: Device to run benchmark on
        num_runs: Number of runs for averaging
    
    Returns:
        results: Benchmark results dictionary
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'input_shape': input_shape
    }
    
    print(f"Benchmark Results (input shape: {input_shape}):")
    print(f"  Mean time: {results['mean_time']:.4f}s ± {results['std_time']:.4f}s")
    print(f"  Min time: {results['min_time']:.4f}s")
    print(f"  Max time: {results['max_time']:.4f}s")
    print(f"  FPS: {results['fps']:.2f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("DGCNN Utils - Example Usage")
    
    # Test memory usage
    get_memory_usage()
    
    # Test model loading (if model exists)
    model_path = "DGCNNTraining/Model/dgcnn_s3dis.pt"
    if os.path.exists(model_path):
        try:
            model, config = load_trained_model(model_path)
            print("Model loaded successfully!")
            
            # Benchmark model
            benchmark_model(model, (1, 6, 2048), num_runs=10)
            
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model not found at {model_path}")
        print("Train the model first using train_model.py") 