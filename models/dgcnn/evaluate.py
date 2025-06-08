import torch
import os
import argparse
from data_processing.chunked_datasets import create_chunked_dataloaders
from utils import load_trained_model, evaluate_model, get_memory_usage
import time


def main():
    parser = argparse.ArgumentParser(description='Evaluate DGCNN on S3DIS dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data_path', type=str, default='../../data_chunked',
                       help='Path to chunked S3DIS data')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DGCNN Evaluation on S3DIS Dataset")
    print("="*60)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Show memory usage
    if device == 'cuda':
        get_memory_usage()
        torch.cuda.empty_cache()
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    try:
        model, config = load_trained_model(args.model_path, device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    print(f"\nLoading test data from {args.data_path}...")
    try:
        # Only load test set (area 6)
        _, test_loader = create_chunked_dataloaders(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=0 if device == 'cuda' else 2,
            load_in_memory=False,
            use_cached_index=True
        )
        print(f"Test data loaded: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying without cached index...")
        try:
            _, test_loader = create_chunked_dataloaders(
                args.data_path,
                batch_size=args.batch_size,
                num_workers=0 if device == 'cuda' else 2,
                load_in_memory=False,
                require_index_file=False
            )
            print(f"Test data loaded: {len(test_loader.dataset)} samples")
        except Exception as e2:
            print(f"Failed to load data: {e2}")
            return
    
    # S3DIS class names
    s3dis_classes = [
        "ceiling", "floor", "wall", "beam", "column",
        "window", "door", "table", "chair", "sofa",
        "bookcase", "board", "clutter"
    ]
    
    # Run evaluation
    print("\nStarting evaluation...")
    start_time = time.time()
    
    try:
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=s3dis_classes
        )
        
        total_time = time.time() - start_time
        print(f"\nTotal evaluation time: {total_time:.2f}s")
        
        # Save results if requested
        if args.save_results:
            os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
            torch.save(results, args.save_results)
            print(f"Results saved to {args.save_results}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if "out of memory" in str(e).lower():
            print("\nMemory error suggestions:")
            print("- Use smaller batch_size")
            print("- Reduce number of points per scene")
            print("- Use CPU evaluation (--device cpu)")
        return


if __name__ == "__main__":
    main() 