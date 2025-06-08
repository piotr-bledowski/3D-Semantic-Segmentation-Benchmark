# DGCNN for S3DIS Semantic Segmentation

This directory contains a complete implementation of **Dynamic Graph CNN (DGCNN)** for 3D point cloud semantic segmentation, specifically adapted for the **S3DIS dataset** and optimized for **GTX 1650** GPU constraints.

## 📁 Files Overview

- **`dgcnn.py`** - Main model implementation with both xyz-only and xyz+rgb variants
- **`train_model.py`** - Training script optimized for GTX1650 memory limitations
- **`utils.py`** - Utility functions for model loading, evaluation, and visualization
- **`evaluate.py`** - Standalone evaluation script for trained models
- **`README.md`** - This documentation file

## 🏗️ Model Architecture

The DGCNN implementation includes:

### Core Components

- **EdgeConv layers**: Dynamic graph convolution with k-nearest neighbors
- **Multi-scale feature extraction**: Concatenating features from different layers
- **Memory-efficient design**: Optimized for 4GB VRAM (GTX1650)

### Two Model Variants

1. **`DGCNN`**: Uses only XYZ coordinates for graph construction
2. **`DGCNNWithColor`**: Incorporates RGB color information

### Key Features

- **Adaptive point processing**: Handles variable-sized point clouds
- **Sliding window inference**: For large scenes that don't fit in GPU memory
- **Memory optimizations**: Reduced embedding dimensions and efficient graph construction

## 🚀 Quick Start

### 1. Training

```bash
# Navigate to the DGCNN directory
cd models/dgcnn

# Run training with default GTX1650-optimized settings
python train_model.py
```

### 2. Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path DGCNNTraining/Model/dgcnn_s3dis.pt
```

### 3. Using in Code

```python
from models.dgcnn.dgcnn import get_model
from models.dgcnn.utils import load_trained_model

# Create a new model
model = get_model(num_classes=13, use_color=True, k=16, emb_dims=256)

# Load a trained model
model, config = load_trained_model('path/to/model.pt')
```

## ⚙️ Configuration for GTX1650

The implementation is specifically optimized for GTX1650 (4GB VRAM):

### Recommended Settings

```python
config = {
    'k': 16,              # Reduced from 20 for faster computation
    'emb_dims': 256,      # Reduced from 1024 for memory efficiency
    'batch_size': 1,      # Keep small for GTX1650
    'cut': 2048,          # Limit points per scene
    'sampling': 0.8,      # Use 80% of points to reduce memory
    'dropout': 0.5
}
```

### Memory Optimization Features

- **Point sampling**: Randomly sample a fraction of points during training
- **Point cutting**: Limit maximum points per scene
- **Reduced embedding dimensions**: Lower feature dimensions
- **Efficient graph construction**: Optimized k-NN computation
- **Gradient checkpointing**: Available for even more memory savings

## 📊 Dataset Compatibility

Works with the **chunked S3DIS dataset** format:

- **Input**: Point clouds with XYZ coordinates (+ optional RGB)
- **Output**: Per-point semantic segmentation (13 classes)
- **Classes**: ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter

### Data Format

```python
# Expected input format
x = torch.tensor([N, 6])  # [xyz + rgb] or [N, 3] for xyz only
y = ["ceiling", "floor", ...]  # String labels per point
```

## 🎯 Performance Expectations

### On GTX1650:

- **Training time**: ~2-3 hours for 30 epochs (depending on data size)
- **Memory usage**: ~3.5GB VRAM with recommended settings
- **Inference speed**: ~0.1-0.5s per scene (depending on size)

### Accuracy (Expected):

- **Overall Accuracy**: 85-90%
- **Mean IoU**: 65-75%
- **Mean Class Accuracy**: 70-80%

## 🔧 Troubleshooting

### Out of Memory Errors

If you encounter GPU memory errors:

1. **Reduce parameters**:

   ```python
   config['cut'] = 1024        # Fewer points per scene
   config['sampling'] = 0.9    # Use fewer points
   config['emb_dims'] = 128    # Smaller embeddings
   config['k'] = 12            # Fewer neighbors
   ```

2. **Use CPU fallback**:

   ```bash
   python train_model.py --device cpu
   ```

3. **Monitor memory**:
   ```python
   from utils import get_memory_usage
   get_memory_usage()
   ```

### Data Loading Issues

If data loading fails:

```python
# Try without cached index
create_chunked_dataloaders(
    data_path,
    require_index_file=False  # Build index on the fly
)
```

## 📈 Advanced Usage

### Custom Training Configuration

```python
from train_model import create_dgcnn_trainer

# Create custom trainer
trainer = create_dgcnn_trainer(
    data_path="path/to/data",
    use_color=True,
    k=20,
    emb_dims=512,
    epochs=50,
    cut=4096,
    sampling=0.7
)

# Run training
model = trainer()
```

### Model Evaluation

```python
from utils import evaluate_model, load_trained_model

# Load model
model, config = load_trained_model('model.pt')

# Evaluate
results = evaluate_model(model, test_loader, device='cuda')
print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
print(f"Mean IoU: {results['mean_iou']:.4f}")
```

### Single Scene Prediction

```python
from utils import predict_single_scene

# Predict on a single point cloud
predictions, confidences = predict_single_scene(
    model,
    points,  # [N, 6] tensor
    device='cuda',
    batch_size=2048
)
```

## 🔬 Model Details

### Architecture Summary

```
Input: [B, 6, N] (xyz + rgb)
├── EdgeConv1: [B, 64, N]
├── EdgeConv2: [B, 64, N]
├── EdgeConv3: [B, 64, N]
├── EdgeConv4: [B, 128, N]
├── Concat: [B, 320, N]
├── Global Conv: [B, emb_dims, N]
├── Combine: [B, 320+emb_dims, N]
├── Seg Head: [B, 512, N] → [B, 256, N]
└── Output: [B, num_classes, N]
```

### Key Innovations

- **Dynamic graphs**: Graph topology updated at each layer
- **Edge features**: Uses both relative and absolute coordinates
- **Multi-scale fusion**: Combines features from multiple resolutions
- **Memory efficiency**: Optimized for resource-constrained environments

## 📚 References

- Original DGCNN paper: "Dynamic Graph CNN for Learning on Point Clouds"
- Implementation inspired by the official DGCNN repository
- Adapted for semantic segmentation and memory efficiency

## 🤝 Contributing

To improve this implementation:

1. Test on different hardware configurations
2. Experiment with different hyperparameters
3. Add support for other datasets
4. Optimize further for memory efficiency

## 📝 License

This implementation follows the same license as the main project.
