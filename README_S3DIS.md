# S3DIS Dataset Memory-Efficient Loader

This code provides a memory-efficient way to work with the large S3DIS (Stanford 3D Indoor Scenes) dataset in PyTorch. It's designed to handle datasets that are too large to fit in memory all at once.

## Dataset Structure

The data is structured as 6 separate `.pt` files:

- `s3dis1.pt` through `s3dis6.pt`
- Each file represents a different area of the S3DIS dataset
- Areas 1-5 are used for training
- Area 6 is used for testing

## Features

- **Memory-Efficient Loading**: Loads data on-demand rather than all at once
- **Index Caching**: Avoids loading files during initialization by caching index structure
- **Flexible Loading Options**: Can switch between memory-efficient and full in-memory loading
- **Proper Train/Test Split**: Automatically splits into training (Areas 1-5) and test (Area 6) sets
- **Batch Processing**: Supports minibatch processing through PyTorch DataLoader
- **Chunked Dataset Support**: Can split large files into smaller chunks for easier handling

## Usage

### Basic Usage with Optimized Loading

```python
from data_processing.datasets import create_s3dis_dataloaders

# Create dataloaders with optimized loading
train_loader, test_loader = create_s3dis_dataloaders(
    data_path='./data',  # Directory containing s3dis1.pt through s3dis6.pt
    batch_size=8,
    num_workers=4,
    load_in_memory=False,  # Load files on demand
    use_cached_index=True  # Use cached index to avoid loading files during initialization
)

# Access data
for batch in train_loader:
    points = batch['x']  # Point coordinates and features [B, N, 6] tensor
    labels = batch['y']  # Point labels (list of label strings)
    area = batch['area']  # Area index
    room_idx = batch['room_idx']  # Room index within the area

    # Your training code here
```

### Chunking Large Files

If the dataset is too large (over 40GB), you can split it into smaller chunks using the provided script:

```bash
python split_s3dis_data.py --input_path ./data --output_path ./data_chunked --n_chunks 10
```

This will:

1. Load each area file one at a time
2. Split it into the specified number of chunks (default: 10)
3. Save each chunk with area information preserved in the filename (e.g., `s3dis1_chunk1.pt`)

### Using Chunked Dataset with Optimized Loading

```python
from data_processing.chunked_datasets import create_chunked_dataloaders

# Create dataloaders from the chunked files with optimized loading
train_loader, test_loader = create_chunked_dataloaders(
    data_path='./data_chunked',  # Directory containing chunked files
    batch_size=8,
    num_workers=4,
    load_in_memory=False,  # Load files on demand
    use_cached_index=True   # Use cached index to avoid loading files during initialization
)

# Access data in the same way as the original dataset
for batch in train_loader:
    points = batch['x']
    labels = batch['y']
    area = batch['area']
    # Your training code here
```

### Example Scripts

Two example scripts are provided:

1. `example_use_s3dis.py` - Demonstrates using the original dataset
2. `example_use_chunked_s3dis.py` - Demonstrates using the chunked dataset

Run either example with:

```
python example_use_s3dis.py
# or
python example_use_chunked_s3dis.py
```

## How the Optimized Loading Works

1. **First-time initialization**:

   - Scans directories to find available files
   - Loads file metadata to determine the number of rooms in each file (minimizes data loading)
   - Creates a mapping of (area_idx, room_idx) or (chunk_idx, room_idx) pairs
   - Saves this mapping to a cache file for future use

2. **Subsequent initializations**:

   - Loads the cached index mapping directly instead of scanning files again
   - Avoids loading any actual data files during initialization
   - Significantly reduces startup time, especially with large datasets

3. **During data access**:
   - Loads only the specific file containing the requested room
   - Keeps the file in memory only as long as needed, then frees memory

## Data Format

Each data point represents a room with:

- `x`: Point cloud data with XYZ coordinates and RGB values [N, 6]
- `y`: List of semantic labels for each point

## Memory Optimization Tips

1. Use `load_in_memory=False` to avoid loading all data at once
2. Use `use_cached_index=True` to avoid repeatedly scanning files during initialization
3. Split large files into smaller chunks using `split_s3dis_data.py`
4. Adjust `batch_size` based on your available GPU memory
5. Use multiple workers (`num_workers`) for faster data loading
6. Consider preprocessing the data to reduce size if needed

## Requirements

- PyTorch
- NumPy
- Matplotlib (for visualization in example)
- tqdm (for progress bars)
- pickle (for index caching)
