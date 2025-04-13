import torch
import os
from data_processing.chunked_datasets import create_chunked_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

def main():
    # Set the path to your chunked data directory
    data_path = './data_chunked'
    
    # Create dataloaders with memory-efficient loading and index caching
    print("Creating data loaders for chunked dataset...")
    start_time = time.time()
    
    train_loader, test_loader = create_chunked_dataloaders(
        data_path,
        batch_size=1,
        num_workers=2,
        load_in_memory=False,  # Load files on demand
        use_cached_index=True   # Use cached index to avoid loading files during initialization
    )
    
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.2f} seconds")
    print(f"Train dataset size: {len(train_loader.dataset)} rooms")
    print(f"Test dataset size: {len(test_loader.dataset)} rooms")
    
    # Time getting the first sample
    print("\nFetching first sample...")
    start_time = time.time()
    sample_batch = next(iter(train_loader))
    sample_time = time.time() - start_time
    print(f"Time to fetch first sample: {sample_time:.2f} seconds")
    
    # Display sample information
    points = sample_batch['x'][0]  # [N, 6] tensor - xyz + rgb
    labels = sample_batch['y'][0]  # List of labels
    area = sample_batch['area'][0].item()
    room_idx = sample_batch['room_idx'][0].item()
    
    print("\nSample Information:")
    print(f"Area: {area}, Room Index: {room_idx}")
    print(f"Point cloud shape: {points.shape}")
    print(f"Number of points: {points.shape[0]}")
    print(f"Number of labels: {len(labels)}")
    
    # Get unique labels
    unique_labels = sorted(set(labels))
    print(f"Unique labels ({len(unique_labels)}): {unique_labels}")
    
    # Show label statistics
    print("\nLabel distribution:")
    for label in unique_labels:
        count = labels.count(label)
        percentage = (count / len(labels)) * 100
        print(f"  {label}: {count} points ({percentage:.2f}%)")
    
    # Display point cloud stats
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    
    print("\nPoint Cloud Statistics:")
    print(f"XYZ min: {xyz.min(dim=0).values}")
    print(f"XYZ max: {xyz.max(dim=0).values}")
    print(f"XYZ mean: {xyz.mean(dim=0)}")
    print(f"RGB min: {rgb.min(dim=0).values}")
    print(f"RGB max: {rgb.max(dim=0).values}")
    print(f"RGB mean: {rgb.mean(dim=0)}")
    
    # Verify we can iterate through the entire dataset
    print("\nVerifying data loading by iterating through a few batches...")
    batch_count = min(5, len(train_loader))
    
    start_time = time.time()
    for i, batch in enumerate(tqdm(train_loader, total=batch_count)):
        if i >= batch_count:
            break
        print(f"Batch {i+1}: Shape {batch['x'].shape}, Area {batch['area'][0].item()}")
    
    batches_time = time.time() - start_time
    print(f"Time to load {batch_count} batches: {batches_time:.2f} seconds")
    
    # Get a sample from test set
    print("\nFetching test sample...")
    start_time = time.time()
    test_sample = next(iter(test_loader))
    test_sample_time = time.time() - start_time
    print(f"Time to fetch test sample: {test_sample_time:.2f} seconds")
    
    test_points = test_sample['x'][0]  # [N, 6] tensor
    test_labels = test_sample['y'][0]  # List of labels
    test_area = test_sample['area'][0].item()
    
    print("\nTest Sample Information:")
    print(f"Area: {test_area}")
    print(f"Point cloud shape: {test_points.shape}")
    print(f"Number of points: {test_points.shape[0]}")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()