import time
import os
import argparse
from data_processing.chunked_datasets import ChunkedS3DISDataset, create_chunked_dataloaders
from data_processing.precompute_chunk_index import precompute_chunk_index

def test_with_precomputed_index(data_path):
    """Test dataset loading with precomputed index"""
    start_time = time.time()
    
    # Create dataset with precomputed index
    dataset = ChunkedS3DISDataset(
        path=data_path,
        load_in_memory=False,
        require_index_file=True
    )
    
    end_time = time.time()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")
    print(f"Total samples: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(0, min(len(dataset), 20), 5):
        sample = dataset[i]
        print(f"Sample {i} - Area: {sample['area']}, X shape: {sample['x'].shape}, Y length: {len(sample['y'])}")
    
    return dataset

def test_with_on_the_fly_index(data_path):
    """Test dataset loading with on-the-fly index building"""
    start_time = time.time()
    
    # Create dataset with on-the-fly index building
    dataset = ChunkedS3DISDataset(
        path=data_path,
        load_in_memory=False,
        require_index_file=False
    )
    
    end_time = time.time()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")
    print(f"Total samples: {len(dataset)}")
    
    return dataset

def test_dataloader(data_path):
    """Test dataloader with precomputed index"""
    start_time = time.time()
    
    # Create dataloaders
    train_loader, test_loader = create_chunked_dataloaders(
        data_path=data_path,
        batch_size=2,
        num_workers=2,
        require_index_file=True
    )
    
    end_time = time.time()
    print(f"Dataloaders created in {end_time - start_time:.2f} seconds")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Test loading a batch
    print("Loading first batch from train loader...")
    batch_start = time.time()
    batch = next(iter(train_loader))
    batch_end = time.time()
    print(f"Batch loaded in {batch_end - batch_start:.2f} seconds")
    
    # Updated to handle lists instead of tensors
    print(f"Batch contains {len(batch['x'])} samples")
    for i in range(len(batch['x'])):
        print(f"  Sample {i} - X shape: {batch['x'][i].shape}, Y length: {len(batch['y'][i])}")
    
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the chunked dataset with precomputed index')
    parser.add_argument('--data_path', type=str, default='./data_chunked',
                        help='Path to directory containing chunked s3dis*.pt files')
    parser.add_argument('--precompute', action='store_true',
                        help='Run precomputation before testing')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare performance with and without precomputed index')
    
    args = parser.parse_args()
    
    # Check if index file exists
    index_file = os.path.join(args.data_path, 'chunked_s3dis_index_mapping.pkl')
    if not os.path.exists(index_file) or args.precompute:
        print("Precomputing index file...")
        precompute_chunk_index(args.data_path, force_recompute=args.precompute)
    
    # Test with precomputed index
    print("\n=== Testing with precomputed index ===")
    dataset_precomputed = test_with_precomputed_index(args.data_path)
    
    # Test dataloader
    print("\n=== Testing dataloader with precomputed index ===")
    train_loader, test_loader = test_dataloader(args.data_path)
    
    # Optionally compare with on-the-fly index building
    if args.compare:
        print("\n=== Testing with on-the-fly index building ===")
        dataset_on_the_fly = test_with_on_the_fly_index(args.data_path) 