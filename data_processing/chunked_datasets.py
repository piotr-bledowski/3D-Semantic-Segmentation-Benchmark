import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np
import pickle


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-sized tensors.
    Instead of stacking tensors of different sizes, it keeps them as a list.
    """
    elem = batch[0]
    result = {}
    
    for key in elem:
        if key == 'x':
            # Keep 'x' tensors as a list rather than trying to stack them
            result[key] = [d[key] for d in batch]
        elif key == 'y':
            # Keep 'y' lists as is
            result[key] = [d[key] for d in batch]
        else:
            # For other fields like 'area' and 'room_idx', convert to tensor if they're scalars
            if isinstance(elem[key], (int, float)):
                result[key] = torch.tensor([d[key] for d in batch])
            else:
                result[key] = [d[key] for d in batch]
    
    return result


class ChunkedS3DISDataset(Dataset):
    def __init__(self, path, area_indices=None, load_in_memory=False, require_index_file=True):
        """
        Dataset for loading chunked S3DIS data
        
        Args:
            path: Path to directory containing chunked s3dis*.pt files
            area_indices: List of area indices to load (1-6)
            load_in_memory: If True, load all data into memory; if False, load on demand
            require_index_file: If True, require a precomputed index file; if False, build index on the fly
        """
        self.path = path
        self.area_indices = area_indices or list(range(1, 7))
        self.load_in_memory = load_in_memory
        self.require_index_file = require_index_file
        
        # Dictionary to store loaded data if load_in_memory=True
        self.data_cache = {}
        
        # Path to the cached index file
        index_cache_file = os.path.join(path, 'chunked_s3dis_index_mapping.pkl')
        
        # Check if precomputed index exists
        if not os.path.exists(index_cache_file):
            if require_index_file:
                raise FileNotFoundError(
                    f"Index file {index_cache_file} not found. Please run precompute_chunk_index.py first "
                    f"or set require_index_file=False to build index on the fly."
                )
            else:
                print("Warning: Index file not found. Building index on the fly may be slow.")
                self._build_index_on_the_fly()
                return
                
        # Load the precomputed index
        print(f"Loading precomputed index from {index_cache_file}")
        with open(index_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            
            # Get only the chunk files for the requested areas
            self.chunk_files = []
            filtered_index_mapping = []
            chunk_idx_mapping = {}  # Maps old chunk_idx to new chunk_idx
            
            # Filter chunk files by area
            for i, chunk_file in enumerate(cached_data['chunk_files']):
                filename = os.path.basename(chunk_file)
                area = int(filename.split('_')[0].replace('s3dis', ''))
                
                if area in self.area_indices:
                    chunk_idx_mapping[i] = len(self.chunk_files)
                    self.chunk_files.append(chunk_file)
            
            # Filter and remap index mapping
            for chunk_idx, room_idx in cached_data['index_mapping']:
                # Check if the chunk is in our filtered set
                if chunk_idx in chunk_idx_mapping:
                    new_chunk_idx = chunk_idx_mapping[chunk_idx]
                    filtered_index_mapping.append((new_chunk_idx, room_idx))
            
            self.index_mapping = filtered_index_mapping
            
        # Load data into memory if requested
        if load_in_memory:
            for chunk_idx, chunk_file in enumerate(tqdm(self.chunk_files, desc="Loading chunks into memory...")):
                self.data_cache[chunk_idx] = torch.load(chunk_file)
                
        print(f'Dataset initialized with {len(self.index_mapping)} rooms from {len(self.chunk_files)} chunk files')
    
    def _build_index_on_the_fly(self):
        """Build the index mapping on the fly by loading each chunk file."""
        # Find all chunk files for the specified areas
        self.chunk_files = []
        for area_idx in self.area_indices:
            pattern = os.path.join(self.path, f's3dis{area_idx}_chunk*.pt')
            area_files = sorted(glob.glob(pattern))
            if not area_files:
                print(f"Warning: No chunk files found for area {area_idx} at {pattern}")
            self.chunk_files.extend(area_files)
        
        # Build index mapping
        self.index_mapping = []  # (chunk_file_idx, room_idx_in_chunk)
        
        print("Building index mapping from chunk files...")
        for chunk_idx, chunk_file in enumerate(tqdm(self.chunk_files, desc="Building index mapping...")):
            if self.load_in_memory:
                # Load data into memory
                chunk_data = torch.load(chunk_file)
                self.data_cache[chunk_idx] = chunk_data
                num_rooms = len(chunk_data)
            else:
                # Just get the length without keeping data in memory
                chunk_data = torch.load(chunk_file)
                num_rooms = len(chunk_data)
                del chunk_data  # Free memory
            
            # Add entries to index mapping
            for room_idx in range(num_rooms):
                self.index_mapping.append((chunk_idx, room_idx))
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        chunk_idx, room_idx = self.index_mapping[idx]
        
        # Extract area from the filename
        chunk_file = self.chunk_files[chunk_idx]
        area = int(os.path.basename(chunk_file).split('_')[0].replace('s3dis', ''))
        
        if self.load_in_memory:
            # Get from memory cache
            if chunk_idx not in self.data_cache:
                # Load on demand if not already in memory
                self.data_cache[chunk_idx] = torch.load(self.chunk_files[chunk_idx])
                print(f"Chunk {chunk_idx} loaded into memory on demand")
            room_data = self.data_cache[chunk_idx][room_idx]
        else:
            # Load on demand
            chunk_data = torch.load(self.chunk_files[chunk_idx])
            room_data = chunk_data[room_idx]
            del chunk_data  # Free memory
        
        # Convert lists to tensors if needed
        x = torch.tensor(room_data['x'], dtype=torch.float32)
        y = room_data['y']  # Keep as list of strings for now
        
        return {'x': x, 'y': y, 'area': area, 'room_idx': room_idx}


def create_chunked_dataloaders(data_path, batch_size=1, num_workers=4, load_in_memory=False, require_index_file=True):
    """Create train and test data loaders for chunked S3DIS dataset.
    Train: Areas 1-5, Test: Area 6
    """
    # Create the datasets
    train_dataset = ChunkedS3DISDataset(
        path=data_path,
        area_indices=[1, 2, 3, 4, 5],
        load_in_memory=load_in_memory,
        require_index_file=require_index_file
    )
    
    test_dataset = ChunkedS3DISDataset(
        path=data_path,
        area_indices=[6],
        load_in_memory=load_in_memory,
        require_index_file=require_index_file
    )
    
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Example usage
    data_path = './data_chunked'
    
    # Test the dataset
    dataset = ChunkedS3DISDataset(data_path, load_in_memory=False, require_index_file=True)
    print(f"Total samples: {len(dataset)}")
    
    # Access a sample
    sample = dataset[0]
    print(f"Sample area: {sample['area']}")
    print(f"Sample x shape: {sample['x'].shape}")
    print(f"Sample y length: {len(sample['y'])}")
    
    # Test the dataloaders
    train_loader, test_loader = create_chunked_dataloaders(
        data_path, 
        batch_size=2,
        require_index_file=True
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}") 