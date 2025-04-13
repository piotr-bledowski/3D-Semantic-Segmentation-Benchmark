import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import random
import numpy as np
from tqdm import tqdm
import pickle
import glob


class S3DISDataset(Dataset):
    def __init__(self, path, area_indices=None, load_in_memory=False, use_cached_index=True):
        """
        Args:
            path: Path to the directory containing s3dis*.pt files
            area_indices: List of area indices to load (1-6)
            load_in_memory: If True, load all data into memory; if False, load on demand
            use_cached_index: If True, use/create a cached index mapping file to avoid loading files during init
        """
        self.path = path
        self.area_indices = area_indices or list(range(1, 7))
        self.load_in_memory = load_in_memory
        self.use_cached_index = use_cached_index
        
        # Dictionary to store loaded data if load_in_memory=True
        self.data_cache = {}
        
        # Build index mapping
        self.index_mapping = []  # (area_idx, room_idx)
        
        # Path to the cached index file
        index_cache_file = os.path.join(path, 's3dis_index_mapping.pkl')
        
        # If use_cached_index is True and cache file exists, load it
        if use_cached_index and os.path.exists(index_cache_file):
            print(f"Loading cached index from {index_cache_file}")
            with open(index_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
                # Only use cached data for areas we need
                for area_idx, num_rooms in cached_data.items():
                    if area_idx in self.area_indices:
                        # Create the index mapping for this area
                        for room_idx in range(num_rooms):
                            self.index_mapping.append((area_idx, room_idx))
                            
                        # If load_in_memory=True, load the area data now
                        if load_in_memory:
                            self.data_cache[area_idx] = torch.load(os.path.join(path, f's3dis{area_idx}.pt'))
                            print(f'Area {area_idx} loaded into memory')
        else:
            # Create the index mapping and optionally cache it
            area_room_counts = {}
            
            for area_idx in tqdm(self.area_indices, desc="Building index mapping..."):
                file_path = os.path.join(path, f's3dis{area_idx}.pt')
                
                if load_in_memory:
                    # Load data into memory
                    self.data_cache[area_idx] = torch.load(file_path)
                    num_rooms = len(self.data_cache[area_idx])
                else:
                    # Get only the length to avoid loading full data
                    try:
                        # Try to get file size first to estimate if it's too large
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        
                        if file_size > 1000:  # If file is larger than 1GB, use a different approach
                            # Try to load only necessary metadata
                            print(f"Area {area_idx} file is large ({file_size:.2f} MB). Loading only metadata...")
                            result = torch.jit._load_for_lite_interpreter(file_path)
                            num_rooms = result.size
                        else:
                            # Just get the length without loading all data
                            tmp_data = torch.load(file_path)
                            num_rooms = len(tmp_data)
                            del tmp_data  # Free memory
                    except:
                        # If the above fails, load the data and get its length
                        tmp_data = torch.load(file_path)
                        num_rooms = len(tmp_data)
                        del tmp_data  # Free memory
                
                # Store the count for caching
                area_room_counts[area_idx] = num_rooms
                
                # Build the index mapping
                for room_idx in range(num_rooms):
                    self.index_mapping.append((area_idx, room_idx))
            
            # Save the index mapping to cache if requested
            if use_cached_index:
                print(f"Caching index mapping to {index_cache_file}")
                with open(index_cache_file, 'wb') as f:
                    pickle.dump(area_room_counts, f)
        
        print(f'Dataset initialized with {len(self.index_mapping)} rooms from areas {self.area_indices}')

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        area_idx, room_idx = self.index_mapping[idx]
        
        if self.load_in_memory:
            # Get from memory
            if area_idx not in self.data_cache:
                # Load area if not in memory
                self.data_cache[area_idx] = torch.load(os.path.join(self.path, f's3dis{area_idx}.pt'))
                print(f'Area {area_idx} loaded into memory on demand')
            room_data = self.data_cache[area_idx][room_idx]
        else:
            # Load on demand
            area_data = torch.load(os.path.join(self.path, f's3dis{area_idx}.pt'))
            room_data = area_data[room_idx]
            del area_data  # Free memory
        
        # Convert lists to tensors if needed
        x = torch.tensor(room_data['x'], dtype=torch.float32)
        y = room_data['y']  # Keep as list of strings for now
        
        return {'x': x, 'y': y, 'area': area_idx, 'room_idx': room_idx}


def create_s3dis_dataloaders(data_path, batch_size=1, num_workers=4, load_in_memory=False, use_cached_index=True):
    """Create train and test data loaders for S3DIS dataset.
    Train: Areas 1-5, Test: Area 6
    """
    # Create the datasets
    train_dataset = S3DISDataset(
        path=data_path,
        area_indices=[1, 2, 3, 4, 5],
        load_in_memory=load_in_memory,
        use_cached_index=use_cached_index
    )
    
    test_dataset = S3DISDataset(
        path=data_path,
        area_indices=[6],
        load_in_memory=load_in_memory,
        use_cached_index=use_cached_index
    )
    
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Example usage
    data_path = './data'
    train_loader, test_loader = create_s3dis_dataloaders(
        data_path,
        batch_size=1,
        num_workers=0,  # Set to 0 for debugging
        load_in_memory=False,  # Set to False for memory efficiency
        use_cached_index=True  # Use cached index to avoid loading files during init
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Get and print a sample from the training dataset
    sample_batch = next(iter(train_loader))
    print("\nSample from training data:")
    print(f"Point cloud shape: {sample_batch['x'].shape}")
    print(f"Number of labels: {len(sample_batch['y'][0])}")
    print(f"Area: {sample_batch['area']}")
    
    # Count unique labels in the sample
    sample_labels = sample_batch['y'][0]
    unique_labels = set(sample_labels)
    print(f"Unique labels in sample: {unique_labels}")
    print(f"Label counts:")
    for label in unique_labels:
        count = sample_labels.count(label)
        print(f"  {label}: {count}")
