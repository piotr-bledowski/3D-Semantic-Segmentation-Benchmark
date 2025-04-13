import torch
import os
import glob
from tqdm import tqdm
import pickle
import argparse

def precompute_chunk_index(data_path, force_recompute=False):
    """
    Precompute the index mapping for chunked S3DIS dataset and save it to a file.
    
    Args:
        data_path: Path to directory containing chunked s3dis*.pt files
        force_recompute: If True, recompute even if cache exists
    """
    # Path to the cached index file
    index_cache_file = os.path.join(data_path, 'chunked_s3dis_index_mapping.pkl')
    
    # Check if the cache already exists
    if os.path.exists(index_cache_file) and not force_recompute:
        print(f"Cache file {index_cache_file} already exists. Use --force to recompute.")
        return
    
    # Find all chunk files
    chunk_files = []
    for area_idx in range(1, 7):
        pattern = os.path.join(data_path, f's3dis{area_idx}_chunk*.pt')
        area_files = sorted(glob.glob(pattern))
        if not area_files:
            print(f"Warning: No chunk files found for area {area_idx} at {pattern}")
        chunk_files.extend(area_files)
    
    # Build index mapping
    index_mapping = []  # (chunk_file_idx, room_idx_in_chunk)
    
    print("Building index mapping from chunk files...")
    for chunk_idx, chunk_file in enumerate(tqdm(chunk_files, desc="Processing chunks")):
        try:
            # Load data to get the number of rooms
            chunk_data = torch.load(chunk_file)
            num_rooms = len(chunk_data)
            del chunk_data  # Free memory immediately
            
            # Add entries to index mapping
            for room_idx in range(num_rooms):
                index_mapping.append((chunk_idx, room_idx))
                
        except Exception as e:
            print(f"Error processing {chunk_file}: {str(e)}")
    
    # Save the index mapping to cache
    print(f"Caching index mapping to {index_cache_file}")
    cache_data = {
        'chunk_files': chunk_files,
        'index_mapping': index_mapping
    }
    with open(index_cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f'Successfully cached index mapping with {len(index_mapping)} rooms from {len(chunk_files)} chunk files')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute chunk index mapping for S3DIS dataset')
    parser.add_argument('--data_path', type=str, default='./data_chunked', 
                        help='Path to directory containing chunked s3dis*.pt files')
    parser.add_argument('--force', action='store_true', 
                        help='Force recomputation even if cache exists')
    
    args = parser.parse_args()
    
    precompute_chunk_index(args.data_path, args.force) 