import torch
import os
import argparse
from tqdm import tqdm
import math

def split_s3dis_data(input_path, output_path, n_chunks=10):
    """
    Splits each S3DIS area file into smaller chunks while preserving area information
    
    Args:
        input_path: Path to directory containing s3dis*.pt files
        output_path: Path to directory to save chunked files
        n_chunks: Number of chunks to split each area into
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process each area file (s3dis1.pt through s3dis6.pt)
    print(f"Starting to split data into {n_chunks} chunks per area")
    
    for area_idx in range(1, 7):
        file_path = os.path.join(input_path, f's3dis{area_idx}.pt')
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
        
        print(f"Loading area {area_idx} data from {file_path}")
        try:
            data = torch.load(file_path)
            print(f"Area {area_idx} loaded. Contains {len(data)} rooms.")
            
            # Calculate chunk size (number of rooms per chunk)
            total_rooms = len(data)
            rooms_per_chunk = math.ceil(total_rooms / n_chunks)
            
            # Split data into chunks
            chunks = []
            for i in range(0, total_rooms, rooms_per_chunk):
                end_idx = min(i + rooms_per_chunk, total_rooms)
                chunks.append(data[i:end_idx])
            
            # Save each chunk with area information in the filename
            for i, chunk in enumerate(tqdm(chunks, desc=f"Saving area {area_idx} chunks")):
                # Format: s3dis{area_idx}_chunk{chunk_idx}.pt
                chunk_file = os.path.join(output_path, f's3dis{area_idx}_chunk{i+1}.pt')
                torch.save(chunk, chunk_file)
                
            print(f"Area {area_idx} split into {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing area {area_idx}: {str(e)}")
    
    print(f"Data splitting complete. Chunked files saved in {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Split S3DIS dataset into smaller chunks')
    parser.add_argument('--input_path', type=str, default='./data', 
                        help='Path to directory containing s3dis*.pt files')
    parser.add_argument('--output_path', type=str, default='./data_chunked', 
                        help='Path to directory to save chunked files')
    parser.add_argument('--n_chunks', type=int, default=10, 
                        help='Number of chunks to split each area into')
    
    args = parser.parse_args()
    
    # Print summary of what will be done
    print(f"Input directory: {args.input_path}")
    print(f"Output directory: {args.output_path}")
    print(f"Number of chunks per area: {args.n_chunks}")
    
    # Split the data
    split_s3dis_data(args.input_path, args.output_path, args.n_chunks)
    
    # Print information about the result
    print("\nSummary of chunked files:")
    if os.path.exists(args.output_path):
        files = [f for f in os.listdir(args.output_path) if f.startswith('s3dis') and f.endswith('.pt')]
        print(f"Total files created: {len(files)}")
        
        # Group by area
        area_counts = {}
        for file in files:
            area = file.split('_')[0]  # Extract s3dis1, s3dis2, etc.
            area_counts[area] = area_counts.get(area, 0) + 1
        
        for area, count in sorted(area_counts.items()):
            print(f"  {area}: {count} chunks")
    
if __name__ == '__main__':
    main()