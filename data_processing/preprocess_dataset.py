import torch
import os
import pickle
import math
from tqdm import tqdm

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter", "stairs"
]
BLOCK_SIZE = 1.0

def get_chunk_indices(input_dir: str) -> list[tuple[int, int]]:
    indices = [
        filename
            .replace('s3dis', '')
            .replace('chunk', '')
            .replace('.pt', '')
            .split('_')
        for filename in os.listdir(input_dir)
        if 'index' not in filename
    ]
    indices = [(int(area_index), int(chunk_index)) for area_index, chunk_index in indices]
    indices.sort()

    return indices


def one_hot_encode_labels(labels: list[str]) -> torch.Tensor:
    one_hot_labels = torch.zeros((len(labels), len(S3DIS_CLASSES)), dtype=torch.uint8)

    for i, label in enumerate(labels):
        j = S3DIS_CLASSES.index(label)
        if j < 0:
            raise ValueError(f'One of the points has an unkown label: {label}.')
        one_hot_labels[i, j] = 1

    return one_hot_labels


def get_block_coords(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    min_x = x.min().item()
    max_x = x.max().item()
    min_y = y.min().item()
    max_y = y.max().item()

    length_x = max_x - min_x
    offset_x = (math.ceil(length_x) - length_x) / 2
    length_y = max_y - min_y
    offset_y = (math.ceil(length_y) - length_y) / 2

    if offset_x < 1e-4:
        offset_x = 0
    if offset_y < 1e-4:
        offset_y = 0

    x_starts = torch.arange(min_x - offset_x, max_x + offset_x, BLOCK_SIZE)
    y_starts = torch.arange(min_y - offset_y, max_y + offset_y, BLOCK_SIZE)

    return x_starts, y_starts


def extract_block(points: torch.Tensor, labels: torch.Tensor, x_start: int, y_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (
        (points[:, 0] >= x_start) & (points[:, 0] < x_start + BLOCK_SIZE) &
        (points[:, 1] >= y_start) & (points[:, 1] < y_start + BLOCK_SIZE)
    )

    return points[mask], labels[mask]


def augment_points(points: torch.Tensor) -> torch.Tensor:
    augmented_points = torch.zeros((points.shape[0], 9), dtype=torch.float32)
    augmented_points[:, :6] = points

    min_x = points[:, 0].min().item()
    min_y = points[:, 1].min().item()
    max_z = points[:, 2].max().item() 
    min_z = points[:, 2].min().item()

    center = torch.tensor([
        min_x + BLOCK_SIZE / 2,
        min_y + BLOCK_SIZE / 2,
        min_z + (max_z - min_z) / 2
    ], dtype=torch.float32)

    augmented_points[:, 6:] = augmented_points[:, :3] - center

    return augmented_points


def preprocess_dataset(input_dir: str, output_dir: str):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f'Input directory {input_dir} does not exist.')

    os.makedirs(output_dir, exist_ok=True)

    chunk_indices = get_chunk_indices(input_dir)

    block_to_coords_mapping = []

    for area_index, chunk_index in tqdm(chunk_indices):
        os.makedirs(os.path.join(output_dir, f'area_{area_index}'), exist_ok=True)

        data = torch.load(os.path.join(input_dir, f's3dis{area_index}_chunk{chunk_index}.pt'))

        for room_index, room in enumerate(data, start=1):
            points = torch.tensor(room['x'], dtype=torch.float32)
            labels = one_hot_encode_labels(room['y'])

            x_starts, y_starts = get_block_coords(points[:, 0], points[:, 1])

            block_index = 1
            for x_start in x_starts:
                for y_start in y_starts:
                    block_to_coords_mapping.append({
                        'area': area_index,
                        'room': room_index,
                        'block': block_index,
                        'x_start': x_start,
                        'y_start': y_start
                    })

                    block_points, block_labels = extract_block(points, labels, x_start, y_start)

                    if block_points.shape[0] < 100:
                        print(f'Area {area_index}, Chunk: {chunk_index}, Room: {room_index}: Skipping small block with {block_points.shape[0]} points.')
                        block_index += 1
                        continue

                    augmented_points = augment_points(block_points)

                    torch.save((augmented_points, block_labels), os.path.join(output_dir, f'area_{area_index}', f'room{room_index:02d}_block{block_index:03d}.pt'))
                    block_index += 1

        del data

    print('Saving the block to coordinates mapping...')
    with open('block_to_coords_mapping.pkl', 'wb') as file:
        pickle.dump(block_to_coords_mapping, file)


if __name__ == '__main__':
    preprocess_dataset('data_chunked', 'S3DIS_blocks')
