import torch
from torch.utils.data import Dataset, DataLoader
import os

def collate_blocks(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates all samples in a batch to a single tensor, by padding the features and labels with zeros at the end.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor]]): Batch to be collated.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Collated and padded points, collated and padded labels and a tensor with the actual lengths of each sample.
    """

    B = len(batch)
    N = max([x.shape[0] for x, _ in batch])

    batch_points = torch.zeros((B, N, 9), dtype=torch.float32)
    batch_labels = torch.zeros((B, N, 14), dtype=torch.uint8)

    for batch_index, (sample_points, sample_labels) in enumerate(batch):
        num_points = sample_points.shape[0]
        batch_points[batch_index, :num_points] = sample_points
        batch_labels[batch_index, :num_points] = sample_labels

    sample_lengths = torch.tensor([x.shape[0] for x, _ in batch], dtype=torch.uint64)

    return batch_points, batch_labels, sample_lengths



class BlockS3DISDataset(Dataset):
    """ The S3DIS dataset split into 1m x 1m blocks. The values are already converted to tensors, augmented with normalized coordinates within each block and the labels are one-hot encoded. """

    def __init__(self, data_dir: str, included_areas: set[int], sampling: int | None = None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            included_areas (set[int]): Areas of the dataset to include in this split. Accepts values from [1, 6].
            sampling (int, optional): If provided, the dataset will randomly sample this number of values from each block.
        """

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Data directory "{data_dir}" does not exist.')

        if any([a < 1 or a > 6 for a in included_areas]):
            raise ValueError(f'Included areas can only contain values from the range [1, 6], got {included_areas}.')

        self.blocks = self._create_block_index(data_dir, included_areas)

        self.data_dir = data_dir
        self.sampling = sampling


    def _create_block_index(self, data_dir: str, included_areas: set[int]) -> torch.Tensor:
        """
        Creates the block index, based on the files found in the dataset directory.

        Args:
            data_dir (str): Dataset directory.
            included_areas (set[int]): Areas of the dataset to include.

        Returns:
            torch.Tensor: Area, room and block indices.
        """

        blocks = []
        for area_index in sorted(list(included_areas)):
            if not os.path.exists(os.path.join(data_dir, f'area_{area_index}')):
                raise FileNotFoundError(f'Directory for area {area_index} does not exist.')

            indices = [
                filename
                    .replace('room', '')
                    .replace('block', '')
                    .replace('.pt', '')
                    .split('_')
                for filename in os.listdir(os.path.join(data_dir, f'area_{area_index}'))
            ]

            if len(indices) == 0:
                raise FileNotFoundError(f'Directory for area {area_index} does not contain any blocks.')

            indices = [(area_index, int(room_index), int(block_index)) for room_index, block_index in indices]
            indices.sort()

            blocks += indices

        return torch.tensor(blocks, dtype=torch.uint16)


    def __len__(self) -> int:
        """
        Returns the number of blocks in the dataset.

        Returns:
            int: Number of blocks in the dataset.
        """

        return self.blocks.shape[0]


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single block from the dataset.

        Args:
            index (int): Index of the block.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Points (N, 9) and labels (N, 14) from the block.
        """

        area_index, room_index, block_index = self.blocks[index]

        points, labels = torch.load(os.path.join(self.data_dir, f'area_{area_index}', f'room{room_index:02d}_block{block_index:03d}.pt'))

        if self.sampling is not None:
            num_points = points.shape[0]

            if num_points > self.sampling:
                sampled_indices = torch.randperm(num_points)[:self.sampling]
            else:
                sampled_indices = torch.randint(num_points, (self.sampling,))

            points = points[sampled_indices]
            labels = labels[sampled_indices]

        return points, labels


def create_block_dataloaders(
        data_dir: str,
        test_areas: set[int],
        batch_size: int = 4,
        num_workers: int = 4,
        train_sampling: int | None = 4096,
        test_sampling: int | None = None,
        train_shuffle: bool = True,
        test_shuffle: bool = False) -> tuple[DataLoader, DataLoader]:
    """
    Creates train and test dataloaders for the block S3DIS dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        test_areas (set[int]): Areas to include in the test loader; the remaining ones will be included in the train loader.
        batch_size (int): Batch size for train and test dataloaders.
        num_workers (int): Number of cpu workers to use.
        train_sampling (int, optional): If provided, the train dataloader will return this number of randomly sampled points from each block.
        test_sampling (int, optional): If provided, the train dataloader will return this number of randomly sampled points from each block.
        train_shuffle (bool): Whether to shuffle the training set, or not.
        test_shuffle (bool): Whether to shuffle the test set, or not.

    Returns:
        tuple[Dataloader, Dataloader]: Train and test dataloaders.
    """

    areas = {1, 2, 3, 4, 5, 6}

    train_dataset = BlockS3DISDataset(data_dir, areas - test_areas, train_sampling)
    test_dataset = BlockS3DISDataset(data_dir, test_areas, test_sampling)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=collate_blocks
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=collate_blocks
    )

    return train_dataloader, test_dataloader
