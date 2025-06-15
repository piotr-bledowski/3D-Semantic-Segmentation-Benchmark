import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_processing.block_datasets import create_block_dataloaders
from models.PointNeXt.PointNeXt import PointNeXt
from models.PointNetpp.PointNetpp import PointNetpp
from models.PointNet.PointNet import PointNetSeg
from models.dgcnn.dgcnn import DGCNNWithColor
from Training.training import train_model
from Training.train_model import masked_onehot_cross_entropy
import os
import argparse
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Because of Cuda out of memory

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 2
TRAIN_SAMPLING = 4096
TEST_SAMPLING = None
TEST_AREAS = {6}
NUM_WORKERS = 2

LOG_INTERVAL = 20
LOG_DIR = 'saved_runs'
MODEL_DIR = 'saved_models'
DATA_DIR = 'S3DIS_blocks'

ALL_AREAS = {1, 2, 3, 4, 5, 6}
S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter", "stairs"
]
NUM_S3DIS_CLASSES = 14


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a selected model.")
    parser.add_argument("model", help="Name of the model to train.", choices=['PointNet', 'PointNet++', 'PointNeXt', 'DeepGraphCnn'])
    args = parser.parse_args()

    run_name = os.path.join(args.model, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_path = os.path.join(LOG_DIR, run_name)
    model_path = os.path.join(MODEL_DIR, run_name)

    os.makedirs(os.path.join(MODEL_DIR, args.model), exist_ok=True)

    logger = SummaryWriter(log_path)

    if args.model == 'PointNet':
        model = PointNetSeg(part_classes=NUM_S3DIS_CLASSES)
    elif args.model == 'PointNet++':
        model = PointNetpp(part_classes=NUM_S3DIS_CLASSES)
    elif args.model == 'PointNeXt':
        model  = PointNeXt(part_classes=NUM_S3DIS_CLASSES)
    elif args.model == 'DeepGraphCnn':
        model = DGCNNWithColor(num_classes=NUM_S3DIS_CLASSES)

    print(f'Starting training of model {args.model}.')

    train_loader, test_loader = create_block_dataloaders(
        data_dir=DATA_DIR,
        test_areas=TEST_AREAS,
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_sampling=TRAIN_SAMPLING,
        test_sampling=TEST_SAMPLING,
        train_shuffle=True,
        test_shuffle=False
    )

    print(f'Initialized train dataloader with areas {ALL_AREAS - TEST_AREAS}, and test dataloader with areas {TEST_AREAS}.')

    criterion = masked_onehot_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Using device {device}.')
    print('-' * 15)

    model = train_model(model, train_loader, test_loader, criterion, optimizer,
                        device, logger, num_epochs=NUM_EPOCHS, log_interval=LOG_INTERVAL)

    torch.save(model.state_dict(), model_path)

    print(f'Model saved to: {model_path}.')
    print(f'View logs with: tensorboard --logdir {LOG_DIR}')
