import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processing.chunked_datasets import create_chunked_dataloaders
from models.PointNeXt import PointNeXt
from Training.train_model import train_model
from models.utils.common import SetAbstraction, InvResMLP, UnitPointNet, FeaturePropagation


class PointNeXt(nn.Module):
    """
    PointNeXt model for point cloud segmentation.
    """

    def __init__(self, part_classes: int):
        super().__init__()
        self.num_classes = part_classes

        self.mlp = UnitPointNet(6, [32])

        self.sa1 = SetAbstraction(1024, 0.1, 32 + 3, [64, 64])
        self.irmlp1 = InvResMLP(0.1, 64 + 3, 64, 32)

        self.sa2 = SetAbstraction(256, 0.1, 64 + 3, [128, 128])
        self.irmlp2 = InvResMLP(0.1, 128 + 3, 128, 32)

        self.fp2 = FeaturePropagation(128 + 64, [64, 64])
        self.fp1 = FeaturePropagation(64 + 32, [32, 32])
        self.fp0 = FeaturePropagation(32 + 3, [32, 32])

        self.drop = nn.Dropout(0.5)
        self.conv = nn.Conv1d(32, part_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        print(f"Input shape: {x.shape}")

        coords_0 = x[:, :3, :]
        features_0 = x[:, 3:, :]
        print(f"0 shape: {coords_0.shape} {features_0.shape}")

        features_1 = self.mlp(x)
        print(f"Features_0 shape after mlp: {features_1.shape}")

        coords_1 = coords_0.permute(0, 2, 1)
        features_1 = features_1.permute(0, 2, 1)
        print(f"0 shape after permute: {coords_1.shape} {features_1.shape}")


        coords_2, features_2 = self.sa1(coords_1, features_1)
        print(f"1 shape: {coords_2.shape} {features_2.shape}")

        coords_2, features_2 = self.irmlp1(coords_2, coords_2, features_2)
        print(f"1 shape after irmlp: {coords_2.shape} {features_2.shape}")


        coords_3, features_3 = self.sa2(coords_2, features_2)
        print(f"1 shape: {coords_3.shape} {features_3.shape}")

        coords_3, features_3 = self.irmlp2(coords_3, coords_3, features_3)
        print(f"1 shape after irmlp: {coords_3.shape} {features_3.shape}")


        print(f'before 2 fp: {coords_2.shape}, {coords_3.shape}, {features_2.shape}, {features_3.shape}')
        features_2 = self.fp2(coords_2, coords_3, features_2, features_3)
        print(f"Features_2 shape after 2 fp: {features_2.shape}")

        print(f'before 1 fp: {coords_1.shape}, {coords_2.shape}, {features_1.shape}, {features_2.shape}')
        features_1 = self.fp1(coords_1, coords_2, features_1, features_2)
        print(f"Features_1 shape after 1 fp: {features_1.shape}")

        coords_0 = coords_0.permute(0, 2, 1)
        features_0 = features_0.permute(0, 2, 1)
        print(f'before 0 fp: {coords_0.shape}, {coords_1.shape}, {features_0.shape}, {features_1.shape}')
        features_0 = self.fp0(coords_0, coords_1, features_0, features_1)
        print(f"Features_0 shape after 0 fp: {features_0.shape}")

        x = self.drop(features_0)
        x = x.permute(0, 2, 1)
        print(f"X shape before conv: {x.shape}")

        x = self.conv(x)
        print(f"X shape after conv: {x.shape}")

        x = x.permute(0, 2, 1)
        print(f"Shape after final permute: {x.shape}")

        x = F.log_softmax(x, dim=-1)
        print(f"Shape after log_softmax: {x.shape}")

        return x, None, None




class TmpDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, point_shape: tuple[int, int], label_shape: tuple[int, int]):
        self.length = length
        self.point_shape = point_shape
        self.label_shape = label_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        points = torch.randn(self.point_shape)
        labels = torch.randint(0, self.label_shape[1], self.label_shape)
        return points, labels

if __name__ == '__main__':
    DATA_DIR_PATH = "test_data"
    TRAINING_DIR_PATH = "saved_training"
    MODEL_SAVE_PATH = os.path.join(TRAINING_DIR_PATH, "models")
    TRAINING_HISTORY_PATH = os.path.join(TRAINING_DIR_PATH, "history")
    BATCH_SIZE = 2
    NUM_CLASSES = 13  # dla S3DIS
    s3dis_classes = [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter"
    ]
    os.makedirs(TRAINING_DIR_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(TRAINING_HISTORY_PATH, exist_ok=True)

    # Create dataloaders with optimized loading
    # UWAGA!! dany plik z mapowaniem: chunked_s3dis_index_mapping.pkl nie wspiera jeżeli kożystamy z podzbioru zbioru
    # danego na dysku - trzeba dać require_index_file=False
    train_loader, test_loader = create_chunked_dataloaders(
        DATA_DIR_PATH,
        batch_size=BATCH_SIZE,
        require_index_file=False
    )

    # train_loader = torch.utils.data.DataLoader(
    #     TmpDataset(100, (6, 1024), (1024, 13)),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True
    # )

    # for points, labels in train_loader:
    #     print(f"Points shape: {points.shape}")
    #     print(f"Labels shape: {labels.shape}")
    #     break

    # !!!! TYLKO DO TESTÓW - w przykładowych danych nie ma ich wystarczająco dużo aby zrobić zbiór testowy (walidacyjny)
    test_loader = train_loader  # usunąć w normalnym treningu
    model_name = 'nextTests'
    raw_model = PointNeXt(part_classes=NUM_CLASSES)

    model_trained = train_model(raw_model, train_loader, test_loader, s3dis_classes, print_records=True,
                                records_dir=TRAINING_HISTORY_PATH, records_filename=model_name, epochs=30, sampling=None, cut=1000)

    with open(os.path.join(MODEL_SAVE_PATH, f"{model_name}.pt"), "wb") as f:
        torch.save(model_trained.state_dict(), f)
