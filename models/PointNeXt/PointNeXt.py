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

    def __init__(self, part_classes: int, version: str = 'b'):
        super().__init__()

        self.num_classes = part_classes

        # For convenience and compatibility I get last layer size to put it into next layers
        # So you can freely change:
        #   - for SetAbstraction
        #       - point count
        #       - radius
        #       - mlp list
        #   - for InvResMLP
        #       - radius
        #       - count in group
        #   - for FeaturePropagation
        #       - mlp list

        self.mlp = UnitPointNet(6, [32])
        mlp_last = self.mlp.conv[-1].out_channels


        self.sa1 = SetAbstraction(1024, 0.1, mlp_last + 3, [32, 32, 64], grouping_norm=True) # what point count and inner layers?
        sa1_last = self.sa1.point_net.conv[-1].out_channels
        self.irmlp1 = InvResMLP(0.1, sa1_last + 3, sa1_last, 32)

        self.sa2 = SetAbstraction(256, 0.2, sa1_last + 3, [64, 64, 128], grouping_norm=True)
        sa2_last = self.sa2.point_net.conv[-1].out_channels
        self.irmlp2 = InvResMLP(0.1, sa2_last + 3, sa2_last, 32)
        self.irmlp2_1 = InvResMLP(0.2, sa2_last + 3, sa2_last, 32)

        self.sa3 = SetAbstraction(64, 0.4, sa2_last + 3, [128, 128, 256], grouping_norm=True)
        sa3_last = self.sa3.point_net.conv[-1].out_channels
        self.irmlp3 = InvResMLP(0.4, sa3_last + 3, sa3_last, 32)

        self.sa4 = SetAbstraction(16, 0.8, sa3_last + 3, [256, 256, 512], grouping_norm=True)
        sa4_last = self.sa4.point_net.conv[-1].out_channels
        self.irmlp4 = InvResMLP(0.8, sa4_last + 3, sa4_last, 16) # there are not enough points (16) to group them in groups of 32


        self.fp4 = FeaturePropagation(sa4_last + sa3_last, [256, 256])
        fp4_last = self.fp4.point_net.conv[-1].out_channels

        self.fp3 = FeaturePropagation(fp4_last + sa2_last, [256, 256])
        fp3_last = self.fp3.point_net.conv[-1].out_channels

        self.fp2 = FeaturePropagation(fp3_last + sa1_last, [256, 128])
        fp2_last = self.fp2.point_net.conv[-1].out_channels

        self.fp1 = FeaturePropagation(fp2_last + mlp_last, [128, 128, 128, 128])
        fp1_last = self.fp1.point_net.conv[-1].out_channels


        self.drop = nn.Dropout(0.5)
        self.conv = nn.Conv1d(fp1_last, part_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        print(f"Input shape: {x.shape}")

        coords_s = x[:, :3, :]
        features_s = x[:, 3:, :]
        print(f"0 shape: {coords_s.shape} {features_s.shape}")

        features_0 = self.mlp(x)
        print(f"features_s shape after mlp: {features_0.shape}")

        coords_0 = coords_s.permute(0, 2, 1)
        features_0 = features_0.permute(0, 2, 1)
        print(f"0 shape after permute: {coords_0.shape} {features_0.shape}")


        coords_1, features_1 = self.sa1(coords_0, features_0)
        print(f"1 shape: {coords_1.shape} {features_1.shape}")
        coords_1, features_1 = self.irmlp1(coords_1, coords_1, features_1)
        print(f"1 shape after irmlp1: {coords_1.shape} {features_1.shape}")

        coords_2, features_2 = self.sa2(coords_1, features_1)
        print(f"1 shape: {coords_2.shape} {features_2.shape}")
        coords_2, features_2 = self.irmlp2(coords_2, coords_2, features_2)
        print(f"1 shape after irmlp2: {coords_2.shape} {features_2.shape}")
        coords_2, features_2 = self.irmlp2_1(coords_2, coords_2, features_2)
        print(f"1 shape after irmlp2_1: {coords_2.shape} {features_2.shape}")

        coords_3, features_3 = self.sa3(coords_2, features_2)
        print(f"1 shape: {coords_3.shape} {features_3.shape}")
        coords_3, features_3 = self.irmlp3(coords_3, coords_3, features_3)
        print(f"1 shape after irmlp3: {coords_3.shape} {features_3.shape}")

        coords_4, features_4 = self.sa4(coords_3, features_3)
        print(f"1 shape: {coords_4.shape} {features_4.shape}")
        coords_4, features_4 = self.irmlp4(coords_4, coords_4, features_4)
        print(f"1 shape after irmlp4: {coords_4.shape} {features_4.shape}")


        print(f'before fp4: {coords_3.shape} {coords_4.shape} {features_3.shape} {features_4.shape}')
        features_3 = self.fp4(coords_3, coords_4, features_3, features_4)
        print(f"features_3 shape after fp4: {features_3.shape}")

        print(f'before fp3: {coords_2.shape} {coords_3.shape} {features_2.shape} {features_3.shape}')
        features_2 = self.fp3(coords_2, coords_3, features_2, features_3)
        print(f"features_2 shape after fp3: {features_2.shape}")

        print(f'before fp2: {coords_1.shape} {coords_2.shape} {features_1.shape} {features_2.shape}')
        features_1 = self.fp2(coords_1, coords_2, features_1, features_2)
        print(f"features_1 shape after fp2: {features_1.shape}")

        print(f'before fp1: {coords_0.shape} {coords_1.shape} {features_0.shape} {features_1.shape}')
        features_0 = self.fp1(coords_0, coords_1, features_0, features_1)
        print(f"features_0 shape after fp1: {features_0.shape}")


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
    #     TmpDataset(100, (6, 1000), (1000, 13)),
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
