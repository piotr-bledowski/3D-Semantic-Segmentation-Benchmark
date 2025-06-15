import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.common import SetAbstraction, FeaturePropagation

class PointNetpp(nn.Module):
    """
    PointNet++ model for semantic segmentation. This is the single scale grouping (ssg) variant.
    """

    def __init__(self, part_classes: int):
        super().__init__()

        self.sa1 = SetAbstraction(1024, 0.1, 9, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 128 + 3, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 256 + 3, [256, 256, 512])

        self.fp4 = FeaturePropagation(512 + 256, [256, 256])
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128, 128])

        self.drop = nn.Dropout(0.5)
        self.conv = nn.Conv1d(128, part_classes, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        # x = x.permute(0, 2, 1)
        coords_0 = x[:, :, :3]
        features_0 = x[:, :, 3:]

        coords_1, features_1 = self.sa1(coords_0, features_0)
        coords_2, features_2 = self.sa2(coords_1, features_1)
        coords_3, features_3 = self.sa3(coords_2, features_2)
        coords_4, features_4 = self.sa4(coords_3, features_3)

        features_3 = self.fp4(coords_3, coords_4, features_3, features_4)
        features_2 = self.fp3(coords_2, coords_3, features_2, features_3)
        features_1 = self.fp2(coords_1, coords_2, features_1, features_2)
        features_0 = self.fp1(coords_0, coords_1, None, features_1)

        x = self.drop(features_0)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = F.log_softmax(x, dim=-1)

        return x
