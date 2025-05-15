import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sample(coords: torch.Tensor, C: int) -> torch.Tensor:
    """
    Samples points to serve as centroids for regions; uses iterative farthest point sampling. 

    Args:
        coords (torch.Tensor): Coordinates of points to sample, (B, N, 3).
        C (int): Number of points to sample.

    Returns:
        Tensor: Coordinates of sampled points, (B, C, 3).
    """
    B, N, _ = coords.shape
    device = coords.device

    point_indices = torch.zeros((B, C), dtype=torch.int)
    distance = torch.full((B, N), torch.inf, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.int, device=device)
    batch_indices = torch.arange(B, dtype=torch.int)

    for i in range(C):
        point_indices[:, i] = farthest
        centroid = coords[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.linalg.vector_norm(coords - centroid, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        _, farthest = torch.max(distance, -1)

    batch_indices = torch.arange(B, dtype=torch.int).repeat(C, 1).T
    return coords[batch_indices, point_indices, :]


def group(centroid_coords: torch.Tensor, coords: torch.Tensor, features: torch.Tensor, r: float, K: int) -> torch.Tensor:
    """
    For each centroid, groups points that are close to it, to create a local region; uses ball query.

    Args:
        centroid_coords (torch.Tensor): Coordinates of centroids, (B, C, 3).
        coords (torch.Tensor): Coordinates of all points, (B, N, 3).
        features (torch.Tensor): Features of all points, (B, N, D).
        r (float): Maximum distance to the centroid.
        K (int): Number of points in each region.

    Returns:
        torch.Tensor: Local regions of points, (B, C, K, 3+D)
    """
    B, N, _ = features.shape
    _, C, _ = centroid_coords.shape

    points_exp = coords.unsqueeze(1).expand(B, C, N, 3)   # (B, C, N, D)
    centroids_exp = centroid_coords.unsqueeze(2).expand(B, C, N, 3)  # (B, C, N, D)
    distances = ((points_exp - centroids_exp) ** 2).sum(dim=-1)  # (B, C, N)

    mask = distances <= r ** 2  # (B, C, N)
    distances[~mask] = torch.inf

    _, topk_indices = torch.topk(distances, K, dim=-1, largest=False, sorted=True)  # (B, C, K)
    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, C, K)

    grouped_coords = coords[batch_indices, topk_indices]
    grouped_features = features[batch_indices, topk_indices]

    grouped_coords -= centroid_coords.view(B, C, 1, 3) # Normalize the coordinates of points within regions. 

    return torch.cat([grouped_coords, grouped_features], dim=-1)


def reduce(x: torch.Tensor, type: str) -> torch.Tensor:
    """
    Similar to a pooling layer. Returns one point for each region.

    Args:
        x (torch.Tensor): Output of the neural network, (B, C, K, 3+D').
        type (str): Type of pooling, either 'max' or 'avg'.

    Returns:
        torch.Tensor: One point per region, (B, C, 3+D).
    """
    if type == 'max':
        return torch.max(x, dim=2)[0]

    if type == 'avg':
        return torch.mean(x, dim=2)[0]

    raise ValueError(f"'{type}' pooling not supported; use 'max' or 'avg'.")


def interpolate(points: torch.Tensor, coords_1: torch.Tensor, coords_2: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Scales points up to the reference size; uses inverse distance weighted average based on k nearest neighbors.

    Args:
        points (torch.Tensor): Points to interpolate, (B, M, D).
        coords_1 (torch.Tensor): Coordinates of reference points, (B, N, 3).
        coords_2 (torch.Tensor): Coordinates of points to interpolate, (B, M, 3).
        k (int): Number of neighbors to use in the algorithm.

    Returns:
        torch.Tensor: Interpolated points, (B, N, D).
    """
    B, N, _ = coords_1.shape
    _, M, _ = coords_2.shape

    coords_1_exp = coords_1.unsqueeze(2).expand(B, N, M, 3)  # (B, N, M, D)
    coords_2_exp = coords_2.unsqueeze(1).expand(B, N, M, 3)   # (B, N, M, D)
    distances = ((coords_2_exp - coords_1_exp) ** 2).sum(dim=-1)  # (B, N, M)

    topk_distances, topk_indices = torch.topk(distances, k, dim=-1, largest=False, sorted=True)  # (B, N, k)
    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, N, k)

    topk_points = points[batch_indices, topk_indices]

    weights = 1.0 / (topk_distances.view(B, N, k, 1) + 1e-9)
    norm = torch.sum(weights, dim=2, keepdim=True)

    return torch.sum(topk_points * weights / norm, dim=2)


class MiniPointNet(nn.Module):
    """
    MLP network with Relu for the Set Abstraction module.
    """

    def __init__(self, in_channels: int, mlps: list[int]):
        """
        Args:
            in_channel (int): Number of input channels.
            mlps (list[int]): Widths of each layer.
        """
        super().__init__()

        self.conv = nn.ModuleList()
        self.batch = nn.ModuleList()
        
        prev = in_channels
        for mlp in mlps:
            self.conv.append(nn.Conv2d(prev, mlp, (1, 1)))
            self.batch.append(nn.BatchNorm2d(mlp))
            prev = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, batch in zip(self.conv, self.batch):
            x = F.relu(batch(conv(x)))
        return x


class UnitPointNet(nn.Module):
    """
    MLP network with Relu for the Feature Propagation module.
    """

    def __init__(self, in_channels: int, mlps: list[int]):
        """
        Args:
            in_channel (int): Number of input channels.
            mlps (list[int]): Widths of each layer.
        """
        super().__init__()

        self.conv = nn.ModuleList()
        self.batch = nn.ModuleList()
        
        prev = in_channels
        for mlp in mlps:
            self.conv.append(nn.Conv1d(prev, mlp, 1))
            self.batch.append(nn.BatchNorm1d(mlp))
            prev = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, batch in zip(self.conv, self.batch):
            x = F.relu(batch(conv(x)))
        return x

class SetAbstraction(nn.Module):
    """
    Set Abstraction module of the PointNet++ architecture.
    """

    def __init__(self, C: int, radius: float, in_channels: int, mlps: list[int], K: int = 32, pooling_type: str = 'max'):
        """
        Args:
            C (int): Number of regions the input will be split into.
            radius (float): Radius of the sphere used to create local regions.
            in_channels (int): Number of input channels for the MLP network; should be set to the number of output features of the previous SA module + 3.
            mlps (list[int]): Widths of each layer in the MLP network.
            K (int): Number of points in each region.
            pooling_type (str): Type of pooling used after the MLP network, either 'max', or 'avg'.
        """
        super().__init__()
        self.point_net = MiniPointNet(in_channels, mlps)
        self.C = C
        self.radius = radius
        self.K = K
        self.pooling_type = pooling_type

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centroid_coords = sample(coords, self.C)
        features = group(centroid_coords, coords, features, self.radius, self.K)

        features = features.permute(0, 3, 1, 2)
        features = self.point_net(features)

        features = features.permute(0, 2, 3, 1)
        features = reduce(features, self.pooling_type)

        return centroid_coords, features


class FeaturePropagation(nn.Module):
    """
    Feature Propagation module of the PointNet++ architecture.
    """

    def __init__(self, in_channels: int, mlps: list[int]):
        """
        Args:
            in_channels (int): Number of input channels for the MLP network; should be set to the sum of the number of output features of the input modules.
            mlps (list[int]): Widths of each layer in the MLP network.
        """
        super().__init__()
        self.point_net = UnitPointNet(in_channels, mlps)
    
    def forward(self, coords_1: torch.Tensor, coords_2: torch.Tensor, features_1: torch.Tensor | None, features_2: torch.Tensor) -> torch.Tensor:
        features_2 = interpolate(features_2, coords_1, coords_2)

        if features_1 is not None:
            features = torch.cat([features_1, features_2], dim=-1)
        else:
            features = features_2

        features = features.permute(0, 2, 1)
        features = self.point_net(features)
        features = features.permute(0, 2, 1)

        return features
