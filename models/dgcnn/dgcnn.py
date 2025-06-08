import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def knn(x, k):
    """
    Compute k-nearest neighbors for each point.
    Args:
        x: input features [B, F, N]
        k: number of neighbors
    Returns:
        idx: neighbor indices [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Get graph feature for dynamic graph CNN
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)  # Use only xyz for knn when using color features
    
    # Use CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    if dim9 == False:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature - x, x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class EdgeConv(nn.Module):
    """
    EdgeConv layer for dynamic graph CNN
    """
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNN(nn.Module):
    """
    DGCNN for semantic segmentation of point clouds
    Adapted for S3DIS dataset with memory efficiency considerations
    """
    def __init__(self, num_classes=13, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        
        # Edge convolution layers
        self.conv1 = EdgeConv(3, 64, k)  # Input: xyz coordinates
        self.conv2 = EdgeConv(64, 64, k)
        self.conv3 = EdgeConv(64, 64, k)
        self.conv4 = EdgeConv(64, 128, k)
        
        # Global feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv1d(320, emb_dims, kernel_size=1, bias=False),  # 64+64+64+128=320
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Segmentation head
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims + 320, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input point cloud [B, 6, N] (xyz + rgb) or [B, 3, N] (xyz only)
        Returns:
            logits: class logits [B, N, num_classes]
            feature: point features [B, emb_dims, N]
            trans_feat: transformation features (empty for compatibility)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Use only xyz coordinates for graph construction
        if x.size(1) == 6:
            xyz = x[:, :3, :]  # Extract xyz
        else:
            xyz = x
        
        # Edge convolutions
        x1 = self.conv1(xyz)  # [B, 64, N]
        x2 = self.conv2(x1)   # [B, 64, N]
        x3 = self.conv3(x2)   # [B, 64, N]
        x4 = self.conv4(x3)   # [B, 128, N]
        
        # Concatenate multi-scale features
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 320, N]
        
        # Global feature
        x5 = self.conv5(x_cat)  # [B, emb_dims, N]
        
        # Combine local and global features
        x_combined = torch.cat((x_cat, x5), dim=1)  # [B, 320+emb_dims, N]
        
        # Segmentation layers
        x6 = self.conv6(x_combined)  # [B, 512, N]
        x7 = self.conv7(x6)          # [B, 256, N]
        logits = self.conv8(x7)      # [B, num_classes, N]
        
        # Transpose to match expected format [B, N, num_classes]
        logits = logits.transpose(2, 1).contiguous()
        
        return logits, x5, None  # Return empty trans_feat for compatibility


class DGCNNWithColor(nn.Module):
    """
    DGCNN variant that also uses color information (RGB)
    """
    def __init__(self, num_classes=13, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNNWithColor, self).__init__()
        self.k = k
        self.num_classes = num_classes
        
        # Edge convolution layers for xyz
        self.conv1 = EdgeConv(3, 64, k)  # xyz
        self.conv2 = EdgeConv(64, 64, k)
        self.conv3 = EdgeConv(64, 64, k)
        self.conv4 = EdgeConv(64, 128, k)
        
        # Color processing
        self.color_conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),  # RGB
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Global feature extraction (384 = 320 from xyz + 64 from color)
        self.conv5 = nn.Sequential(
            nn.Conv1d(384, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Segmentation head
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims + 384, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input point cloud [B, 6, N] (xyz + rgb)
        Returns:
            logits: class logits [B, N, num_classes]
            feature: point features [B, emb_dims, N]
            trans_feat: transformation features (empty for compatibility)
        """
        if x.size(1) != 6:
            raise ValueError("DGCNNWithColor expects 6-channel input (xyz + rgb)")
            
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Split xyz and rgb
        xyz = x[:, :3, :]  # [B, 3, N]
        rgb = x[:, 3:6, :] # [B, 3, N]
        
        # Edge convolutions on xyz
        x1 = self.conv1(xyz)  # [B, 64, N]
        x2 = self.conv2(x1)   # [B, 64, N]
        x3 = self.conv3(x2)   # [B, 64, N]
        x4 = self.conv4(x3)   # [B, 128, N]
        
        # Process color information
        color_feat = self.color_conv(rgb)  # [B, 64, N]
        
        # Concatenate all features
        x_cat = torch.cat((x1, x2, x3, x4, color_feat), dim=1)  # [B, 384, N]
        
        # Global feature
        x5 = self.conv5(x_cat)  # [B, emb_dims, N]
        
        # Combine local and global features
        x_combined = torch.cat((x_cat, x5), dim=1)  # [B, 384+emb_dims, N]
        
        # Segmentation layers
        x6 = self.conv6(x_combined)  # [B, 512, N]
        x7 = self.conv7(x6)          # [B, 256, N]
        logits = self.conv8(x7)      # [B, num_classes, N]
        
        # Transpose to match expected format [B, N, num_classes]
        logits = logits.transpose(2, 1).contiguous()
        
        return logits, x5, None  # Return empty trans_feat for compatibility


def get_model(num_classes=13, use_color=True, **kwargs):
    """
    Factory function to get DGCNN model
    Args:
        num_classes: number of semantic classes
        use_color: whether to use RGB color information
        **kwargs: additional arguments (k, emb_dims, dropout)
    Returns:
        model: DGCNN model
    """
    if use_color:
        return DGCNNWithColor(num_classes=num_classes, **kwargs)
    else:
        return DGCNN(num_classes=num_classes, **kwargs)


def get_loss():
    """
    Get appropriate loss function for semantic segmentation
    """
    return nn.CrossEntropyLoss(ignore_index=-1)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with xyz only
    model_xyz = DGCNN(num_classes=13, k=20).to(device)
    x_xyz = torch.randn(2, 3, 1024).to(device)  # [B, 3, N]
    
    print("Testing DGCNN with xyz only:")
    print(f"Input shape: {x_xyz.shape}")
    
    logits, feat, _ = model_xyz(x_xyz)
    print(f"Output logits shape: {logits.shape}")
    print(f"Feature shape: {feat.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_xyz.parameters() if p.requires_grad):,}")
    
    # Test with xyz + rgb
    model_color = DGCNNWithColor(num_classes=13, k=20).to(device)
    x_color = torch.randn(2, 6, 1024).to(device)  # [B, 6, N]
    
    print("\nTesting DGCNN with xyz + rgb:")
    print(f"Input shape: {x_color.shape}")
    
    logits, feat, _ = model_color(x_color)
    print(f"Output logits shape: {logits.shape}")
    print(f"Feature shape: {feat.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_color.parameters() if p.requires_grad):,}")

