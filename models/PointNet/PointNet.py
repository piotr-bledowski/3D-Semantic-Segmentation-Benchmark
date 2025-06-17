import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Spatial Transformer Network for input or feature transform.
    Predicts a k x k transformation matrix.
    """
    def __init__(self, k=9):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        # initialize as identity
        init = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = self.fc3(x) + init
        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    Shared MLP + TNet modules. Outputs global features and per-point local features.
    """
    def __init__(self, global_feat=True, feature_transform=False, k = 9):
        super(PointNetEncoder, self).__init__()
        self.stn = TNet(k=k)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = TNet(k=64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        # x: (B, 9, N)
        batch_size, _, num_points = x.size()
        # input transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # feature transform
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # shared MLP
        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # symmetric function: max pooling
        x = torch.max(x, 2, keepdim=False)[0]

        # x.shape = (bs, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_feat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    """
    PointNet for classification tasks.
    """
    def __init__(self, k=40, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, k)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetSeg(nn.Module):
    """
    PointNet for segmentation tasks.
    """
    def __init__(self, part_classes=13, feature_transform=False):
        super(PointNetSeg, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, part_classes, 1)

    def forward(self, x):
        # x: (B, 6, N)
        x = torch.transpose(x, -1, -2)
        x, trans, trans_feat = self.feat(x)
        # x.shape = (bs, 1024 - cechy globalne calości + 64 - cechy indywidualne punktu, ilosc_punktow)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        # x.shape = (bs, ilosc_punktow, ilosc_klas)
        # softmax ręczny - bo mi nie działa :(
        x = torch.exp(x)
        sum_exp = torch.sum(x, keepdim=True, dim=-1)
        x = x / sum_exp
        return x#, trans, trans_feat