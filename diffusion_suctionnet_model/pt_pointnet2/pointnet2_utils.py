import torch
import warnings
from .ball_query import ball_query
from .group_points import group_points
from .interpolate import three_nn_interpolate
from .sampling import farthest_point_sample

# 球查询分组模块
class QueryAndGroup(torch.nn.Module):
    def __init__(self, radius, nsample, use_xyz=True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(new_xyz, xyz, self.radius, self.nsample)  # (B, S, nsample)
        grouped_xyz = group_points(xyz, idx)  # (B, S, nsample, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        if features is not None:
            grouped_features = group_points(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "不能既没有特征又不使用xyz作为特征！"
            new_features = grouped_xyz
        # (B, S, nsample, C)
        # 转换为 (B, C, S, nsample)
        return new_features.permute(0, 3, 1, 2).contiguous()

# 全局分组模块
class GroupAll(torch.nn.Module):
    def __init__(self, use_xyz=True):
        super().__init__()
        self.use_xyz = use_xyz
    def forward(self, xyz, new_xyz, features=None):
        grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
        if features is not None:
            grouped_features = features.unsqueeze(1)  # (B, 1, N, C)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        # (B, 1, N, C)
        return new_features.permute(0, 3, 1, 2).contiguous()

# 三近邻查找和插值
class ThreeNN:
    @staticmethod
    def apply(unknown, known):
        dist = torch.cdist(unknown, known, p=2)  # (B, n, m)
        dist, idx = torch.topk(dist, 3, dim=-1, largest=False, sorted=True)
        return dist, idx

class ThreeInterpolate:
    @staticmethod
    def apply(features, idx, weight):
        # features: (B, c, m), idx: (B, n, 3), weight: (B, n, 3)
        B, c, m = features.shape
        n = idx.shape[1]
        interpolated = torch.zeros(B, c, n, device=features.device)
        for i in range(3):
            interpolated += torch.gather(features, 2, idx[..., i].unsqueeze(1).expand(-1, c, -1)) * weight[..., i].unsqueeze(1)
        return interpolated
