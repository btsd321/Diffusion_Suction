import torch

def group_points(points, idx):
    """
    纯PyTorch实现的分组操作
    Args:
        points: (B, N, C) 点云特征
        idx: (B, S, nsample) 分组索引
    Returns:
        grouped_points: (B, S, nsample, C)
    """
    B = points.shape[0]
    S = idx.shape[1]
    nsample = idx.shape[2]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1).repeat(1, S, nsample)
    return points[batch_indices, idx, :]
