import torch

def ball_query(new_xyz, xyz, radius, nsample):
    """
    纯PyTorch实现的球查询
    Args:
        new_xyz: (B, S, 3) 查询中心点
        xyz: (B, N, 3) 所有点
        radius: float 球半径
        nsample: int 每个球最多采样点数
    Returns:
        group_idx: (B, S, nsample) 邻居索引
    """
    B, S, _ = new_xyz.shape
    N = xyz.shape[1]
    sqrdists = torch.cdist(new_xyz, xyz, p=2)  # (B, S, N)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    mask = sqrdists > radius
    group_idx[mask] = N  # 超出半径的点索引设为N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 取最近的nsample个
    group_idx[group_idx == N] = group_idx[:, :, 0].unsqueeze(-1).expand_as(group_idx)[group_idx == N]
    return group_idx
