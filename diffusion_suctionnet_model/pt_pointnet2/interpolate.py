import torch
import torch.nn.functional as F

def three_nn_interpolate(unknown, known, known_feats, k=3):
    """
    纯PyTorch实现的三近邻插值
    Args:
        unknown: (B, N, 3) 需要插值的点
        known: (B, M, 3) 已知点
        known_feats: (B, M, C) 已知点特征
        k: int 近邻数
    Returns:
        interpolated_feats: (B, N, C)
    """
    dist = torch.cdist(unknown, known, p=2)  # (B, N, M)
    dist, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)  # (B, N, k)
    dist_recip = 1.0 / (dist + 1e-8)  # 防止除零
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)
    weight = dist_recip / norm  # (B, N, k)
    interpolated = torch.sum(
        F.embedding(idx, known_feats.transpose(1, 2)).transpose(2, 3) * weight.unsqueeze(-1), dim=2
    )
    return interpolated
