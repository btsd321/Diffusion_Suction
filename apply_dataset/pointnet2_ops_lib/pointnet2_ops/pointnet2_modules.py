from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    # 构建共享的多层感知机（MLP），可选是否使用批归一化
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        参数说明
        ----------
        xyz : torch.Tensor
            (B, N, 3) 代表特征点的三维坐标
        features : torch.Tensor
            (B, C, N) 代表特征点的特征描述

        返回值
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) 新采样点的三维坐标
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) 新采样点的特征描述
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample) 分组后的特征

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample) 经过MLP处理
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1) 在每个分组内做最大池化
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint) 去掉最后一维

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""PointNet多尺度分组的集合抽象层

    参数说明
    ----------
    npoint : int
        采样点数量
    radii : list of float32
        每个分组的半径列表
    nsamples : list of int32
        每个球查询的采样点数
    mlps : list of list of int32
        每个尺度下MLP的结构
    bn : bool
        是否使用批归一化
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""PointNet集合抽象层（单尺度分组）

    参数说明
    ----------
    npoint : int
        采样点数量
    radius : float
        球查询半径
    nsample : int
        球查询内采样点数
    mlp : list
        MLP结构
    bn : bool
        是否使用批归一化
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""特征传播模块，将已知点集的特征传播到未知点集

    参数说明
    ----------
    mlp : list
        PointNet模块参数
    bn : bool
        是否使用批归一化
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        参数说明
        ----------
        unknown : torch.Tensor
            (B, n, 3) 未知点的三维坐标
        known : torch.Tensor
            (B, m, 3) 已知点的三维坐标
        unknow_feats : torch.Tensor
            (B, C1, n) 需要传播特征的未知点特征
        known_feats : torch.Tensor
            (B, C2, m) 需要传播的已知点特征

        返回值
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) 未知点的新特征
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n) 拼接特征
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
