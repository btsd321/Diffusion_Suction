import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        使用迭代的最远点采样方法，选择一组具有最大最小距离的npoint个特征点

        参数
        ----------
        xyz : torch.Tensor
            (B, N, 3) 张量，N > npoint，表示输入点云的三维坐标
        npoint : int32
            需要采样的特征点数量

        返回
        -------
        torch.Tensor
            (B, npoint) 张量，包含采样得到的点的索引
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        根据索引从特征张量中收集指定的特征

        参数
        ----------
        features : torch.Tensor
            (B, C, N) 特征张量

        idx : torch.Tensor
            (B, npoint) 需要收集的特征索引

        返回
        -------
        torch.Tensor
            (B, C, npoint) 收集后的特征张量
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        查找unknown中每个点在known中的三个最近邻点

        参数
        ----------
        unknown : torch.Tensor
            (B, n, 3) 需要查找最近邻的点
        known : torch.Tensor
            (B, m, 3) 参考点集

        返回
        -------
        dist : torch.Tensor
            (B, n, 3) 到三个最近邻的欧氏距离
        idx : torch.Tensor
            (B, n, 3) 三个最近邻的索引
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
        对三个特征点进行加权线性插值

        参数
        ----------
        features : torch.Tensor
            (B, c, m) 需要插值的特征描述
        idx : torch.Tensor
            (B, n, 3) 目标点在features中的三个最近邻索引
        weight : torch.Tensor
            (B, n, 3) 插值权重

        返回
        -------
        torch.Tensor
            (B, c, n) 插值后的特征
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        反向传播

        参数
        ----------
        grad_out : torch.Tensor
            (B, c, n) 输出的梯度

        返回
        -------
        grad_features : torch.Tensor
            (B, c, m) 特征的梯度

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        根据索引对特征进行分组

        参数
        ----------
        features : torch.Tensor
            (B, C, N) 需要分组的特征张量
        idx : torch.Tensor
            (B, npoint, nsample) 每个分组的特征索引

        返回
        -------
        torch.Tensor
            (B, C, npoint, nsample) 分组后的特征张量
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        反向传播

        参数
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) 前向输出的梯度

        返回
        -------
        torch.Tensor
            (B, C, N) 特征的梯度
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        球查询操作，在指定半径内查找邻域点

        参数
        ----------
        radius : float
            球的半径
        nsample : int
            每个球内最多采样的点数
        xyz : torch.Tensor
            (B, N, 3) 所有点的三维坐标
        new_xyz : torch.Tensor
            (B, npoint, 3) 球心坐标

        返回
        -------
        torch.Tensor
            (B, npoint, nsample) 组成球查询的点的索引
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    使用球查询（ball query）方式进行分组

    参数
    ---------
    radius : float32
        球的半径
    nsample : int32
        每个球内最多采样的点数
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        参数
        ----------
        xyz : torch.Tensor
            (B, N, 3) 所有点的三维坐标
        new_xyz : torch.Tensor
            (B, npoint, 3) 球心坐标
        features : torch.Tensor
            (B, C, N) 点的特征描述

        返回
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) 分组后的特征张量
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "不能既没有特征又不使用xyz作为特征！"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    对所有点进行分组（全局分组）

    参数
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        参数
        ----------
        xyz : torch.Tensor
            (B, N, 3) 所有点的三维坐标
        new_xyz : torch.Tensor
            忽略该参数
        features : torch.Tensor
            (B, C, N) 点的特征描述

        返回
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) 分组后的特征张量
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
