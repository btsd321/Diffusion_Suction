import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import QueryAndGroup, GroupAll, ThreeNN, ThreeInterpolate

class SharedMLP(nn.Sequential):
    def __init__(self, mlp_spec, bn=True):
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i+1], 1))
            if bn:
                layers.append(nn.BatchNorm2d(mlp_spec[i+1]))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
    def forward(self, xyz, features=None):
        new_features_list = []
        new_xyz = None
        if self.npoint is not None:
            fps_idx = self.furthest_point_sample(xyz, self.npoint)
            new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)
    @staticmethod
    def furthest_point_sample(xyz, npoint):
        # xyz: (B, N, 3)
        from .sampling import farthest_point_sample
        return farthest_point_sample(xyz, npoint)

class PointnetSAModuleMSG(_PointnetSAModuleBase):
    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(SharedMLP(mlp_spec, bn=bn))

class PointnetSAModule(PointnetSAModuleMSG):
    def __init__(self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz)

class PointnetFPModule(nn.Module):
    def __init__(self, mlp, bn=True):
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn)
    def forward(self, unknown, known, unknow_feats, known_feats):
        if known is not None:
            dist, idx = ThreeNN.apply(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = ThreeInterpolate.apply(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0:2] + [unknown.size(1)]))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)
