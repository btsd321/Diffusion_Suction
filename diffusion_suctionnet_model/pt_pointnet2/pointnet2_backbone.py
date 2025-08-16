import torch
import torch.nn as nn
from .pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG

class Pointnet2Backbone(nn.Module):
    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        super().__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.SA_modules = nn.ModuleList([
            PointnetSAModule(npoint=npoint_per_layer[0], radius=radius_per_layer[0], nsample=32, mlp=[input_feature_dims, 32, 32, 64], use_xyz=use_xyz),
            PointnetSAModule(npoint=npoint_per_layer[1], radius=radius_per_layer[1], nsample=32, mlp=[64, 64, 64, 128], use_xyz=use_xyz),
            PointnetSAModule(npoint=npoint_per_layer[2], radius=radius_per_layer[2], nsample=32, mlp=[128, 128, 128, 256], use_xyz=use_xyz),
            PointnetSAModule(npoint=npoint_per_layer[3], radius=radius_per_layer[3], nsample=32, mlp=[256, 256, 256, 512], use_xyz=use_xyz)
        ])
        self.FP_modules = nn.ModuleList([
            PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]),
            PointnetFPModule(mlp=[256 + 64, 256, 128]),
            PointnetFPModule(mlp=[256 + 128, 256, 256]),
            PointnetFPModule(mlp=[512 + 256, 256, 256])
        ])
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        return l_features[0]

class Pointnet2MSGBackbone(nn.Module):
    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        super().__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.nscale = len(radius_per_layer[0])
        self.SA_modules = nn.ModuleList([
            PointnetSAModuleMSG(npoint=npoint_per_layer[0], radii=radius_per_layer[0], nsamples=[32]*self.nscale, mlps=[[input_feature_dims, 32, 32, 64] for _ in range(self.nscale)], use_xyz=use_xyz),
            PointnetSAModuleMSG(npoint=npoint_per_layer[1], radii=radius_per_layer[1], nsamples=[32]*self.nscale, mlps=[[64*self.nscale, 64, 64, 128] for _ in range(self.nscale)], use_xyz=use_xyz),
            PointnetSAModuleMSG(npoint=npoint_per_layer[2], radii=radius_per_layer[2], nsamples=[32]*self.nscale, mlps=[[128*self.nscale, 128, 128, 256] for _ in range(self.nscale)], use_xyz=use_xyz),
            PointnetSAModuleMSG(npoint=npoint_per_layer[3], radii=radius_per_layer[3], nsamples=[32]*self.nscale, mlps=[[256*self.nscale, 256, 256, 512] for _ in range(self.nscale)], use_xyz=use_xyz)
        ])
        self.FP_modules = nn.ModuleList([
            PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]),
            PointnetFPModule(mlp=[256 + 64*self.nscale, 256, 128]),
            PointnetFPModule(mlp=[512 + 128*self.nscale, 256, 256]),
            PointnetFPModule(mlp=[512*self.nscale + 256*self.nscale, 512, 512])
        ])
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        Global_features = l_features[-1]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        return l_features[0], Global_features
