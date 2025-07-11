""" 
dsnet的Pytorch实现版本。
作者: HDT
"""
import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

import torch
import torch.nn as nn
import numpy as np
import backbone2
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional

class SpatialAttention(nn.Module):
    """
    空间注意力机制模块。
    通过对输入特征在通道维度做平均池化和最大池化, 拼接后经过卷积和sigmoid激活, 生成空间注意力权重, 对输入特征进行加权。
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class ChannelAttention(nn.Module):
    """
    通道注意力机制模块。
    通过全局平均池化和最大池化, 经过两层卷积和激活, 生成通道注意力权重, 对输入特征进行加权。
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class ScheduledCNNRefine(nn.Module):
    """
    带有噪声和时间步嵌入的卷积细化模块。
    用于扩散模型的去噪预测, 支持噪声特征和时间步特征的融合, 并集成通道和空间注意力机制。
    """
    def __init__(self, channels_in = 128, channels_noise = 4, **kwargs):
        super().__init__(**kwargs)
        # 噪声嵌入网络, 将噪声特征映射到与主特征相同的通道数
        self.noise_embedding = nn.Sequential(
            nn.Conv1d(channels_noise, 64, 1),
            nn.GroupNorm(4, 64),
            # 不能用batch norm, 会统计输入方差, 方差会不停的变
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(4, 128),
            nn.ReLU(True),
            nn.Conv1d(128, channels_in, 1),
        )

        # 时间步嵌入, 最大支持1280个时间步
        self.time_embedding = nn.Embedding(1280, channels_in)

        # 主预测网络
        self.pred = nn.Sequential(
            nn.Conv1d(channels_in, 64, 1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(4, 128),
            nn.ReLU(True),
            nn.Conv1d(128, channels_noise, 1),
        )

        self.channelattention = ChannelAttention(128)
        self.spatialattention = SpatialAttention()

    def forward(self, noisy_image, t, feat):
        """
        前向传播, 融合噪声、时间步和主特征, 输出去噪预测。

        参数:
            noisy_image: 输入噪声图像 (B, N, C_noise)
            t: 时间步 (B,) 或标量
            feat: 主特征 (B, N, C_feat)

        返回:
            ret: 去噪预测 (B, C_noise, N)
        """
        if t.numel() == 1:
            feat = feat + self.time_embedding(t)[..., None] # feat( n ,16384,128   )   time_embedding(t) (128) 
        else:
            feat = feat + self.time_embedding(t)[..., None,]
        feat = feat + self.noise_embedding(noisy_image.permute(0, 2, 1))

        feat = self.channelattention(feat)
        feat = self.spatialattention(feat)

        ret = self.pred(feat)+noisy_image.permute(0, 2, 1)

        return ret

class CNNDDIMPipiline:
    '''
    DDIM采样推理流程封装类。
    用于扩散模型的采样过程, 支持自定义步数、噪声、特征输入等。
    '''
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            features,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            **kwargs,
    ) -> Union[Dict, Tuple]:
        """
        执行DDIM采样过程, 生成最终预测结果。

        参数:
            batch_size: 批量大小
            device: 设备
            dtype: 数据类型
            shape: 输出形状(不含batch维)
            features: 主特征输入
            generator: 随机数生成器
            eta: 采样噪声系数
            num_inference_steps: 采样步数

        返回:
            image: 采样得到的最终结果 (B, N, C)
        """
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # 初始化高斯噪声作为采样起点
        image_shape = (batch_size, *shape)
        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)

        # 设置采样步数
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # 1. 预测噪声
            model_output = self.model(image, t.to(device), features)
            model_output = model_output.permute(0, 2, 1)
            # 2. 反向采样一步
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']

        return image

class dsnet(nn.Module):
    """
    dsnet主网络类, 集成了点云特征提取、扩散模型、损失计算等功能。
    支持训练和推理两种模式。
    """
    def __init__(self, use_vis_branch, return_loss):
        super().__init__()
        self.use_vis_branch = use_vis_branch
        self.loss_weights =  {
                                'suction_seal_scores_head': 50.0, 
                                'suction_wrench_scores_head': 50.0,
                                'suction_feasibility_scores_head': 50.0,
                                'individual_object_size_lable_head': 50.0,
                                }
        self.return_loss = return_loss
        
        backbone_config = {
            'npoint_per_layer': [4096,1024,256,64],
            'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]],
            'input_feature_dims':3,
        }
        self.backbone = backbone2.Pointnet2MSGBackbone(**backbone_config)
        backbone_feature_dim = 128
        
        # add diffusion
        self.model = ScheduledCNNRefine(channels_in=backbone_feature_dim, channels_noise=4 )
        self.diffusion_inference_steps = 20
        num_train_timesteps=1000
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)

        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        self.bit_scale = 0.5

    def ddim_loss(self, condit, gt,):
        """
        计算DDIM扩散模型的损失(MSE), 用于训练阶段。

        参数:
            condit: 条件特征(如点云特征)
            gt: 真实标签

        返回:
            loss: 均方误差损失
        """
        # 采样噪声
        noise = torch.randn(gt.shape).to(gt.device)
        bs = gt.shape[0]

        gt_norm = (gt - 0.5) * 2 * self.bit_scale

        # 随机采样每个样本的时间步
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt.device).long()
        # 前向扩散过程, 添加噪声
        noisy_images = self.scheduler.add_noise(gt_norm, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, condit)
        noise_pred = noise_pred.permute(0, 2, 1)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def forward(self, inputs):
        """
        网络前向推理或训练。

        参数:
            inputs: 输入字典, 包含点云和标签

        返回:
            pred_results: 推理结果或None
            ddim_loss: 损失(训练时返回, 否则为None)
        """
        batch_size = inputs['point_clouds'].shape[0]
        num_point = inputs['point_clouds'].shape[1]
        
        # -----------------------------------------------------pointnet++提取堆叠场景点云
        input_points = inputs['point_clouds']  # torch.Size([4, 16384, 3])
        input_points = torch.cat((input_points, inputs['labels']['suction_or']), dim=2)
        features, global_features = self.backbone(input_points)

        if self.return_loss:  # 训练模式, 计算损失
            s1 = inputs['labels']['suction_seal_scores'].unsqueeze(-1)
            s2 = inputs['labels']['suction_wrench_scores'].unsqueeze(-1)
            s3 = inputs['labels']['suction_feasibility_scores'].unsqueeze(-1)
            s4 = inputs['labels']['individual_object_size_lable'].unsqueeze(-1)
            gt = torch.cat((s1, s2, s3, s4), dim=2)
            
            pred_results = self.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(16384,4),
                features = features,
                num_inference_steps=self.diffusion_inference_steps,
            )

            # self-diff
            # ddim_loss1 = self.ddim_loss(features, pred_results)
            ddim_loss1 = self.ddim_loss(features, gt)

            ddim_loss2 = F.mse_loss(pred_results, gt)
            ddim_loss = [ddim_loss1, ddim_loss2]
            
            pred_results = None
        else:  # 推理模式
            pred_results = self.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(16384,4),
                features = features,
                num_inference_steps=self.diffusion_inference_steps,
            )
            ddim_loss = None
        return pred_results, ddim_loss

    def visibility_loss(self, pred_vis, vis_label):
        """
        计算可见性损失(L1损失)。

        参数:
            pred_vis: 预测值
            vis_label: 标签

        返回:
            loss: 平均绝对误差
        """
        loss = torch.mean( torch.abs(pred_vis - vis_label) )
        return loss

    def _compute_loss(self, preds_flatten, labels):
        """
        计算各分支损失及总损失。

        参数:
            preds_flatten: 预测结果(各分支)
            labels: 标签字典

        返回:
            losses: 各分支损失及总损失的字典
        """
        batch_size, num_point = labels['suction_seal_scores'].shape[0:2]
        suction_seal_scores_label_flatten = labels['suction_seal_scores'].view(batch_size * num_point)  # (B*N,)
        suction_wrench_scores_flatten = labels['suction_wrench_scores'].view(batch_size * num_point)  # (B*N,)
        suction_feasibility_scores_label_flatten = labels['suction_feasibility_scores'].view(batch_size * num_point)  # (B*N,)
        individual_object_size_lable_flatten = labels['individual_object_size_lable'].view(batch_size * num_point)  # (B*N,)
        
        pred_suction_seal_scores, pred_suction_wrench_scores, pred_suction_feasibility_scores,pred_individual_object_size_lable = preds_flatten
        
        losses = dict()
        losses['suction_seal_scores_head'] = self.visibility_loss(pred_suction_seal_scores, suction_seal_scores_label_flatten) * self.loss_weights['suction_seal_scores_head'] 
        losses['suction_wrench_scores_head'] = self.visibility_loss(pred_suction_wrench_scores, suction_wrench_scores_flatten) * self.loss_weights['suction_wrench_scores_head'] 
        losses['suction_feasibility_scores_head'] = self.visibility_loss(pred_suction_feasibility_scores, suction_feasibility_scores_label_flatten) * self.loss_weights['suction_feasibility_scores_head'] 
        losses['individual_object_size_lable_head'] = self.visibility_loss(pred_individual_object_size_lable, individual_object_size_lable_flatten) * self.loss_weights['individual_object_size_lable_head'] 
        losses['total'] = losses['suction_seal_scores_head'] + losses['suction_wrench_scores_head'] + losses['suction_feasibility_scores_head'] + losses['individual_object_size_lable_head'] 
        
        return losses

    def _build_head(self, nchannels):
        """
        构建多层1D卷积预测头。

        参数:
            nchannels: 通道数列表

        返回:
            head: nn.Sequential预测头
        """
        assert len(nchannels) > 1
        num_layers = len(nchannels) - 1

        head = nn.Sequential()
        for idx in range(num_layers):
            if idx != num_layers - 1:
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
                head.add_module( "bn_%d"%(idx+1), nn.BatchNorm1d(nchannels[idx+1]))
                head.add_module( "relu_%d"%(idx+1), nn.ReLU())
            else:   # 最后一层不加BN和ReLU
                head.add_module( "conv_%d"%(idx+1), nn.Conv1d(nchannels[idx], nchannels[idx+1], 1))
        return head

# 网络保存与加载辅助函数
def load_checkpoint(checkpoint_path, net, map_location=None,optimizer=None):
    """ 
    加载网络和优化器的断点。

    参数:
        checkpoint_path: 断点文件路径
        net: torch.nn.Module实例
        optimizer: torch.optim.Optimizer实例或None
        map_location: 加载设备

    返回:
        net: 加载参数后的网络
        optimizer: 加载参数后的优化器
        start_epoch: 起始epoch
    """
    checkpoint = torch.load(checkpoint_path,map_location=map_location)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    return net, optimizer, start_epoch

def save_checkpoint(checkpoint_path, current_epoch, net, optimizer, loss):
    """ 
    保存网络和优化器的断点。

    参数:
        checkpoint_path: 保存路径
        current_epoch: 当前epoch编号
        net: torch.nn.Module实例
        optimizer: torch.optim.Optimizer实例
        loss: 当前损失
    """
    save_dict = {'epoch': current_epoch+1, # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
    try: # 如果使用了nn.DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    torch.save(save_dict, checkpoint_path)

def save_pth(pth_path, current_epoch, net, optimizer, loss):
    """
    保存网络和优化器的断点为.pth文件。

    参数:
        pth_path: 保存路径(不含后缀)
        current_epoch: 当前epoch编号
        net: torch.nn.Module实例
        optimizer: torch.optim.Optimizer实例
        loss: 当前损失
    """
    save_dict = {'epoch': current_epoch+1, # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }
    try: # 如果使用了nn.DataParallel
        save_dict['model_state_dict'] = net.module.state_dict()
    except:
        save_dict['model_state_dict'] = net.state_dict()
    torch.save(save_dict, pth_path + '.pth')


