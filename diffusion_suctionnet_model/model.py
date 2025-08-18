""" 
dsnet的Pytorch实现版本。
作者: HDT
"""
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union, Dict, Tuple, Optional

# 使用相对导入来导入包内模块
try:
    from . import pointnet2
    from .diffusers.schedulers.scheduling_ddim import DDIMScheduler
except ImportError:
    # 如果相对导入失败，回退到绝对导入（用于调试）
    import pointnet2
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
        try:
            if t.numel() == 1:
                feat = feat + self.time_embedding(t)[..., None] # feat( n ,16384,128   )   time_embedding(t) (128) 
            else:
                feat = feat + self.time_embedding(t)[..., None,]
            
            feat = feat + self.noise_embedding(noisy_image.permute(0, 2, 1))

            feat = self.channelattention(feat)
            feat = self.spatialattention(feat)

            ret = self.pred(feat)+noisy_image.permute(0, 2, 1)

            return ret
        except Exception as e:
            print(e)
            raise e

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
    def __init__(self, use_vis_branch, return_loss, pointnet_type: str = "cuda"):
        super().__init__()
        self.use_vis_branch = use_vis_branch
        self.loss_weights =  {
            'normal_flip_mask_head': 50.0,
            'wrench_scores_head': 50.0,
            'feasibility_scores_head': 50.0,
            'visibility_scores_head': 50.0,
        }
        self.bool_channels = [0, 2]  # normal_flip_mask, feasibility_scores
        self.continuous_channels = [1, 3]  # wrench_scores, visibility_scores
        self.return_loss = return_loss
        self.pointnet_type = pointnet_type
        self.debug_loss = True  # 启用调试模式，打印各个损失的数值

        backbone_config = {
            'npoint_per_layer': [4096,1024,256,64],
            'radius_per_layer': [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]],
            'input_feature_dims':3,
        }
        if self.pointnet_type == "cuda":
            self.backbone = pointnet2.Pointnet2MSGBackbone(**backbone_config)
        elif self.pointnet_type == "pt":
            from pt_pointnet2.pointnet2_backbone import Pointnet2MSGBackbone as PTPointnet2MSGBackbone
            self.backbone = PTPointnet2MSGBackbone(**backbone_config)
        else:
            raise ValueError(f"Unknown pointnet_type: {self.pointnet_type}, must be 'cuda' or 'pt'")
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
        针对混合数据类型（bool + 连续值）进行优化处理。

        参数:
            condit: 条件特征(如点云特征)
            gt: 真实标签 (B, N, 4) - [normal_flip_mask, wrench_scores, feasibility_scores, visibility_scores]

        返回:
            loss: 均方误差损失
        """
        # 采样噪声
        noise = torch.randn(gt.shape).to(gt.device)
        bs = gt.shape[0]

        # 分别处理不同类型的数据
        gt_norm = gt.clone()

        # Bool数据：0 -> -1, 1 -> 1
        for ch in self.bool_channels:
            gt_norm[:, :, ch] = (gt[:, :, ch] * 2 - 1) * self.bit_scale
        
        # 连续值数据：假设原本在[0,1]范围，映射到[-bit_scale, bit_scale]
        for ch in self.continuous_channels:
            gt_norm[:, :, ch] = (gt[:, :, ch] - 0.5) * 2 * self.bit_scale

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
        # 内存优化：清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        batch_size = inputs['point_clouds'].shape[0]
        num_point = inputs['point_clouds'].shape[1]
        
        # -----------------------------------------------------pointnet++提取堆叠场景点云
        input_points = inputs['point_clouds']  # torch.Size([4, 16384, 3])
        input_points = torch.cat((input_points, inputs['labels']['normals']), dim=2)
        features, global_features = self.backbone(input_points)
        
        # 释放不需要的变量
        del input_points
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.return_loss:  # 训练模式, 计算损失
            # 构建ground truth标签
            s1 = inputs['labels']['normal_flip_mask'].unsqueeze(-1)
            s2 = inputs['labels']['wrench_scores'].unsqueeze(-1)
            s3 = inputs['labels']['feasibility_scores'].unsqueeze(-1)
            s4 = inputs['labels']['visibility_scores'].unsqueeze(-1)
            gt = torch.cat((s1, s2, s3, s4), dim=2)
            
            # 释放临时变量
            del s1, s2, s3, s4
            
            pred_results = self.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(16384,4),
                features = features,
                num_inference_steps=self.diffusion_inference_steps,
            )

            ddim_loss, loss_weight_list, loss2_weight_list = self._compute_loss(features, pred_results, gt)
            
            # 训练模式：立即释放预测结果以节省内存
            del pred_results
            pred_results = None
            
            # 强制清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

    def _compute_loss(self, features, predict_results, labels):
        """
        计算各分支损失及总损失，包含动态权重平衡策略。

        参数:
            features: 点云特征
            predict_results: 预测结果 (B, N, 4) - [normal_flip_mask, wrench_scores, feasibility_scores, visibility_scores]
            labels: 标签数据 (B, N, 4) - [normal_flip_mask, wrench_scores, feasibility_scores, visibility_scores]

        返回:
            losses: 各分支损失及总损失的列表 [ddim_loss1, ddim_loss2]，
                   其中ddim_loss1和ddim_loss2已经按动态权重平衡
        """
        # 第一个损失：扩散模型训练损失
        ddim_loss1 = self.ddim_loss(features, labels)
        
        # 分离各个分支的预测结果和标签
        pred_normal_flip_mask = predict_results[:, :, 0]      # bool类型预测
        pred_wrench_scores = predict_results[:, :, 1]         # 连续值预测
        pred_feasibility_scores = predict_results[:, :, 2]    # bool类型预测
        pred_visibility_scores = predict_results[:, :, 3]     # 连续值预测
        
        true_normal_flip_mask = labels[:, :, 0]               # bool类型真值
        true_wrench_scores = labels[:, :, 1]                  # 连续值真值
        true_feasibility_scores = labels[:, :, 2]             # bool类型真值
        true_visibility_scores = labels[:, :, 3]              # 连续值真值
        
        # 对于bool类型数据，使用BCE损失
        normal_flip_mask_loss = F.binary_cross_entropy_with_logits(
            pred_normal_flip_mask, true_normal_flip_mask.float()
        )
        feasibility_scores_loss = F.binary_cross_entropy_with_logits(
            pred_feasibility_scores, true_feasibility_scores.float()
        )
        
        # 对于连续值数据，使用MSE损失
        wrench_scores_loss = F.mse_loss(pred_wrench_scores, true_wrench_scores)
        visibility_scores_loss = F.mse_loss(pred_visibility_scores, true_visibility_scores)
        
        # 获取权重
        normal_flip_mask_weight = self.loss_weights['normal_flip_mask_head']
        wrench_scores_weight = self.loss_weights['wrench_scores_head']
        feasibility_scores_weight = self.loss_weights['feasibility_scores_head']
        visibility_scores_weight = self.loss_weights['visibility_scores_head']
        
        # 动态权重平衡策略 - 根据损失值自动调整权重
        with torch.no_grad():
            # 计算各损失的相对大小
            losses_values = [
                normal_flip_mask_loss.item(),
                wrench_scores_loss.item(), 
                feasibility_scores_loss.item(),
                visibility_scores_loss.item()
            ]
            
            # 找到最大损失作为基准
            max_loss = max(losses_values)
            
            # 计算动态缩放因子，让所有损失在相似的数量级
            normal_scale = max_loss / (normal_flip_mask_loss.item() + 1e-8)
            wrench_scale = max_loss / (wrench_scores_loss.item() + 1e-8)
            feasibility_scale = max_loss / (feasibility_scores_loss.item() + 1e-8)
            visibility_scale = max_loss / (visibility_scores_loss.item() + 1e-8)
            
            # 限制缩放因子的范围，避免过度放大
            normal_scale = min(normal_scale, 20.0)
            wrench_scale = min(wrench_scale, 20.0)  
            feasibility_scale = min(feasibility_scale, 20.0)
            visibility_scale = min(visibility_scale, 20.0)
        
        # 加权组合损失 - 使用动态权重
        all_weights = (normal_flip_mask_weight * normal_scale + 
                         wrench_scores_weight * wrench_scale +
                         feasibility_scores_weight * feasibility_scale + 
                         visibility_scores_weight * visibility_scale)
        normal_flip_mask_final_weight = normal_flip_mask_weight * normal_scale / all_weights
        wrench_scores_final_weight = wrench_scores_weight * wrench_scale / all_weights
        feasibility_scores_final_weight = feasibility_scores_weight * feasibility_scale / all_weights
        visibility_scores_final_weight = visibility_scores_weight * visibility_scale / all_weights
        ddim_loss2 = (normal_flip_mask_loss * normal_flip_mask_final_weight +
                      wrench_scores_loss * wrench_scores_final_weight +
                      feasibility_scores_loss * feasibility_scores_final_weight +
                      visibility_scores_loss * visibility_scores_final_weight)
        loss2_weight_list = [normal_flip_mask_final_weight, wrench_scores_final_weight, 
                            feasibility_scores_final_weight, visibility_scores_final_weight]
        
        # Loss1和Loss2之间的动态权重平衡策略
        loss1_val = ddim_loss1.item()
        loss2_val = ddim_loss2.item()
        
        # 计算动态权重，让两个损失在相似的贡献度
        if loss1_val > 0 and loss2_val > 0:
            # 使用指数移动平均来稳定权重计算
            if not hasattr(self, 'loss1_ema'):
                self.loss1_ema = loss1_val
                self.loss2_ema = loss2_val
            else:
                self.loss1_ema = 0.9 * self.loss1_ema + 0.1 * loss1_val
                self.loss2_ema = 0.9 * self.loss2_ema + 0.1 * loss2_val
            
            # 计算平衡权重
            total_ema = self.loss1_ema + self.loss2_ema
            loss1_weight = total_ema / (2 * self.loss1_ema + 1e-8)  # 扩散损失权重
            loss2_weight = total_ema / (2 * self.loss2_ema + 1e-8)  # 重建损失权重
            
            # 限制权重范围，避免过度调整
            loss1_weight = max(0.5, min(3.0, loss1_weight))
            loss2_weight = max(0.5, min(3.0, loss2_weight))
        else:
            loss1_weight = 1.0
            loss2_weight = 1.0
        
        # 应用主损失权重
        ddim_loss1_weighted = loss1_weight * ddim_loss1
        ddim_loss2_weighted = loss2_weight * ddim_loss2
        
        # 调试信息输出
        if hasattr(self, 'debug_loss') and self.debug_loss:
            print(f"Individual losses - Normal(BCE): {normal_flip_mask_loss.item():.4f} (scale: {normal_scale:.2f}), "
                  f"Wrench(MSE): {wrench_scores_loss.item():.4f} (scale: {wrench_scale:.2f}), "
                  f"Feasibility(BCE): {feasibility_scores_loss.item():.4f} (scale: {feasibility_scale:.2f}), "
                  f"Visibility(MSE): {visibility_scores_loss.item():.4f} (scale: {visibility_scale:.2f})")
            print(f"Combined Loss2: {ddim_loss2.item():.4f}")
            
            # 主损失权重信息
            if hasattr(self, 'loss1_ema') and hasattr(self, 'loss2_ema'):
                print(f"Main Loss weights - L1: {loss1_weight:.3f}, L2: {loss2_weight:.3f} | "
                      f"EMA - L1: {self.loss1_ema:.4f}, L2: {self.loss2_ema:.4f}")
        loss_weight_list = [loss1_weight, loss2_weight]
        losses = [ddim_loss1_weighted, ddim_loss2_weighted]
        return losses, loss_weight_list, loss2_weight_list

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


