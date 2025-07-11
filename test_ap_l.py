""" 
扩散吸取网络的AP测试脚本
作者: HDT
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
sys.path.append(ROOT_DIR)

import math
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import time
import copy
import open3d as o3d
import random
import h5py
import json

from torchvision import transforms
from torch.utils.data import DataLoader

from models.model import dsnet, load_checkpoint
from models.data.dataset_plus import DiffusionSuctionNetDataset
from models.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from suction_nms import nms_suction

# --------------------------------------------------------------------------------------------需要修改的参数
OUTPUT_DIR = 'train_ok1'
ROOT_DIR = "/opt/data/private/suctionnet-Packag/data_gen_for_package/diffusion_scution_net/sd-net-diff-main/good/train_ok1/logs/aaaa/train_ok1"
CHECKPOINT_PATH = os.path.join(ROOT_DIR) + "/checkpoint.tar"
TEST_CYCLE_RANGE = [990,992]  # 测试cycle编号范围(左闭右开)
TEST_SCENE_RANGE = [1,51]     # 测试scene编号范围(左闭右开)
DATASET_DIR = "/opt/data/private/suctionnet-Packag/data_gen_for_package_eval/h5_dataset/train"
# --------------------------------------------------------------------------------------------需要修改的参数

# --------------------------------------------------------------------------------------------默认参数
topk = 2000
BATCH_SIZE = 1
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
# 数据增强与转换
transforms = transforms.Compose(
    [
        PointCloudShuffle(),
        ToTensor()
    ]
)
# 构建测试数据集和数据加载器
test_dataset = DiffusionSuctionNetDataset(DATASET_DIR, TEST_CYCLE_RANGE, TEST_SCENE_RANGE, transforms=transforms, collect_names=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# --------------------------------------------------------------------------------------------默认参数

# --------------------------------------------------------------------------------------------模型加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = dsnet(True, False)
net.to(device)
net, _, start_epoch = load_checkpoint(CHECKPOINT_PATH, net, device)
# --------------------------------------------------------------------------------------------模型加载

def eval_one_epoch(loader):
    """
    单轮测试评估函数。
    遍历测试集, 逐样本推理并保存NMS后的吸取点结果。

    参数:
        loader: 测试数据加载器
    """
    net.eval()  # 设置为评估模式

    for batch_idx, batch_samples in enumerate(loader):
        # ---------------------------------------------------------------数据准备
        # 为点云添加高斯噪声, 模拟实际场景下的扰动
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['points'].shape)).float()
        input_points_with_noise = batch_samples['points'] + xyz_noise * 2
        # 标签整理
        labels = {
            'suction_or': batch_samples['suction_or'].to(device),
            'suction_seal_scores': batch_samples['suction_seal_scores'].to(device),
            'suction_wrench_scores': batch_samples['suction_wrench_scores'].to(device),
            'suction_feasibility_scores': batch_samples['suction_feasibility_scores'].to(device),
            'individual_object_size_lable': batch_samples['individual_object_size_lable'].to(device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(device),
            'labels': labels
        }
        # ---------------------------------------------------------------数据准备

        # -------------------------------------------------------------------------模型推理与NMS
        with torch.no_grad():
            time_start = time.time()
            pred_results, _ = net(inputs)  # 网络前向推理
            print("Forward time:", time.time()-time_start)

            # 提取各分支预测分数
            pred_suction_seal_scores = pred_results[0][:,0].cpu().numpy()
            pred_suction_wrench_scores = pred_results[0][:,1].cpu().numpy()
            pred_suction_feasibility_scores = pred_results[0][:,2].cpu().numpy()
            pred_individual_object_size_lable = pred_results[0][:,3].cpu().numpy()
            
            # 计算总分数(各分支分数相乘)
            all_scroe = pred_suction_seal_scores * pred_suction_wrench_scores * pred_suction_feasibility_scores * pred_individual_object_size_lable

            # 组装吸取点信息(分数、法向量、点坐标)
            suction_points = input_points_with_noise[0].cpu().numpy()
            suction_normals = batch_samples['suction_or'][0].cpu().numpy()
            suction_group = np.concatenate([all_scroe[..., np.newaxis], suction_normals, suction_points], axis=-1)
            # 非极大值抑制, 去除冗余吸取点
            suction_group_nms = nms_suction(suction_group, 0.02, 181.0 / 180 * np.pi)
            suction_group_nms = np.array(suction_group_nms)

        # -------------------------------------------------------------------------结果保存
        output_file_name = batch_samples['name']
        output_file_name_ = 'cycle_{:0>4}_'.format(output_file_name[0].item()) + "scene_" + "{:0>3}".format(output_file_name[1].item())
        save_path = os.path.join(OUTPUT_DIR, output_file_name_ + '.npz')
        np.savez(save_path, suction_group_nms)
        # exit()

if __name__ == "__main__":
    """
    主入口, 执行单轮评估
    """
    eval_one_epoch(test_loader)

