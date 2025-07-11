'''
作者: HDT
'''

import os
import sys
from joblib import Parallel, delayed
import numpy as np
import torch
import h5py
import open3d as o3d
import torch.utils.data as data
from torchvision import transforms

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
sys.path.append(FILE_DIR)

from pointcloud_transforms import PointCloudShuffle, ToTensor
from torch.utils.data import DataLoader

def collect_cycle_obj_sence_dir(data_dir, cycle_range, scene_range):
    """
    收集指定cycle和scene范围内所有h5数据文件的路径。

    参数:
        data_dir: 数据集根目录
        cycle_range: tuple, cycle编号范围(起始, 结束), 左闭右开
        scene_range: tuple, scene编号范围(起始, 结束), 左闭右开

    返回:
        dirs: 包含所有目标h5文件路径的列表
    """
    dirs = []
    for cycle_id in range(cycle_range[0], cycle_range[1]):
        for scene_id in range(scene_range[0], scene_range[1]):
            dirs.append(os.path.join(
                data_dir,
                'cycle_{:0>4}'.format(cycle_id),
                '{:0>3}.h5'.format(scene_id)
            ))
    return dirs

def load_dataset_by_cycle_layer(dir, mode='train', collect_names=False):
    """
    加载单个h5文件的数据, 并转换为numpy数组。

    参数:
        dir: h5文件路径
        mode: 模式(未使用, 保留接口)
        collect_names: 是否收集名称(未使用, 保留接口)

    返回:
        dataset: 字典, 包含点云及吸取相关标签
    """
    num_point_in_h5 = 16384

    f = h5py.File(dir)
    points = f['points'][:].reshape(num_point_in_h5, 3) * 1000  # 单位转换为毫米
    suction_or = f['suction_or'][:].reshape(num_point_in_h5, 3)
    suction_seal_scores = f['suction_seal_scores'][:]
    suction_wrench_scores = f['suction_wrench_scores'][:]
    suction_feasibility_scores = f['suction_feasibility_scores'][:]
    individual_object_size_lable = f['individual_object_size_lable'][:]

    dataset = {
        'points': points,
        'suction_or': suction_or,
        'suction_seal_scores': suction_seal_scores,
        'suction_wrench_scores': suction_wrench_scores,
        'suction_feasibility_scores': suction_feasibility_scores,
        'individual_object_size_lable': individual_object_size_lable,
    }

    return dataset

class DiffusionSuctionNetDataset(data.Dataset):
    """
    吸取点云数据集类, 继承自torch.utils.data.Dataset。
    支持按cycle和scene范围批量加载h5数据, 支持数据变换和样本名称收集。

    参数:
        data_dir: 数据集根目录
        cycle_range: tuple, cycle编号范围
        scene_range: tuple, scene编号范围
        mode: 模式(如'train'或'test')
        transforms: 数据增强或转换操作
        collect_names: 是否收集样本名称信息
    """
    def __init__(self, data_dir, cycle_range, scene_range, mode='train', transforms=None, collect_names=False):
        self.mode = mode
        self.collect_names = collect_names
        self.transforms = transforms
        # 收集所有目标h5文件路径
        self.dataset_dir = collect_cycle_obj_sence_dir(data_dir, cycle_range, scene_range)

    def __len__(self):
        """
        返回数据集样本总数(即h5文件数量)。
        """
        return len(self.dataset_dir)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本, 并进行必要的转换和信息收集。

        参数:
            idx: 样本索引

        返回:
            sample: 字典, 包含点云、标签及可选的名称信息
        """
        # 加载h5文件的数据
        dataset = load_dataset_by_cycle_layer(self.dataset_dir[idx])

        # 构建样本字典, 并确保类型为float32
        sample = {
            'points': dataset['points'].copy().astype(np.float32),
            'suction_or': dataset['suction_or'].copy().astype(np.float32),
            'suction_seal_scores': dataset['suction_seal_scores'].copy().astype(np.float32),
            'suction_wrench_scores': dataset['suction_wrench_scores'].copy().astype(np.float32),
            'suction_feasibility_scores': dataset['suction_feasibility_scores'].copy().astype(np.float32),
            'individual_object_size_lable': dataset['individual_object_size_lable'].copy().astype(np.float32),
        }

        # 可选：收集样本的cycle和scene编号信息
        if self.collect_names:
            cycle_temp = self.dataset_dir[idx].split('/')[-2]
            cycle_index = int(cycle_temp.split('_')[1])
            obj_and_scene_temp = self.dataset_dir[idx].split('/')[-1]
            scene_index = int(obj_and_scene_temp[0:3])
            name = [cycle_index, scene_index]
            sample['name'] = name

        # 可选：对样本进行数据增强或转换
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


