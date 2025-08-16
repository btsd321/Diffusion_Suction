from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    # 对归一化的Tensor进行反归一化处理, 恢复到原始像素分布
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    # Denormalize类, 用于将归一化的Tensor或numpy数组反归一化
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        # 支持Tensor和numpy数组的反归一化
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    # 设置模型中所有BatchNorm2d层的动量参数
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    # 将模型中所有BatchNorm2d层设置为评估模式(冻结均值和方差)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    # 创建目录, 如果目录不存在则新建
    if not os.path.exists(path):
        os.mkdir(path)
