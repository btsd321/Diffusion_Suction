"""
Diffusion SuctionNet Model Package

A PyTorch implementation for 6DoF suction grasping using diffusion models.
"""

import os
import sys
import warnings

# 添加当前目录到 Python 路径
__file_dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__file_dir__)

# 版本信息
from ._version import __version__

# 导入主要模块
from .model import (
    dsnet,
    ScheduledCNNRefine,
    CNNDDIMPipiline,
    SpatialAttention,
    ChannelAttention,
    load_checkpoint,
    save_checkpoint,
    save_pth,
)

# 导入 PointNet2 模块
try:
    from . import pointnet2
except ImportError:
    import warnings
    warnings.warn("PointNet2 CUDA extensions not available. Please compile them first.")
    pointnet2 = None

# 导入调度器
try:
    from .diffusers.schedulers.scheduling_ddim import DDIMScheduler
except ImportError:
    import warnings
    warnings.warn("DDIM scheduler not available.")
    DDIMScheduler = None

# 导入数据相关模块
try:
    from . import data
except ImportError:
    import warnings
    warnings.warn("Data modules not available.")
    data = None

# 导入工具模块
try:
    from . import utils
except ImportError:
    import warnings
    warnings.warn("Utility modules not available.")
    utils = None

# 定义对外接口
__all__ = [
    '__version__',
    'dsnet',
    'ScheduledCNNRefine',
    'CNNDDIMPipiline',
    'SpatialAttention',
    'ChannelAttention',
    'load_checkpoint',
    'save_checkpoint',
    'save_pth',
    'DDIMScheduler',
    'pointnet2',
    'data',
    'utils',
]

# 包信息
__author__ = "btsd321"
__email__ = ""
__description__ = "Diffusion SuctionNet Model - A PyTorch implementation for 6DoF suction grasping"
__url__ = "https://github.com/btsd321/diffusion_suctionnet_model"
