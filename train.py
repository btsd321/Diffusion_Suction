""" 
扩散吸取网络训练主脚本
作者: dingtao huang
for diffusion_scution_net
"""
import os    
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
PROJECT_NAME = os.path.basename(FILE_DIR)
ROOT_DIR = FILE_DIR  # 修改：ROOT_DIR应该是当前项目根目录

# 修正路径设置
sys.path.append(ROOT_DIR)
# 修正：使用FILE_DIR而不是FILE_PATH
diffusion_model_path = os.path.join(FILE_DIR, "diffusion_suctionnet_model")

sys.path.append(diffusion_model_path)

# 打印路径用于调试
# print("当前工作目录:", os.getcwd())
# print("FILE_DIR:", FILE_DIR)
# print("添加的路径:", diffusion_model_path)
# print("sys.path:", sys.path[-3:])  # 打印最后添加的几个路径

# 其他导入...
import math
from datetime import datetime
import h5py
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist

# 尝试条件导入 TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Logging will be disabled.")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

import time
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入自定义模块
from diffusion_suctionnet_model.model import dsnet, load_checkpoint, save_checkpoint
from diffusion_suctionnet_model.utils.train_helper import BNMomentumScheduler, OptimizerLRScheduler, SimpleLogger
from diffusion_suctionnet_model.data.pointcloud_transforms import PointCloudShuffle, ToTensor
from diffusion_suctionnet_model.data.dataset_plus import DiffusionSuctionNetDataset

# 本工程的py资源
import utils

parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet/train', help='数据集根目录')
# 训练循环编号
parser.add_argument('--train_cycle_list', type=str, required=True, 
                   help='训练循环编号，支持格式: "5"(单个), "[1,10]"(闭区间), "[1,10:2]"(带步长闭区间，冒号后面是步长), "{1,3,5}"(列表)')
# 训练场景编号  
parser.add_argument('--train_scene_list', type=str, required=True, 
                   help='训练循环编号，支持格式: "5"(单个), "[1,10]"(闭区间), "[1,10:2]"(带步长闭区间，冒号后面是步长), "{1,3,5}"(列表)')
# 测试循环编号
parser.add_argument('--test_cycle_list', type=str, required=True, 
                   help='测试循环编号，支持格式: "5"(单个), "[1,10]"(闭区间), "[1,10:2]"(带步长闭区间，冒号后面是步长), "{1,3,5}"(列表)')
# 测试场景编号
parser.add_argument('--test_scene_list', type=str, required=True, 
                   help='测试场景编号，支持格式: "5"(单个), "[1,10]"(闭区间), "[1,10:2]"(带步长闭区间，冒号后面是步长), "{1,3,5}"(列表)')
parser.add_argument('--output_dir', type=str, default='D:\\Project\\Diffusion_Suction\\output', help='输出目录')
parser.add_argument('--checkpoint_path', type=str, default='', help='模型检查点路径')
parser.add_argument('--device_list', type=str, default='0', help='训练设备GPU编号, 输入支持单数字如"0", 列表[0,2], "[1,10:2]"(带步长闭区间，冒号后面是步长), 集合{0,2,3}')
parser.add_argument('--batch_size', type=int, default=4, help='训练的batch size - 增大以提高训练稳定性')
parser.add_argument('--max_epoch', type=int, default=200, help='最大训练epoch')
parser.add_argument('--train_data_hold_epoch', type=int, default=3, help='一个cycle训练多少个epoch')
parser.add_argument('--eval_stap', type=int, default=10, help='多少个epoch进行一次验证')
parser.add_argument('--display_batch_step', type=int, default=20, help='每多少个batch打印一次loss - 减小以便更频繁监控')
parser.add_argument('--save_stap', type=int, default=50, help='每多少个epoch保存一次权重文件')
parser.add_argument('--base_learning_rate', type=float, default=0.003, help='初始学习率 - 提高以加快收敛')
parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='学习率衰减率 - 调整衰减策略')
parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率 - 适当提高下限')
parser.add_argument('--lr_decay_step', type=int, default=50, help='学习率衰减步长 - 更频繁衰减')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], 
                   help='学习率调度器类型: step(指数衰减), cosine(余弦退火), plateau(自适应)')
parser.add_argument('--bn_momentum_init', type=float, default=0.5, help='BN动量初始值')
parser.add_argument('--bn_momentum_min', type=float, default=0.001, help='BN动量最小值')
parser.add_argument('--bn_decay_step', type=int, default=80, help='BN动量衰减步长')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='BN动量衰减率')
parser.add_argument('--log_dir', type=str, default='D:\\Project\\Diffusion_Suction\\output\\log', help='日志保存路径名')
ARGS = parser.parse_args()



# ======================== 训练与BN/LR参数 ========================
BATCH_SIZE = ARGS.batch_size  # 批量大小，每次训练处理的样本数量。较大的batch size可以提高训练稳定性，但需要更多GPU内存
MAX_EPOCH = ARGS.max_epoch  # 最大训练轮数，完整遍历数据集的次数。训练将在达到此轮数后停止
TRAIN_DATA_HOLD_EPOCH = ARGS.train_data_hold_epoch  # 每个cycle数据集持续训练的epoch数。为了节省内存，每隔几个epoch会切换到新的数据子集
EVAL_STAP = ARGS.eval_stap  # 验证间隔，每隔多少个epoch在验证集上评估一次模型性能
DISPLAY_BATCH_STEP = ARGS.display_batch_step  # 日志显示间隔，每处理多少个batch打印一次训练损失，用于监控训练进度
SAVE_STAP = ARGS.save_stap  # 模型保存间隔，每隔多少个epoch保存一次模型权重文件，防止训练意外中断时丢失进度

# 学习率与BN动量调度参数
BASE_LEARNING_RATE = ARGS.base_learning_rate  # 初始学习率，控制梯度下降的步长。较大的学习率收敛快但可能不稳定
LR_DECAY_RATE = ARGS.lr_decay_rate  # 学习率衰减倍数，每次衰减时学习率乘以此值。0.7表示每次衰减到原来的70%
MIN_LR = ARGS.min_lr  # 最小学习率阈值，学习率不会衰减到此值以下，防止学习率过小导致训练停滞
LR_DECAY_STEP = ARGS.lr_decay_step  # 学习率衰减步长，每隔多少个epoch进行一次学习率衰减
BN_MOMENTUM_INIT = ARGS.bn_momentum_init  # BatchNorm动量初始值，控制BN层的滑动平均更新速度。较大值表示更依赖历史统计
BN_MOMENTUM_MIN = ARGS.bn_momentum_min  # BatchNorm动量最小值，动量不会衰减到此值以下
BN_DECAY_STEP = ARGS.bn_decay_step  # BN动量衰减步长，通常与学习率衰减步长保持一致
BN_DECAY_RATE = ARGS.bn_decay_rate  # BN动量衰减倍数，每次衰减时动量乘以此值
LR_SCHEDULER_TYPE = ARGS.lr_scheduler_type  # 学习率调度器类型

# 学习率衰减函数：随epoch增长，学习率按指数规律衰减，但不低于最小学习率
LR_LAMBDA = lambda epoch: max(BASE_LEARNING_RATE * LR_DECAY_RATE**(int(epoch / LR_DECAY_STEP)), MIN_LR)
# BN动量衰减函数：随训练进行，BN动量逐渐减小，使模型后期更依赖当前batch的统计信息
BN_LAMBDA = lambda epoch: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(epoch / BN_DECAY_STEP)), BN_MOMENTUM_MIN)

def generate_list(start, end, step):
    """
    生成cycle区间列表, 每个元素为[start, end)的区间
    """
    result = []
    for i in range(start, end, step):
        result.append([i,i+step])
    return result

# ======================== 需要根据实际情况修改的参数 ========================
LOG_NAME = 'train'  # 日志保存路径名
# ======================== 需要根据实际情况修改的参数 ========================

class DiffusionSuctionNetTrainInput:
    def __init__(self):
        self.dataset_dir = ARGS.data_dir  # 展开 ~ 符号
        self.start_epoch = 0  # 开始训练的epoch编号
        self.gpus_is = False  # 是否使用GPU训练
        self.logger = None  # 日志记录器
        self.net = None  # 网络模型
        self.optimizer = None  # 优化器
        self.scheduler = None  # 学习率/BN动量调度器
        self.writer = None  # 日志记录器
        self.bnm_scheduler = None  # BN动量调度器
        self.lr_scheduler = None  # 学习率调度器
        self.log_dir = ARGS.log_dir  # 展开 ~ 符号
        self.checkpoint_path = None  # 断点恢复路径
        self.test_dataset = None  # 测试数据集
        self.test_sampler = None  # 测试数据集采样器
        self.test_loader = None # 测试数据加载器
        self.device = None  # 训练设备
        self.train_dataset = None  # 训练数据集
        
        # 解析输入参数并验证
        try:
            self.train_cycle_list = utils.parse_range_or_single(ARGS.train_cycle_list)
            self.train_scene_list = utils.parse_range_or_single(ARGS.train_scene_list)
            self.test_cycle_list = utils.parse_range_or_single(ARGS.test_cycle_list)
            self.test_scene_list = utils.parse_range_or_single(ARGS.test_scene_list)
            
            self.train_cycle_range = utils.list_to_range(self.train_cycle_list)
            self.train_scene_range = utils.list_to_range(self.train_scene_list)
            self.test_cycle_range = utils.list_to_range(self.test_cycle_list)
            self.test_scene_range = utils.list_to_range(self.test_scene_list)
            
            # 验证范围
            utils.validate_range(self.train_cycle_range, "train_cycle_range")
            utils.validate_range(self.train_scene_range, "train_scene_range")
            utils.validate_range(self.test_cycle_range, "test_cycle_range")
            utils.validate_range(self.test_scene_range, "test_scene_range")
            
            utils.print_range_info(self.train_cycle_range, self.train_scene_range, "训练集")
            utils.print_range_info(self.test_cycle_range, self.test_scene_range, "测试集")
            
            self.device_list = utils.parse_range_or_single(ARGS.device_list)
            
        except Exception as e:
            print(f"参数解析错误: {e}")
            sys.exit(1)

def train_one_epoch(loader, epoch, input):
    """
    单轮训练过程。遍历训练集, 前向、反向传播并统计损失。

    参数:
        loader: 训练数据加载器
        epoch: 当前epoch编号
    """
    input.logger.reset_state_dict('train loss1','train loss2', 'grad_norm')
    if input.gpus_is:
        if dist.get_rank() == 0:
            input.logger.log_string('----------------TRAIN STATUS---------------')
    else:
        input.logger.log_string('----------------TRAIN STATUS---------------')
    input.net.train() 
    
    for batch_idx, batch_samples in enumerate(loader):
        # if batch_idx == 2:
        #     start_time = time.time()
        
        # ---------------------数据准备---------------------
        # 减小噪声强度，避免过度干扰训练
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['points'].shape)).float()
        # 降低噪声系数从2.0到0.5，减少对原始数据的干扰
        input_points_with_noise = batch_samples['points'] + xyz_noise * 0.5
        labels = {
            'normals': batch_samples['normals'].to(input.device),
            'normal_flip_mask': batch_samples['normal_flip_mask'].to(input.device),
            'wrench_scores': batch_samples['wrench_scores'].to(input.device),
            'feasibility_scores': batch_samples['feasibility_scores'].to(input.device),
            'visibility_scores': batch_samples['visibility_scores'].to(input.device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(input.device),
            'labels': labels
        }
        # ---------------------数据准备---------------------

        input.optimizer.zero_grad()
        _, losses = input.net(inputs)
        losses_all = losses[0] + losses[1]
        
        # 添加梯度监控和异常检测
        if torch.isnan(losses_all) or torch.isinf(losses_all):
            print(f"警告: 检测到异常损失值 - loss1: {losses[0].item()}, loss2: {losses[1].item()}")
            continue  # 跳过这个batch
            
        losses_all.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(input.net.parameters(), max_norm=1.0)
        
        input.optimizer.step()
        
        # 计算梯度范数（每个batch都计算，避免键不匹配问题）
        total_norm = 0
        for p in input.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        log_state_dict = {
            'train loss1': losses[0].item(),
            'train loss2': losses[1].item(),
            'grad_norm': total_norm,  # 每个batch都包含梯度范数
        }
        input.logger.update_state_dict(log_state_dict)
        torch.cuda.empty_cache()  # 释放未使用显存

    if input.gpus_is:
        if dist.get_rank() == 0:
            input.logger.print_state_dict(log=True)
            loss_info = input.logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                input.writer.add_scalar(k, v, epoch)
    else:
        input.logger.print_state_dict(log=True)
        loss_info = input.logger.return_state_dict()
        for i, (k, v) in enumerate(loss_info.items()):
            input.writer.add_scalar(k, v, epoch)

def eval_one_epoch(loader, epoch, input):
    """
    单轮验证过程。遍历验证集, 统计损失。

    参数:
        loader: 验证数据加载器
        epoch: 当前epoch编号
    """
    input.logger.reset_state_dict('eval loss1','eval loss2')
        
    if input.gpus_is:
        if dist.get_rank() == 0:
            input.logger.log_string('----------------EVAL STATUS---------------')
    else:
        input.logger.log_string('----------------EVAL STATUS---------------')

    input.net.eval() 
    loss_sum = 0
    for batch_idx, batch_samples in enumerate(loader):
        xyz_noise = torch.from_numpy(np.random.standard_normal(batch_samples['points'].shape)).float()
        # 验证时也使用相同的噪声强度
        input_points_with_noise = batch_samples['points'] + xyz_noise * 0.5
        labels = {
            'normals': batch_samples['normals'].to(input.device),
            'normal_flip_mask': batch_samples['normal_flip_mask'].to(input.device),
            'wrench_scores': batch_samples['wrench_scores'].to(input.device),
            'feasibility_scores': batch_samples['feasibility_scores'].to(input.device),
            'visibility_scores': batch_samples['visibility_scores'].to(input.device),
        }
        inputs = {
            'point_clouds': input_points_with_noise.to(input.device),
            'labels': labels
        }

        with torch.no_grad():
            _, losses = input.net(inputs)
            losses_all = losses[0] + losses[1]
            loss_sum += losses_all.item()
            log_state_dict = {
                'eval loss1': losses[0].item(),
                'eval loss2': losses[1].item(), 
            }
            input.logger.update_state_dict(log_state_dict)
                         
    if input.gpus_is:
        if dist.get_rank() == 0:
            input.logger.print_state_dict(log=True)
            loss_info = input.logger.return_state_dict()
            for i, (k, v) in enumerate(loss_info.items()):
                input.writer.add_scalar(k, v, epoch)
    else:
        input.logger.print_state_dict(log=True)
        loss_info = input.logger.return_state_dict()
        for i, (k, v) in enumerate(loss_info.items()):
            input.writer.add_scalar(k, v, epoch)         
           
    return loss_sum

def train_environment_init():
    input = DiffusionSuctionNetTrainInput()
    train_files_path = input.dataset_dir
    # 验证数据集目录是否存在
    if not os.path.exists(train_files_path):
        raise FileNotFoundError(f"数据集目录不存在: {train_files_path}")
    
    # ======================== 数据集文件详细检查 ========================
    print(f"=== 数据集文件详细检查 ===")
    
    def check_cycle_files(cycle_range, scene_range, description):
        """检查指定cycle和scene范围的文件是否存在"""
        print(f"\n{description} 文件检查:")
        missing_files = []
        existing_files = []
        
        for cycle_id in range(cycle_range[0], cycle_range[1]):
            cycle_dir = os.path.join(train_files_path, f'cycle_{cycle_id:04d}')
            # print(f"  检查 cycle_{cycle_id:04d}:")
            
            if not os.path.exists(cycle_dir):
                print(f"    ❌ 目录不存在: {cycle_dir}")
                continue
            
            for scene_id in range(scene_range[0], scene_range[1]):
                h5_file = os.path.join(cycle_dir, f'{scene_id:03d}.h5')
                if os.path.exists(h5_file):
                    existing_files.append(h5_file)
                    # print(f"    ✓ {scene_id:03d}.h5")
                else:
                    missing_files.append(h5_file)
                    print(f"    ❌ {scene_id:03d}.h5 (不存在)")
        
        print(f"  总结: 存在 {len(existing_files)} 个文件，缺失 {len(missing_files)} 个文件")
        if missing_files:
            print(f"  缺失的文件示例: {missing_files[:5]}")
        
        return existing_files, missing_files
    
    # 检查训练集文件
    train_existing, train_missing = check_cycle_files(
        input.train_cycle_range, input.train_scene_range, "训练集"
    )
    
    # 检查测试集文件
    test_existing, test_missing = check_cycle_files(
        input.test_cycle_range, input.test_scene_range, "测试集"
    )
    
    # 如果训练集有缺失文件，给出建议
    if train_missing:
        print(f"\n⚠️  训练集有 {len(train_missing)} 个文件缺失！")
        print("建议：")
        print("1. 检查数据集是否完整下载/生成")
        print("2. 调整训练参数，只使用存在的文件")
        print("3. 重新生成缺失的数据文件")
        
        # 提供调整建议
        if train_existing:
            print(f"\n当前可用的训练文件: {len(train_existing)} 个")
            print("你可以尝试调整参数只使用存在的cycle/scene")
    
    # ======================== 日志与Tensorboard初始化 ========================
    # 先创建日志目录
    os.makedirs(input.log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'log_train_{timestamp}.txt'
    input.logger = SimpleLogger(input.log_dir, FILE_PATH, log_filename)
    
    # 创建 TensorBoard 日志目录（包含所有父目录）
    SummaryWriter_log_dir = os.path.join(input.log_dir, PROJECT_NAME, LOG_NAME, "tensorboard")
    os.makedirs(SummaryWriter_log_dir, exist_ok=True)
    
    try:
        if len(input.device_list) == 1:
            # 单卡训练
            input.gpus_is = False
            os.environ['CUDA_VISIBLE_DEVICES'] = str(input.device_list[0])
        else:
            # 多卡训练
            input.gpus_is = True
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in input.device_list)
    except ValueError as e:
        print(f"设备列表解析错误: {e}")
        sys.exit(1)
        
    # ======================== 分布式训练参数与环境初始化 ========================
    if input.gpus_is:
        parser_local = argparse.ArgumentParser()
        parser_local.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
        args = parser_local.parse_args()

        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            input.device = torch.device("cuda", args.local_rank)

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='59882') 
            torch.distributed.init_process_group(backend="nccl",init_method=dist_init_method, world_size=world_size,rank=rank)
    else:
        input.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ======================== 网络初始化与断点恢复 ========================
    if input.gpus_is:
        # (use_vis_branch, return_loss)
        input.net = dsnet(True, True, "cuda")
        input.net = input.net.to(input.device)
        input.net = nn.SyncBatchNorm.convert_sync_batchnorm(input.net)
        num_gpus = torch.cuda.device_count()

        if input.checkpoint_path is not None:
            input.net, input.optimizer, input.start_epoch = load_checkpoint(input.checkpoint_path, input.net, None)
        else:
            input.start_epoch = 0

        if num_gpus > 1:
            input.net = nn.parallel.DistributedDataParallel(input.net, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
        if dist.get_rank() == 0 and TENSORBOARD_AVAILABLE:
            input.writer = SummaryWriter(SummaryWriter_log_dir)   
    else:
        input.net = dsnet(True, True, "cuda")
        input.net.to(input.device)
        if TENSORBOARD_AVAILABLE:
            input.writer = SummaryWriter(SummaryWriter_log_dir)

        if input.checkpoint_path is not None:
            input.net, input.optimizer, input.start_epoch = load_checkpoint(input.checkpoint_path, input.net, None)
        else:
            input.start_epoch = 0

    # ======================== 优化器、BN、LR调度器初始化 ========================
    # 使用AdamW优化器，添加权重衰减以防止过拟合
    input.optimizer = torch.optim.AdamW(
        input.net.parameters(), 
        lr=BASE_LEARNING_RATE,
        weight_decay=1e-4,  # 添加L2正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 根据参数选择不同的学习率调度器
    if LR_SCHEDULER_TYPE == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        input.main_scheduler = CosineAnnealingLR(
            input.optimizer, 
            T_max=MAX_EPOCH, 
            eta_min=MIN_LR
        )
        print(f"使用余弦退火学习率调度器: T_max={MAX_EPOCH}, eta_min={MIN_LR}")
        
    elif LR_SCHEDULER_TYPE == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        input.main_scheduler = ReduceLROnPlateau(
            input.optimizer,
            mode='min',
            factor=LR_DECAY_RATE,
            patience=10,
            min_lr=MIN_LR,
            verbose=True
        )
        print(f"使用自适应学习率调度器: factor={LR_DECAY_RATE}, patience=10")
        
    elif LR_SCHEDULER_TYPE == 'step':
        from torch.optim.lr_scheduler import StepLR
        input.main_scheduler = StepLR(
            input.optimizer,
            step_size=LR_DECAY_STEP,
            gamma=LR_DECAY_RATE
        )
        print(f"使用步长衰减调度器: step_size={LR_DECAY_STEP}, gamma={LR_DECAY_RATE}")
        
    else:
        raise ValueError(f"不支持的学习率调度器类型: {LR_SCHEDULER_TYPE}")
    
    input.bnm_scheduler = BNMomentumScheduler(input.net, bn_lambda=BN_LAMBDA, last_epoch=input.start_epoch-1)
    # 保留原有的调度器作为备选
    input.lr_scheduler = OptimizerLRScheduler(input.optimizer, lr_lambda=LR_LAMBDA, last_epoch=input.start_epoch-1)

    # ======================== 数据增强与转换初始化 ========================
    input.transform_list = transforms.Compose([
        PointCloudShuffle(),  # 点云乱序
        ToTensor()
    ])

    # ======================== 验证集加载 ========================
    try:
        print('\n=== 开始加载测试数据集 ===')
        input.test_dataset = DiffusionSuctionNetDataset(
            input.dataset_dir, 
            input.test_cycle_range, 
            input.test_scene_range, 
            transforms=input.transform_list
        )
        
        if len(input.test_dataset) == 0:
            raise ValueError(f"测试数据集为空！请检查数据路径和范围设置。\n"
                           f"数据目录: {input.dataset_dir}\n"
                           f"测试cycle范围: {input.test_cycle_range}\n"
                           f"测试scene范围: {input.test_scene_range}")
        
        if input.gpus_is:
            if dist.get_rank() == 0:
                print(f'✓ 测试数据集加载成功，样本数量: {len(input.test_dataset)}')
            input.test_sampler = torch.utils.data.distributed.DistributedSampler(input.test_dataset)
            input.test_loader = DataLoader(input.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=input.test_sampler)
        else:
            print(f'✓ 测试数据集加载成功，样本数量: {len(input.test_dataset)}')
            input.test_loader = DataLoader(input.test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            
    except Exception as e:
        print(f"\n❌ 测试数据集加载失败: {e}")
        print("请检查数据集路径和参数设置")
        raise
    
    input.train_dataset = None
    return input
    

def train(start_epoch, input=None):
    """
    训练主循环。每个epoch动态加载训练集, 训练并定期验证与保存模型。

    参数:
        start_epoch: 训练起始epoch编号
    """
    min_loss = 1e10
    epoch_times = []  # 记录每个epoch的耗时
    
    for epoch in range(start_epoch, MAX_EPOCH):
        epoch_start_time = time.time()  # 记录epoch开始时间 
        # ---------------------动态加载训练集---------------------
        if epoch%TRAIN_DATA_HOLD_EPOCH == 0 or input.train_dataset is None:
            cid = int(epoch/TRAIN_DATA_HOLD_EPOCH) % len(input.train_cycle_list)
            if input.gpus_is:
                if dist.get_rank() == 0:
                    print('Loading train dataset...')
                # 修复：使用单个cycle，但需要+1来形成有效范围
                train_cycle_range = [input.train_cycle_list[cid], input.train_cycle_list[cid] + 1]
                print(f"Loading cycle {input.train_cycle_list[cid]}, range: {train_cycle_range}")
                input.train_dataset = DiffusionSuctionNetDataset(
                    input.dataset_dir, 
                    train_cycle_range, 
                    input.train_scene_range, 
                    transforms=input.transform_list
                )
                input.train_sampler = torch.utils.data.distributed.DistributedSampler(input.train_dataset)
                input.train_loader = DataLoader(input.train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=input.train_sampler)
                if dist.get_rank() == 0:
                    print('Train dataset loaded, train point cloud size:', len(input.train_dataset))
            else:
                print('Loading train dataset...')
                # 修复：使用单个cycle，但需要+1来形成有效范围
                train_cycle_range = [input.train_cycle_list[cid], input.train_cycle_list[cid] + 1]
                print(f"Loading cycle {input.train_cycle_list[cid]}, range: {train_cycle_range}")
                input.train_dataset = DiffusionSuctionNetDataset(
                    input.dataset_dir, 
                    train_cycle_range, 
                    input.train_scene_range, 
                    transforms=input.transform_list
                )
                input.train_loader = DataLoader(input.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
                print('Train dataset loaded, train point cloud size:', len(input.train_dataset))
        # ---------------------动态加载训练集---------------------

        # ---------------------配置lr和bn参数---------------------
        input.bnm_scheduler.step(epoch) 
        
        if input.gpus_is:
            if dist.get_rank() == 0:
                input.logger.log_string('************** EPOCH %03d **************' % (epoch))
                input.logger.log_string(str(datetime.now()))
                input.logger.log_string('Current learning rate: %f (scheduler: %s)' % (input.optimizer.param_groups[0]['lr'], LR_SCHEDULER_TYPE))
                input.logger.log_string('Current BN decay momentum: %f'%(input.bnm_scheduler.get_bn_momentum(epoch)))
        else:
            input.logger.log_string('************** EPOCH %03d **************' % (epoch))
            input.logger.log_string(str(datetime.now()))
            input.logger.log_string('Current learning rate: %f (scheduler: %s)' % (input.optimizer.param_groups[0]['lr'], LR_SCHEDULER_TYPE))
            input.logger.log_string('Current BN decay momentum: %f'%(input.bnm_scheduler.get_bn_momentum(epoch)))
        
        train_one_epoch(input.train_loader, epoch, input)
        
        # 在训练完一个epoch后更新学习率调度器（除了plateau类型）
        if LR_SCHEDULER_TYPE != 'plateau':
            input.main_scheduler.step()
        
        if epoch % EVAL_STAP == 0 and epoch > 50:
            loss = eval_one_epoch(input.test_loader, epoch, input)
            
            # 如果使用plateau调度器，根据验证损失更新学习率
            if LR_SCHEDULER_TYPE == 'plateau':
                input.main_scheduler.step(loss)
                if input.gpus_is:
                    if dist.get_rank() == 0:
                        input.logger.log_string('Plateau scheduler updated, current lr: %f' % input.optimizer.param_groups[0]['lr'])
                else:
                    input.logger.log_string('Plateau scheduler updated, current lr: %f' % input.optimizer.param_groups[0]['lr'])
            
            if loss < min_loss:
                min_loss = loss
                save_checkpoint(os.path.join(input.log_dir, 'checkpoint.tar'), epoch, input.net, input.optimizer, loss)
                input.logger.log_string("Model saved in file: %s" % os.path.join(input.log_dir, 'checkpoint.tar'))
        
        if epoch % SAVE_STAP == 0 and epoch > 50:
            save_checkpoint(os.path.join(input.log_dir, str(epoch)+'_'+'checkpoint.tar'), epoch, input.net, input.optimizer, min_loss)
        
        # 计算并打印预计剩余时间
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 计算平均每个epoch耗时（取最近10个epoch的平均值，更准确）
        recent_times = epoch_times[-10:] if len(epoch_times) >= 10 else epoch_times
        avg_epoch_time = sum(recent_times) / len(recent_times)
        
        # 计算剩余时间
        remaining_epochs = MAX_EPOCH - epoch - 1
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        
        # 格式化时间显示
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        # 打印时间信息
        if input.gpus_is:
            if dist.get_rank() == 0:
                print(f"Epoch {epoch} completed, duration: {format_time(epoch_duration)}")
                print(f"Average epoch time: {format_time(avg_epoch_time)}")
                print(f"Estimated remaining time: {format_time(estimated_remaining_time)} ({remaining_epochs} epochs left)")
                print(f"Estimated completion time: {datetime.fromtimestamp(time.time() + estimated_remaining_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 60)
        else:
            print(f"Epoch {epoch} completed, duration: {format_time(epoch_duration)}")
            print(f"Average epoch time: {format_time(avg_epoch_time)}")
            print(f"Estimated remaining time: {format_time(estimated_remaining_time)} ({remaining_epochs} epochs left)")
            print(f"Estimated completion time: {datetime.fromtimestamp(time.time() + estimated_remaining_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
    
    print(f'训练的场景完成！！！！！！！！！！！！！')

if __name__ == '__main__':
    output = train_environment_init()
    try:
        train(start_epoch = output.start_epoch, input = output)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        if output.gpus_is:
            if dist.get_rank() == 0:
                print('Saving model...')
                save_checkpoint(os.path.join(output.log_dir, 'checkpoint.tar'), output.start_epoch, output.net, output.optimizer, 0)
                output.logger.log_string("Model saved in file: %s" % os.path.join(output.log_dir, 'checkpoint.tar'))
        else:
            print('Saving model...')
            save_checkpoint(os.path.join(output.log_dir, 'checkpoint.tar'), output.start_epoch, output.net, output.optimizer, 0)
            output.logger.log_string("Model saved in file: %s" % os.path.join(output.log_dir, 'checkpoint.tar'))
    except Exception as e:
        print(f"训练过程出现错误: {e}")
