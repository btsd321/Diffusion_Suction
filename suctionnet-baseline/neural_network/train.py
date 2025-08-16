import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from suctionnet.suctionnet_dataset import SuctionNetDataset
import ConvNet
import DeepLabV3Plus.network as network
from utils.avgmeter import AverageMeter
import time

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='deeplabv3plus_resnet101', help='模型文件名 [默认: votenet]')
parser.add_argument('--checkpoint_path', default=None, help='模型权重路径 [默认: None]')
parser.add_argument("--num_classes", type=int, default=2)# 类别数
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])# 输出步长
parser.add_argument('--camera', default='realsense', help='相机名称, kinect或realsense [默认: realsense]')
parser.add_argument('--log_dir', default='/DATA2/Benchmark/suction/models/log_kinectV6', help='模型日志与权重保存目录 [默认: log]')
parser.add_argument('--data_root', default='/DATA2/Benchmark/graspnet', help='数据集根目录 [默认: log]')
parser.add_argument('--label_root', default='/ssd1/hanwen/grasping/graspnet_label', help='标签根目录 [默认: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='训练轮数 [默认: 100]')
parser.add_argument('--batch_size', type=int, default=24, help='训练时的批次大小 [默认: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率 [默认: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='优化器L2正则 [默认: 0]')
parser.add_argument('--bn_decay_step', type=int, default=10, help='BN衰减周期(单位：epoch)[默认: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='BN衰减率 [默认: 0.5]')
parser.add_argument('--lr_decay_steps', default='20,40,60', help='学习率衰减的epoch [默认: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.7,0.7,0.7', help='学习率衰减率 [默认: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='是否覆盖已有日志和权重文件夹')
parser.add_argument('--finetune', action='store_true', help='是否微调backbone网络')
FLAGS = parser.parse_args()

# 参数初始化
DATA_ROOT = FLAGS.data_root
LABEL_ROOT = FLAGS.label_root
BATCH_SIZE = FLAGS.batch_size
CAMERA = FLAGS.camera
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path

BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]

# 日志目录准备
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('日志文件夹 %s 已存在, 确定要覆盖吗？(Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('退出程序...')
        exit()
    elif c == 'y' or c == 'Y':
        print('覆盖日志和权重文件夹...')
        os.system('rm -r %s' % (LOG_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def my_worker_init_fn(worker_id):
    # 多进程数据加载时的随机种子初始化
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# 构建训练集
TRAIN_DATASET = SuctionNetDataset(DATA_ROOT, LABEL_ROOT, camera=CAMERA, split='train', input_size=(480, 480))
print(len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, drop_last=True, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER))

# 模型选择映射表
'''
1. 主干网络类型
    ResNet50/ResNet101: 经典的残差网络, 层数不同(50层/101层), ResNet101更深, 表达能力更强, 但计算量更大。
    MobileNet: 轻量级网络, 参数量和计算量都比ResNet小, 适合对速度和资源有要求的场景。
2. 网络结构
    deeplabv3: DeepLabV3 结构, 适合语义分割, 特征提取能力强。
    deeplabv3plus: DeepLabV3+, 是在DeepLabV3基础上增加了解码器模块, 分割边界更精细, 效果通常更好。
    convnet_resnet101: 你项目自定义的网络, 主干是ResNet101, 通常用于特定任务(如吸取点检测)。
    deeplabv3plus_resnet101_depth: 可能是支持深度输入的DeepLabV3+, 适合RGBD等多模态输入。
3. 选型建议
    追求精度, 显存和速度不是瓶颈：
    推荐 deeplabv3plus_resnet101 或 deeplabv3plus_resnet101_depth(如果你有深度图)。

    追求速度或设备资源有限：
    推荐 deeplabv3plus_mobilenet 或 deeplabv3_mobilenet。

    想要平衡速度和精度：
    推荐 deeplabv3plus_resnet50 或 deeplabv3_resnet50。

    你的任务是吸取点检测且有自定义网络：
    可以尝试 convnet_resnet101。
'''
model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'convnet_resnet101': ConvNet.convnet_resnet101,
        'deeplabv3plus_resnet101_depth': network.deeplabv3plus_resnet101_depth
    }
# 实例化模型
net = model_map[FLAGS.model](num_classes=FLAGS.num_classes, output_stride=FLAGS.output_stride, pretrained_backbone=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("检测到多块GPU, 使用", torch.cuda.device_count(), "块GPU进行训练！")
    net = nn.DataParallel(net)

EPOCH_CNT = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    print('从以下路径加载模型权重:')
    print(CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    EPOCH_CNT = checkpoint['epoch']

net.to(device)

criterion = nn.MSELoss()  # 损失函数, 均方误差

# 优化器设置, 支持微调
if FLAGS.finetune:
    if 'deeplabv3' in FLAGS.model:
        conv1_params = list(map(id, net.module.backbone.conv1.parameters()))
        classifier_params = list(map(id, net.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in conv1_params + classifier_params,
                            net.parameters())
        optimizer = optim.Adam([
                {'params': base_params, 'lr': 0},
                {'params': net.module.backbone.conv1.parameters(), 'lr': BASE_LEARNING_RATE},
                {'params': net.module.classifier.parameters(), 'lr': BASE_LEARNING_RATE}], lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    elif 'convnet' in FLAGS.model:
        fuse_layer_params = list(map(id, net.module.fuselayers.parameters()))
        classifier_params = list(map(id, net.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fuse_layer_params + classifier_params,
                            net.parameters())
        optimizer = optim.Adam([
                {'params': base_params, 'lr': 0},
                {'params': net.module.fuselayers.parameters(), 'lr': BASE_LEARNING_RATE},
                {'params': net.module.classifier.parameters(), 'lr': BASE_LEARNING_RATE}], lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    else:
        raise NotImplementedError('未识别的模型名称')
else:
    optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# 获取当前学习率
def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

# 动态调整学习率
def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 单个epoch的训练过程
def train_one_epoch():
    losses = AverageMeter()
    score_losses = AverageMeter()
    center_losses = AverageMeter()

    adjust_learning_rate(optimizer, EPOCH_CNT)
    
    net.train()
    for batch_idx, (rgbs, depths, scores, wrenches, _) in enumerate(TRAIN_DATALOADER):
        optimizer.zero_grad()
        depths = torch.clamp(depths, 0, 1)
        # 输入数据拼接
        if FLAGS.model == 'convnet_resnet101':
            depths = depths.unsqueeze(-1).repeat([1, 1, 1, 3])
            rgbds = torch.cat([rgbs, depths], dim=-1)
        elif 'depth' in FLAGS.model:
            rgbds = depths.unsqueeze(-1)
        else:
            rgbds = torch.cat([rgbs, depths.unsqueeze(-1)], dim=-1)
        
        rgbds = rgbds.permute(0, 3, 1, 2)
        rgbds = rgbds.to(device)
        scores = scores.to(device)
        wrenches = wrenches.to(device)
        
        pred = net(rgbds)
        
        score_loss = criterion(pred[:, 0, ...], scores)
        wrench_loss = criterion(pred[:, 1, ...], wrenches)
        loss = score_loss + wrench_loss

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), rgbs.size(0))
        score_losses.update(score_loss.item(), rgbs.size(0))
        center_losses.update(wrench_loss.item(), rgbs.size(0))
        
        if (batch_idx+1) % 10 == 0:
            log_string('学习率: {lr:.3e} | '
                        'Epoch: [{0}][{1}/{2}] | '
                        '分数损失: {score_loss.val:.4f} ({score_loss.avg:.4f}) | '
                        '中心损失: {center_loss.val:.4f} ({center_loss.avg:.4f}) | '
                        '总损失: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        EPOCH_CNT, batch_idx+1, len(TRAIN_DATALOADER), lr=get_current_lr(EPOCH_CNT),
                        score_loss=score_losses, center_loss=center_losses, loss=losses))
    


# 总训练流程
def train():
    global EPOCH_CNT
    # 记录起始时间
    start_time = time.time()

    for epoch in range(EPOCH_CNT, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** 训练 EPOCH %03d ****' % (epoch))
        log_string('当前学习率: %f'%(get_current_lr(epoch)))
        train_one_epoch()
        
        if EPOCH_CNT % 10 == 0: # 每10个epoch保存一次模型

            save_dict = {'epoch': epoch+1, # 训练完一个epoch后, 下次从epoch+1开始
                        'optimizer_state_dict': optimizer.state_dict()}
            try: # 如果使用了nn.DataParallel(), 模型会作为DataParallel的子模块
                save_dict['model_state_dict'] = net.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_'+str(epoch)))
        
        end_time = time.time()
        log_string('Epoch %d 训练完毕, 用时 %d s' % (epoch, end_time - start_time))
        start_time = end_time
            

if __name__ == "__main__":
    train()