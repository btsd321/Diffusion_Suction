""" 
PyTorch训练辅助类与函数。

作者: Zhikai Dong
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict

def set_bn_momentum_default(bn_momentum):
    """
    返回一个函数, 用于设置BN层的动量。

    参数:
        bn_momentum: BN层的动量值

    返回:
        fn: 可用于nn.Module.apply的函数, 设置BN层动量
    """
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):
    """
    BN动量调度器。用于在训练过程中动态调整BatchNorm层的动量。

    参数:
        model: 需要调整BN动量的模型
        bn_lambda: 动量变化函数, 输入epoch返回动量
        last_epoch: 上一次的epoch编号
        setter: 设置动量的函数, 默认为set_bn_momentum_default
    """
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        """
        调整BN层动量。

        参数:
            epoch: 当前epoch编号(可选)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_bn_momentum(self, epoch):
        """
        获取指定epoch的BN动量。

        参数:
            epoch: epoch编号

        返回:
            bn_momentum: 动量值
        """
        if epoch is None:
            epoch = self.last_epoch
        bn_momentum = self.lmbd(epoch)
        return bn_momentum

class OptimizerLRScheduler(object):
    """
    优化器学习率调度器。用于兼容不同PyTorch版本的学习率调整。
    该调度器每次step时将所有参数组的学习率设置为指定值。

    参数:
        optimizer: 优化器对象
        lr_lambda: 学习率变化函数, 输入epoch返回学习率
        last_epoch: 上一次的epoch编号
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError(
                "Class '{}' is not a PyTorch torch.optim.Optimizer".format(
                    type(optimizer).__name__
                )
            )
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.step(last_epoch + 1)   # 初始化学习率
        self.last_epoch = last_epoch    # 重置last_epoch

    def step(self, epoch=None):
        """
        调整优化器的学习率。

        参数:
            epoch: 当前epoch编号(可选)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_lambda(epoch)

    def get_optimizer_lr(self):
        """
        获取当前优化器所有参数组的学习率。

        返回:
            lrs: 学习率列表
        """
        lrs = [ g['lr'] for g in self.optimizer.param_groups ]
        return lrs

class SimpleLogger():
    """
    简单日志记录器。支持训练过程中的日志输出、状态统计与保存。

    参数:
        log_dir: 日志保存目录
        file_path: 训练脚本路径(用于备份)
    """
    def __init__(self, log_dir, file_path):
        if os.path.exists(log_dir):
            # 如果目录已存在, 可根据需要决定是否报错
            pass
        else:
            os.makedirs(log_dir)
 
        os.system('cp %s %s' % (file_path, log_dir)) # 备份训练脚本
        self.log_file = open(os.path.join(log_dir, 'log_train.txt'), 'w')
        self.log_file.write('\n')
        self.cnt = 0
        self.state_dict = OrderedDict()

    def log_string(self, out_str):
        """
        写入日志并打印到控制台。

        参数:
            out_str: 日志字符串
        """
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)

    def reset_state_dict(self, *args):
        """
        重置状态统计字典。

        参数:
            *args: 需要统计的指标名称(字符串)
        """
        self.cnt = 0
        self.state_dict = OrderedDict()
        for k in args:
            assert isinstance(k, str)
            self.state_dict[k] = 0.0
    
    def update_state_dict(self, state_dict):
        """
        累加更新状态统计字典。

        参数:
            state_dict: 当前batch的指标字典
        """
        self.cnt += 1
        assert set(state_dict.keys()) == set(self.state_dict.keys())
        for k in state_dict.keys():
            self.state_dict[k] += state_dict[k]

    def print_state_dict(self, log=True, one_line=True, line_len=None):
        """
        打印当前统计的平均指标。

        参数:
            log: 是否写入日志文件(否则仅打印)
            one_line: 是否一行输出所有指标
            line_len: 每行输出的指标数(可选)
        """
        log_fn = self.log_string if log==True else print
        out_str = ''
        for i, (k,v) in enumerate(self.state_dict.items()):
            out_str += '%s: %f' % (k, 1.0*v/self.cnt)
            if i != len(self.state_dict.keys()):
                if line_len is not None and (i+1)%line_len==0:
                    out_str += '\n'
                else:
                    out_str += '\t' if one_line else '\n'
        log_fn(out_str)
    
    def return_state_dict(self, log=True, one_line=True, line_len=None):
        """
        返回当前统计的平均指标字典。

        参数:
            log: 是否写入日志文件(无实际作用, 仅接口一致)
            one_line: 是否一行输出(无实际作用, 仅接口一致)
            line_len: 每行输出的指标数(无实际作用, 仅接口一致)

        返回:
            xx: 平均指标的有序字典
        """
        log_fn = self.log_string if log==True else print
        out_str = ''
        xx = OrderedDict()
        for i, (k, v) in enumerate(self.state_dict.items()):
            xx[k] = 1.0 * v / self.cnt
        return xx

