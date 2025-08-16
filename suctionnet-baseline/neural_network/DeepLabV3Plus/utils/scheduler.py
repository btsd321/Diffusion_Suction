from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    """
    多项式学习率(PolyLR)调度器, 常用于语义分割等任务的训练过程。

    参数说明:
        optimizer: 优化器对象
        max_iters: 最大迭代次数(训练总步数)
        power: 多项式幂指数, 控制学习率衰减曲线, 默认0.9
        last_epoch: 上一次迭代的epoch数, 默认-1
        min_lr: 最小学习率, 防止学习率降为0, 默认1e-6
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # 最大迭代次数, 用于计算学习率衰减
        self.min_lr = min_lr        # 最小学习率, 防止学习率为0
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # 按照多项式策略计算每个参数组的当前学习率
        # lr = max( base_lr * (1 - 当前步数/最大步数)^power, min_lr )
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]