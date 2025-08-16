import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    """
    Focal Loss 损失函数, 常用于处理类别不平衡的分割或检测任务。

    参数说明:
        alpha (float): 平衡因子, 控制正负样本的权重, 默认1
        gamma (float): 调整难易样本的聚焦参数, 默认0(等价于普通交叉熵)
        size_average (bool): 是否对损失取均值, True为均值, False为求和
        ignore_index (int): 忽略的标签索引, 默认为255(常用于分割任务的无效像素)
    """
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # 计算每个像素的交叉熵损失(不做reduction, 保留每个像素)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        # 计算概率pt(预测正确的概率)
        pt = torch.exp(-ce_loss)
        # 计算Focal Loss
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        # 按需取均值或求和
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()