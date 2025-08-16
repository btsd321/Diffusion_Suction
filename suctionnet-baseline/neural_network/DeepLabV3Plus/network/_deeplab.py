import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel, _MySegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    DeepLabV3模型实现, 来自论文
    《Rethinking Atrous Convolution for Semantic Image Segmentation》
    https://arxiv.org/abs/1706.05587

    参数说明:
        backbone (nn.Module): 用于提取特征的主干网络。
            主干网络应返回一个OrderedDict[Tensor], key为"out"表示最后一层特征图, 
            若有辅助分类器则还应有"aux"。
        classifier (nn.Module): 用于对主干网络输出的"out"特征进行密集预测的模块。
        aux_classifier (nn.Module, 可选): 训练时使用的辅助分类器。
    """
    pass

class DeepLabV3TwoTower(_MySegmentationModel):
    """
    双塔结构的DeepLabV3模型实现, 来自论文
    《Rethinking Atrous Convolution for Semantic Image Segmentation》
    https://arxiv.org/abs/1706.05587

    参数说明:
        backbone (nn.Module): 用于提取特征的主干网络。
            主干网络应返回一个OrderedDict[Tensor], key为"out"表示最后一层特征图, 
            若有辅助分类器则还应有"aux"。
        classifier (nn.Module): 用于对主干网络输出的"out"特征进行密集预测的模块。
        aux_classifier (nn.Module, 可选): 训练时使用的辅助分类器。
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    # DeepLabV3+的头部结构, 融合高低层特征并输出分割结果
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # 低层特征通道降维
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ASPP模块用于高层特征提取
        self.aspp = ASPP(in_channels, aspp_dilate)

        # 融合后特征的分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # 低层特征降维
        low_level_feature = self.project( feature['low_level'] )
        # 高层特征经过ASPP
        output_feature = self.aspp(feature['out'])
        # 上采样到低层特征空间
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        # 拼接高低层特征并分类
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        # 初始化卷积和归一化层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    # DeepLabV3的头部结构, 仅用高层特征
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # 只用高层特征进行分类
        return self.classifier( feature['out'] )

    def _init_weight(self):
        # 初始化卷积和归一化层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ 空洞可分离卷积
    先进行深度可分离卷积(分组数等于输入通道数), 再进行逐点卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # 深度可分离卷积(每个输入通道单独卷积)
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # 逐点卷积(1x1卷积, 整合通道信息)
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        # 初始化卷积和归一化层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    # ASPP中的带空洞卷积分支
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    # ASPP中的全局池化分支
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]  # 获取输入特征图的空间尺寸
        x = super(ASPPPooling, self).forward(x)
        # 上采样到输入特征图大小
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    # 空洞空间金字塔池化模块(ASPP), 用于多尺度特征提取
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        # 1x1卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 三个不同空洞率的3x3卷积分支
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        # 全局平均池化分支
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合所有分支的输出
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        # 多分支并行处理
        for conv in self.convs:
            res.append(conv(x))
        # 拼接所有分支的输出
        res = torch.cat(res, dim=1)
        # 融合输出
        return self.project(res)



def convert_to_separable_conv(module):
    """
    将普通卷积层(kernel_size>1)替换为空洞可分离卷积层
    递归地遍历模块的所有子模块并替换
    """
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module