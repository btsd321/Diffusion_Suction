from .utils import IntermediateLayerGetter
from .backbone import resnet, resnetRGBD
import torch.nn as nn
import torch
import torch.nn.functional as F

def _convnet_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):
    # 构建基于ResNet骨干网络的ConvNet分割模型
    # 参数说明：
    # backbone_name: 使用的resnet骨干网络名称(如'resnet101')
    # num_classes: 输出类别数
    # output_stride: 输出特征图的下采样倍数
    # pretrained_backbone: 是否加载预训练权重

    if output_stride==8:
        # 若输出步长为8, 则后两层使用空洞卷积替换stride
        replace_stride_with_dilation=[False, True, True]
    else:
        # 默认输出步长为16, 仅最后一层使用空洞卷积
        replace_stride_with_dilation=[False, False, True]

    # 构建两个resnet骨干网络, 分别处理不同输入
    backbone1 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    backbone2 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048  # 主干网络输出通道数

    # 特征融合层, 先拼接再降维
    fuselayers = nn.Sequential(
            nn.Conv2d(inplanes*2, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )
    
    return_layers = {'layer4': 'out'}  # 指定提取主干网络的哪一层输出
    classifier = ConvNetHead(inplanes, num_classes)  # 分类头
    
    # 用IntermediateLayerGetter包装主干网络, 便于获取中间层输出
    backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
    backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)

    model = _MySegmentationModel(backbone1, backbone2, fuselayers, classifier)
    return model

class ConvNetHead(nn.Module):
    # 分割头部结构, 将主干输出映射到类别空间
    def __init__(self, in_channels, num_classes):
        super(ConvNetHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        # 输入为字典, 取出主干输出特征
        return self.classifier(feature['out'])
    
    def _init_weight(self):
        # 初始化卷积和归一化层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _MySegmentationModel(nn.Module):
    # 主分割模型, 包含两个主干网络、融合层和分类头
    def __init__(self, backbone1, backbone2, fuselayers, classifier):
        super(_MySegmentationModel, self).__init__()
        self.backbone1 = backbone1  # 第一个主干网络
        self.backbone2 = backbone2  # 第二个主干网络
        self.fuselayers = fuselayers  # 融合层
        self.classifier = classifier  # 分类头
        
    def forward(self, x):
        # 前向传播
        input_shape = x.shape[-2:]  # 输入原始空间尺寸
        x1 = x[:, :3, ...]  # 取前3通道(如RGB)
        x2 = x[:, 3:, ...]  # 取后面通道(如D或其他)
        features1 = self.backbone1(x1)  # 主干1特征
        features2 = self.backbone2(x2)  # 主干2特征

        features = {}
        # 拼接两个主干输出的特征图
        features['out'] = torch.cat([features1['out'], features2['out']], dim=1)
        # 融合特征
        features['out'] = self.fuselayers(features['out'])
        # 分类预测
        x = self.classifier(features)
        # 上采样到输入空间大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def _load_model(backbone, num_classes, output_stride, pretrained_backbone):
    # 根据骨干网络名称构建分割模型
    if backbone.startswith('resnet'):
        model = _convnet_resnet(backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    
    return model

def convnet_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建一个以ResNet-101为骨干的DeepLabV3+分割模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
