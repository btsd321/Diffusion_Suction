from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, DeepLabV3TwoTower
from .backbone import resnet, resnetRGBD, resnetDepth
from .backbone import mobilenetv2
import torch.nn as nn

def _segm_resnetRGBD(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    # 构建基于RGBD输入的ResNet骨干的DeepLab分割模型
    # 参数说明：
    # name: 模型类型('deeplabv3plus'或'deeplabv3')
    # backbone_name: 主干网络名称(如'resnet50')
    # num_classes: 输出类别数
    # output_stride: 主干网络输出特征图的下采样倍数
    # pretrained_backbone: 是否加载预训练主干权重

    if output_stride==8:
        # 输出步长为8时, 后两层使用空洞卷积
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        # 默认输出步长为16, 仅最后一层使用空洞卷积
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    # 构建主干网络
    backbone = resnetRGBD.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048  # 主干输出通道数
    low_level_planes = 256  # 低层特征通道数

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnetDepth(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    # 构建仅以深度图为输入的ResNet骨干DeepLab分割模型
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnetDepth.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _mysegm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    # 构建双塔结构的ResNet分割模型(如RGB和D分别用两个主干)
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    # 两个主干网络分别处理不同输入
    backbone1 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    backbone2 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    # 低层特征融合层
    fuselayers1 = nn.Sequential(
            nn.Conv2d(low_level_planes*2, low_level_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_level_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_level_planes, low_level_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_level_planes),
            nn.ReLU(inplace=True),
        )

    # 高层特征融合层
    fuselayers2 = nn.Sequential(
            nn.Conv2d(inplanes*2, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
    backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)

    model = DeepLabV3TwoTower(backbone1, backbone2, fuselayers1, fuselayers2, classifier)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    # 构建基于MobileNetV2主干的DeepLab分割模型
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # 重命名特征层, 便于后续提取
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    # 根据主干类型和名称构建对应的分割模型
    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet') and 'depth' in backbone:
        print('load model which only takes in depth')
        model = _segm_resnetDepth(arch_type, backbone.split('_')[0], num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        print('load model which takes in rgbd')
        model = _segm_resnetRGBD(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model

def _load_mymodel(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    # 构建双塔结构分割模型
    model = _mysegm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model

# 以下为各类DeepLab模型的构建函数

# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-50为主干的DeepLabV3模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-101为主干的DeepLabV3模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """构建以MobileNetV2为主干的DeepLabV3模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-50为主干的DeepLabV3+模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50_2tower(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-50为主干的双塔DeepLabV3+模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_mymodel('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-101为主干的DeepLabV3+模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101_depth(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以ResNet-101为主干的仅深度输入DeepLabV3+模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3plus', 'resnet101_depth', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """构建以MobileNetV2为主干的DeepLabV3+模型

    参数说明:
        num_classes (int): 类别数
        output_stride (int): 主干网络输出特征图的下采样倍数
        pretrained_backbone (bool): 是否加载预训练主干权重
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)