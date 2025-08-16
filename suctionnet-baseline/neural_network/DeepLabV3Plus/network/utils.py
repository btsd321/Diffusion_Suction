import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    # 简单分割模型, 包含一个主干网络和一个分类头
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone  # 主干网络
        self.classifier = classifier  # 分类头
        
    def forward(self, x):
        input_shape = x.shape[-2:]  # 输入的空间尺寸(高、宽)
        features = self.backbone(x)  # 提取特征
        x = self.classifier(features)  # 分类预测
        # 上采样到输入空间大小, 保证输出与输入尺寸一致
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class _MySegmentationModel(nn.Module):
    # 双塔分割模型, 包含两个主干网络、两个融合层和一个分类头
    def __init__(self, backbone1, backbone2, fuselayers1, fuselayers2, classifier):
        super(_MySegmentationModel, self).__init__()
        self.backbone1 = backbone1  # 第一个主干网络(如RGB)
        self.backbone2 = backbone2  # 第二个主干网络(如D)
        self.fuselayers1 = fuselayers1  # 第一个融合层(低层特征融合)
        self.fuselayers2 = fuselayers2  # 第二个融合层(高层特征融合)
        self.classifier = classifier    # 分类头
        
    def forward(self, x):
        input_shape = x.shape[-2:]  # 输入的空间尺寸
        x1 = x[:, :3, ...]  # 取前3通道(如RGB)
        x2 = x[:, 3:, ...]  # 取后面通道(如D或其他)
        features1 = self.backbone1(x1)  # 主干1特征
        features2 = self.backbone2(x2)  # 主干2特征
        features = {}
        # 拼接两个主干网络的低层特征, 并通过融合层
        features['low_level'] = torch.cat([features1['low_level'], features2['low_level']], dim=1)
        features['low_level'] = self.fuselayers1(features['low_level'])
        # 拼接两个主干网络的高层特征, 并通过融合层
        features['out'] = torch.cat([features1['out'], features2['out']], dim=1)
        features['out'] = self.fuselayers2(features['out'])
        # 分类预测
        x = self.classifier(features)
        # 上采样到输入空间大小, 保证输出与输入尺寸一致
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    模型包装器, 用于返回模型的中间层输出

    该模块有一个重要假设：模型的子模块注册顺序与实际前向传播顺序一致。
    也就是说, 如果你希望正常工作, 不要在forward中重复使用同一个nn.Module。

    此外, 只能获取直接赋值给模型的子模块(如model.layer1), 
    不能获取更深层次的子模块(如model.layer1.layer2)。

    参数说明:
        model (nn.Module): 需要提取特征的模型
        return_layers (Dict[name, new_name]): 一个字典, key为需要返回的模块名, 
            value为返回结果中的新名字(用户可自定义)

    示例::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # 提取layer1和layer3, 分别命名为'feat1'和'feat2'
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        # 检查return_layers中的层名是否都在模型的直接子模块中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        # 只保留需要的子模块, 并按顺序注册
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次前向传播, 并收集指定层的输出
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
