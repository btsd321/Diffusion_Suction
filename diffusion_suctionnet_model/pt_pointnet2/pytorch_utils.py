import torch
import torch.nn as nn

class SharedMLP(nn.Sequential):
    def __init__(self, mlp_spec, bn=True):
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i+1], 1))
            if bn:
                layers.append(nn.BatchNorm2d(mlp_spec[i+1]))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

# 可选：实现 feature_dropout_no_scaling
class FeatureDropoutNoScaling(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask.float()
