import numpy as np
import torch

class PointCloudShuffle(object):
    """
    点云随机打乱变换类。
    用于将点云及其所有标签同步随机打乱, 增强模型的泛化能力。
    """
    def __init__(self):
        """
        初始化, 设置点云数量(默认16384)。
        """
        self.num_point = 16384

    def __call__(self, sample):
        """
        对输入样本进行点云及标签的同步随机打乱。

        参数:
            sample: 字典, 包含点云和所有标签的numpy数组

        返回:
            sample: 字典, 打乱后的点云和标签
        """
        pt_idxs = np.arange(0, self.num_point)
        np.random.shuffle(pt_idxs)

        # 将打乱之后的点云和标签一一对应, 保证标签与点云顺序一致
        sample['points'] = sample['points'][pt_idxs]
        sample['suction_or'] = sample['suction_or'][pt_idxs]
        sample['suction_seal_scores'] = sample['suction_seal_scores'][pt_idxs]
        sample['suction_wrench_scores'] = sample['suction_wrench_scores'][pt_idxs]
        sample['suction_feasibility_scores'] = sample['suction_feasibility_scores'][pt_idxs]
        sample['individual_object_size_lable'] = sample['individual_object_size_lable'][pt_idxs]
        return sample

class ToTensor(object):
    """
    numpy数组转为PyTorch张量的变换类。
    用于将样本中的所有字段从numpy格式转换为torch.Tensor格式, 便于后续模型训练。
    """
    def __call__(self, sample):
        """
        将样本中的所有numpy数组字段转换为torch张量。

        参数:
            sample: 字典, 包含点云和所有标签的numpy数组

        返回:
            sample: 字典, 所有字段均为torch.Tensor类型
        """
        sample['points'] = torch.from_numpy(sample['points'])
        sample['suction_or'] = torch.from_numpy(sample['suction_or'])
        sample['suction_seal_scores'] = torch.from_numpy(sample['suction_seal_scores'])
        sample['suction_wrench_scores'] = torch.from_numpy(sample['suction_wrench_scores'])
        sample['suction_feasibility_scores'] = torch.from_numpy(sample['suction_feasibility_scores'])
        sample['individual_object_size_lable'] = torch.from_numpy(sample['individual_object_size_lable'])
        return sample
