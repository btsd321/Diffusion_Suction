# 本文件受项目根目录下 LICENSE 文件的约束。

class AverageMeter(object):
  """
  用于统计和存储指标的当前值、平均值、总和和计数的工具类。
  常用于训练过程中对loss、准确率等指标的动态统计与展示。
  """

  def __init__(self):
    # 初始化各项统计量
    self.reset()

  def reset(self):
    # 重置所有统计量为0
    self.val = 0      # 当前值
    self.avg = 0      # 平均值
    self.sum = 0      # 总和
    self.count = 0    # 计数

  def update(self, val, n=1):
    # 更新统计量
    # val: 新增的数值
    # n: 新增数值的权重(默认1, 适用于批量更新)
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count  # 计算新的平均值
