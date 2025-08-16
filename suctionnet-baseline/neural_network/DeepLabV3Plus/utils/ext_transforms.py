import torchvision
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image

#
#  语义分割扩展数据增强变换
#

class ExtRandomHorizontalFlip(object):
    """以给定概率对输入的PIL图像进行随机水平翻转

    参数说明:
        p (float): 图像被翻转的概率, 默认值为0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待翻转的图像
            lbl (PIL Image): 待翻转的标签
        返回:
            PIL Image: 随机翻转后的图像和标签
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtCompose(object):
    """将多个变换组合在一起, 顺序依次对图像和标签进行处理
    参数说明:
        transforms (list): 变换对象列表
    示例:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ExtCenterCrop(object):
    """对输入的PIL图像进行中心裁剪
    参数说明:
        size (序列或int): 裁剪输出的目标尺寸。如果size为int, 则输出为正方形裁剪
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待裁剪的图像
            lbl (PIL Image): 待裁剪的标签
        返回:
            PIL Image: 裁剪后的图像和标签
        """
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class ExtRandomScale(object):
    """对输入的PIL图像进行随机缩放, 缩放比例在给定范围内随机采样
    参数说明:
        scale_range (tuple): 缩放比例范围
        interpolation: 插值方式, 默认双线性插值
    """
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待缩放的图像
            lbl (PIL Image): 待缩放的标签
        返回:
            PIL Image: 随机缩放后的图像和标签
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ExtScale(object):
    """将输入的PIL图像缩放到指定比例
    参数说明:
        scale (float): 缩放比例
        interpolation: 插值方式, 默认双线性插值
    """
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待缩放的图像
            lbl (PIL Image): 待缩放的标签
        返回:
            PIL Image: 缩放后的图像和标签
        """
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ExtRandomRotation(object):
    """对输入的PIL图像进行随机旋转
    参数说明:
        degrees (float/tuple): 旋转角度范围。如果为单个数, 则范围为(-degrees, +degrees)
        resample: 重采样方式
        expand (bool): 是否扩展输出以包含整个旋转后的图像
        center (tuple): 旋转中心, 默认图像中心
    """
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """为随机旋转获取参数
        返回:
            随机旋转角度
        """
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待旋转的图像
            lbl (PIL Image): 待旋转的标签
        返回:
            PIL Image: 旋转后的图像和标签
        """
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(lbl, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class ExtRandomHorizontalFlip(object):
    """以给定概率对输入的PIL图像进行随机水平翻转
    参数说明:
        p (float): 图像被翻转的概率, 默认值为0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待翻转的图像
            lbl (PIL Image): 待翻转的标签
        返回:
            PIL Image: 随机翻转后的图像和标签
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtRandomVerticalFlip(object):
    """以给定概率对输入的PIL图像进行随机垂直翻转
    参数说明:
        p (float): 图像被翻转的概率, 默认值为0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待翻转的图像
            lbl (PIL Image): 待翻转的标签
        返回:
            PIL Image: 随机翻转后的图像和标签
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtPad(object):
    # 对输入图像和标签进行填充, 使其尺寸能被diviser整除
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbl):
        h, w = img.size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0
        im = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        lbl = F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2))
        return im, lbl

class ExtToTensor(object):
    """将PIL图像或numpy数组转换为Tensor
    图像会被归一化到[0,1], 标签不会归一化
    参数说明:
        normalize (bool): 是否归一化图像
        target_type (str): 标签的目标类型
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self, pic, lbl):
        """
        注意标签不会归一化到[0, 1]
        参数说明:
            pic (PIL Image或numpy.ndarray): 待转换的图像
            lbl (PIL Image或numpy.ndarray): 待转换的标签
        返回:
            Tensor: 转换后的图像和标签
        """
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy( np.array( lbl, dtype=self.target_type) )
        else:
            return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) ), torch.from_numpy( np.array( lbl, dtype=self.target_type) )

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtNormalize(object):
    """对Tensor图像进行归一化处理
    给定均值mean和标准差std, 对每个通道进行归一化
    参数说明:
        mean (序列): 每个通道的均值
        std (序列): 每个通道的标准差
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        参数说明:
            tensor (Tensor): 需要归一化的图像Tensor, 形状为(C, H, W)
            lbl (Tensor): 标签Tensor, 仅作占位, 不做处理
        返回:
            Tensor: 归一化后的图像
            Tensor: 原始标签
        """
        return F.normalize(tensor, self.mean, self.std), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ExtRandomCrop(object):
    """对输入的PIL图像进行随机裁剪
    参数说明:
        size (序列或int): 裁剪输出的目标尺寸。如果size为int, 则输出为正方形裁剪
        padding (int或序列): 可选, 裁剪前的填充
        pad_if_needed (bool): 若为True, 当输入尺寸小于目标尺寸时自动填充
    """
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """为随机裁剪获取参数
        参数说明:
            img (PIL Image): 待裁剪的图像
            output_size (tuple): 裁剪输出的目标尺寸
        返回:
            tuple: (i, j, h, w) 随机裁剪参数
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待裁剪的图像
            lbl (PIL Image): 待裁剪的标签
        返回:
            PIL Image: 裁剪后的图像和标签
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # 若宽度不足则填充
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # 若高度不足则填充
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class ExtResize(object):
    """将输入的PIL图像缩放到指定尺寸
    参数说明:
        size (序列或int): 输出目标尺寸。如果为序列如(h, w), 则输出为该尺寸；如果为int, 则短边缩放到该值
        interpolation: 插值方式, 默认双线性插值
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 待缩放的图像
            lbl (PIL Image): 待缩放的标签
        返回:
            PIL Image: 缩放后的图像和标签
        """
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 
    
class ExtColorJitter(object):
    """随机改变图像的亮度、对比度、饱和度和色调
    参数说明:
        brightness (float或tuple): 亮度扰动范围
        contrast (float或tuple): 对比度扰动范围
        saturation (float或tuple): 饱和度扰动范围
        hue (float或tuple): 色调扰动范围
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number或长度为2的list/tuple".format(name))

        # 如果扰动为0则不做变换
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """获取用于颜色扰动的随机变换
        参数同__init__
        返回:
            一个组合的颜色扰动变换
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl):
        """
        参数说明:
            img (PIL Image): 输入图像
            lbl (PIL Image): 标签
        返回:
            PIL Image: 颜色扰动后的图像
            PIL Image: 原始标签
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """对图像应用自定义lambda函数
    参数说明:
        lambd (function): 用于变换的lambda或函数
    """
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    """将多个变换组合在一起, 依次对图像进行处理
    参数说明:
        transforms (list): 变换对象列表
    示例:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string