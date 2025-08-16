# modified from https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/schedulers/scheduling_ddim.py

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from typing import Dict 
import functools 
import inspect 
from types import SimpleNamespace 

def register_to_config(init):
    """
    装饰器：用于自动将__init__的参数注册到配置中。
    适用于继承自ConfigMixin的类。可通过ignore_for_config类变量忽略部分参数。
    注意：被装饰后, 所有以_开头的私有参数不会被注册。
    """
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # 忽略私有参数
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        
        ignore = getattr(self, "ignore_for_config", [])
        # 获取参数名与位置参数对齐
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # 添加所有关键字参数
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    根据累计alpha_bar函数生成beta序列, 用于扩散过程的噪声调度。

    参数:
        num_diffusion_timesteps (int): beta的步数
        max_beta (float): beta的最大值, 防止数值不稳定

    返回:
        betas (torch.Tensor): 调度器使用的beta序列
    """
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

class DDIMScheduler():
    """
    DDIM调度器类, 实现去噪扩散隐式模型(DDIM)的采样调度逻辑。
    支持多种beta调度方式、采样步数设置、噪声添加、反向采样等功能。
    """
    config_name = "scheduler_config.json"
    _deprecated_kwargs = ["predict_epsilon"]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        **kwargs,
    ):
        """
        初始化DDIM调度器。

        参数:
            num_train_timesteps: 训练时扩散步数
            beta_start: beta起始值
            beta_end: beta结束值
            beta_schedule: beta调度方式(linear/scaled_linear/squaredcos_cap_v2)
            trained_betas: 直接指定的beta序列
            clip_sample: 是否对预测结果裁剪到[-1,1]
            set_alpha_to_one: 最后一步alpha是否设为1
            steps_offset: 步数偏移
            prediction_type: 预测类型(epsilon/sample/v_prediction)
            **kwargs: 兼容历史参数
        """
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDIMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = kwargs.get('predict_epsilon', None)
        if predict_epsilon is not None:
            self.register_to_config(prediction_type="epsilon" if predict_epsilon else "sample")

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # 适用于latent diffusion模型的特殊调度
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 每一步都需要用到前一步的alphas_cumprod
        # 最后一步没有前一步, set_alpha_to_one决定是否直接设为1
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的参数
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def register_to_config(self, **kwargs):
        """
        将参数注册到内部配置字典。
        """
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                print(f"Can't set {key} with value {value} for {self}")
                raise err

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            print(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = internal_dict
    
    @property
    def config(self):
        """
        获取当前对象的配置(以SimpleNamespace形式返回)。
        返回:
            SimpleNamespace对象, 包含所有配置参数。
        """
        return SimpleNamespace(**self._internal_dict)
        
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        (兼容接口)部分调度器需要根据timestep缩放输入, 本实现直接返回原样。

        参数:
            sample: 输入样本
            timestep: 当前步数(可选)

        返回:
            缩放后的样本(本实现为原样返回)
        """
        return sample

    def _get_variance(self, timestep, prev_timestep):
        """
        计算当前步与前一步的方差, 用于采样过程中的噪声注入。

        参数:
            timestep: 当前步数
            prev_timestep: 前一步步数

        返回:
            variance: 方差值
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置采样过程中的离散步数(timesteps), 推理前需调用。

        参数:
            num_inference_steps: 推理时的采样步数
            device: 设备(可选)
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # 生成采样步数序列(倒序)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict, Tuple]:
        """
        反向采样一步, 根据当前模型输出和采样状态预测上一步的样本。

        参数:
            model_output: 扩散模型的直接输出(通常为噪声)
            timestep: 当前步数
            sample: 当前采样状态
            eta: 噪声权重
            use_clipped_model_output: 是否使用裁剪后的模型输出
            generator: 随机数生成器
            variance_noise: 直接指定的方差噪声(如CycleDiffusion用)
            return_dict: 是否以字典形式返回结果

        返回:
            如果return_dict为True, 返回{'prev_sample': ..., 'pred_original_sample': ...}
            否则返回(prev_sample,)
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. 计算上一步的timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. 计算当前和前一步的alpha/beta
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. 根据模型输出预测原始样本x0
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. 可选：对预测的x0裁剪到[-1,1]
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. 计算方差和标准差
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # 若使用裁剪后的x0, 需重新计算模型输出
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. 计算采样方向
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. 计算上一步的采样结果(不加噪声)
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # 8. 若eta>0, 加入噪声
        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                if device.type == "mps":
                    variance_noise = torch.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
                    variance_noise = variance_noise.to(device)
                else:
                    variance_noise = torch.randn(
                        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                    )
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise
            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return dict(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        给定原始样本、噪声和步数, 生成扩散过程中的带噪样本。

        参数:
            original_samples: 原始样本
            noise: 噪声
            timesteps: 步数张量

        返回:
            noisy_samples: 加噪后的样本
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        计算扩散过程中的速度(velocity), 用于某些扩散模型的特殊训练目标。

        参数:
            sample: 当前样本
            noise: 噪声
            timesteps: 步数张量

        返回:
            velocity: 速度张量
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        """
        返回训练时扩散步数。
        """
        return self.config.num_train_timesteps


