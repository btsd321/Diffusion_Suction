""" 
扩散吸取网络推理部署脚本
作者: dingtao huang
用于加载训练好的模型进行吸取点预测推理
"""
import os    
import sys
import numpy as np
import torch
import argparse
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
from suction_nms import nms_suction
from preprocess import resample

# 设置项目路径
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
print(f"当前文件所在文件夹路径: {FILE_DIR}")
ROOT_DIR = os.path.dirname(FILE_DIR)
print(f"项目根目录路径: {ROOT_DIR}")

# 添加模块路径
sys.path.append(ROOT_DIR)
sys.path.append(FILE_DIR)  # 添加当前目录以导入 preprocess 模块
thirdparty_path = os.path.join(ROOT_DIR, "thirdparty", "Diffusion_Suction")
diffusion_model_path = os.path.join(ROOT_DIR, "thirdparty", "Diffusion_Suction", "diffusion_suctionnet_model")
sys.path.append(thirdparty_path)
sys.path.append(diffusion_model_path)

try:
    from diffusion_suctionnet_model.model import dsnet
    from preprocess import preprocess  # 导入预处理函数
    print("成功导入模型和工具函数")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("尝试从当前目录导入...")
    try:
        import sys
        sys.path.append(os.path.join(FILE_DIR, "thirdparty", "Diffusion_Suction", "diffusion_suctionnet_model"))
        from model import dsnet
        from preprocess import preprocess  # 导入预处理函数
        print("成功从当前目录导入模型和工具函数")
    except ImportError as e2:
        print(f"从当前目录导入也失败: {e2}")
        sys.exit(1)


class MemoryProfiler:
    """内存使用分析器"""
    def __init__(self):
        self.gpu_stats = []
        self.cpu_stats = []
    
    def reset_gpu_stats(self):
        """重置GPU内存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_gpu_memory(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated
            }
        return None
    
    def get_cpu_memory(self):
        """获取CPU内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3  # GB
    
    def log_memory(self, stage_name):
        """记录内存使用情况"""
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()
        
        print(f"\n=== {stage_name} 内存使用情况 ===")
        if gpu_mem:
            print(f"GPU内存 - 已分配: {gpu_mem['allocated']:.2f}GB, "
                  f"已保留: {gpu_mem['reserved']:.2f}GB, "
                  f"峰值: {gpu_mem['max_allocated']:.2f}GB")
        print(f"CPU内存: {cpu_mem:.2f}GB")
        
        self.gpu_stats.append((stage_name, gpu_mem))
        self.cpu_stats.append((stage_name, cpu_mem))
        
        return gpu_mem, cpu_mem


class ComputeProfiler:
    """计算性能分析器"""
    def __init__(self):
        self.timings = defaultdict(list)
        
    def estimate_flops(self, model_config, num_points, actual_diffusion_steps=None):
        """根据模型配置估算FLOPs"""
        if model_config is None:
            return 0
            
        pointnet_config = model_config.get('pointnet_config', {})
        diffusion_config = model_config.get('diffusion_config', {})
        
        # PointNet++ FLOPs估算
        pointnet_flops = 0
        npoint_per_layer = pointnet_config.get('npoint_per_layer', [4096, 64])
        backbone_dim = pointnet_config.get('backbone_feature_dim', 128)
        
        # 简化估算：每层的计算量
        for npoint in npoint_per_layer:
            pointnet_flops += npoint * backbone_dim * 64
        
        # Diffusion模型FLOPs估算 - 使用实际运行时的步数
        if actual_diffusion_steps is not None:
            diffusion_steps = actual_diffusion_steps
        else:
            diffusion_steps = diffusion_config.get('diffusion_inference_steps', 20)
            
        channels_in = diffusion_config.get('channels_in', 128)
        channels_noise = diffusion_config.get('channels_noise', 4)
        
        # 扩散过程的计算量
        diffusion_flops = diffusion_steps * num_points * channels_in * channels_noise * 10
        
        total_flops = pointnet_flops + diffusion_flops
        
        # 调试信息
        print(f"FLOPs计算详情:")
        print(f"  - PointNet++ FLOPs: {pointnet_flops/1e9:.2f} GFLOPs")
        print(f"  - Diffusion FLOPs: {diffusion_flops/1e9:.2f} GFLOPs (步数: {diffusion_steps})")
        print(f"  - 总计: {total_flops/1e9:.2f} GFLOPs")
        
        return total_flops


class SuctionNetInference:
    """
    扩散吸取网络推理类
    用于加载训练好的模型并进行吸取点预测
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda',
                 num_points: int = 16384,
                 diffusion_steps: int = 20,
                 enable_profiling: bool = False):
        """
        初始化推理器
        
        参数:
            checkpoint_path: 模型检查点文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            num_points: 点云采样点数
            diffusion_steps: 扩散推理步数
            enable_profiling: 是否启用性能分析
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        self.diffusion_steps = diffusion_steps
        self.enable_profiling = enable_profiling
        
        # 初始化性能分析器
        if self.enable_profiling:
            self.memory_profiler = MemoryProfiler()
            self.compute_profiler = ComputeProfiler()
        
        print(f"使用设备: {self.device}")
        print(f"点云采样点数: {self.num_points}")
        print(f"扩散推理步数: {self.diffusion_steps}")
        print(f"性能分析: {'开启' if self.enable_profiling else '关闭'}")
        
        # 加载模型配置
        self.model_config = self._load_model_config(checkpoint_path)
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print("模型加载完成，已设置为推理模式")
    
    def _load_model_config(self, checkpoint_path: str) -> Optional[dict]:
        """加载模型配置文件"""
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'model_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"成功加载模型配置: {config_path}")
                return config
            except Exception as e:
                print(f"加载模型配置失败: {e}")
        return None
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        加载训练好的模型
        
        参数:
            checkpoint_path: 检查点文件路径
            
        返回:
            model: 加载的模型
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"正在加载模型检查点: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型实例
        model = dsnet(
            use_vis_branch=False,  # 推理模式不需要可见性分支
            return_loss=False,     # 推理模式不返回损失
        )
        
        # 设置扩散推理步数
        model.diffusion_inference_steps = self.diffusion_steps
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("成功加载模型权重")
        
        model.to(self.device)
        return model
    
    def preprocess_point_cloud(self, 
                             point_cloud: np.ndarray,
                             normals: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """
        预处理输入点云数据
        
        参数:
            point_cloud: 点云坐标 (N, 3) numpy数组
            normals: 点云法向量 (N, 3) numpy数组，可选
            
        返回:
            inputs: 模型输入字典
        """
        print(f"输入点云形状: {point_cloud.shape}")
        print(f"输入法向量形状: {normals.shape}")
        
        point_cloud, normals = resample(point_cloud, normals, output_points_num=self.num_points)
        
        print(f"预处理后点云形状: {point_cloud.shape}")
        print(f"预处理后法向量形状: {normals.shape}")
        
        # 转换为tensor并添加batch维度
        point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(self.device)
        normals_tensor = torch.from_numpy(normals).float().unsqueeze(0).to(self.device)
        
        # 创建输入字典
        inputs = {
            'point_clouds': point_cloud_tensor,  # (1, num_points, 3)
            'labels': {
                'normals': normals_tensor  # (1, num_points, 3)
            }
        }
        
        return inputs
    
    def predict_with_profiling(self, point_cloud: np.ndarray, 
                              normals: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
        """
        带性能分析的预测函数
        
        参数:
            point_cloud: 点云坐标 (N, 3) numpy数组
            normals: 点云法向量 (N, 3) numpy数组，可选
            
        返回:
            results: 预测结果字典，包含各种吸取评分
            preprocessed_pc: 预处理后的点云坐标
            performance_stats: 性能统计信息
        """
        print("开始进行吸取点预测（带性能分析）...")
        
        performance_stats = {
            'total_time': 0,
            'preprocess_time': 0,
            'inference_time': 0,
            'postprocess_time': 0,
            'memory_usage': {},
            'flops_estimate': 0,
            'throughput': 0
        }
        
        total_start_time = time.perf_counter()
        
        # 重置GPU内存统计
        if self.enable_profiling:
            self.memory_profiler.reset_gpu_stats()
            self.memory_profiler.log_memory("推理开始前")
        
        # 数据预处理
        preprocess_start = time.perf_counter()
        inputs = self.preprocess_point_cloud(point_cloud, normals)
        preprocess_end = time.perf_counter()
        performance_stats['preprocess_time'] = preprocess_end - preprocess_start
        
        if self.enable_profiling:
            self.memory_profiler.log_memory("数据预处理后")
        
        # 获取预处理后的点云坐标（用于后续的最佳点定位）
        preprocessed_pc = inputs['point_clouds'].squeeze(0).cpu().numpy()  # (num_points, 3)
        preprocessed_normals = inputs['labels']['normals'].squeeze(0).cpu().numpy()  # (num_points, 3)
        
        # 临时修改模型的pipeline调用，使用动态点数
        original_forward = self.model.forward
        
        def patched_forward(inputs_dict):
            """修补的forward函数，支持动态点数"""
            batch_size = inputs_dict['point_clouds'].shape[0]
            num_point = inputs_dict['point_clouds'].shape[1]
            
            # pointnet++提取堆叠场景点云
            input_points = inputs_dict['point_clouds']
            input_points = torch.cat((input_points, inputs_dict['labels']['normals']), dim=2)
            features, global_features = self.model.backbone(input_points)

            # 推理模式
            pred_results = self.model.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(num_point, 4),  # 使用动态点数
                features=features,
                num_inference_steps=self.model.diffusion_inference_steps,
            )
            ddim_loss = None
            return pred_results, ddim_loss
        
        # 暂时替换forward方法
        self.model.forward = patched_forward
        
        try:
            # 模型推理 - 优化时间测量
            # 预热GPU（第一次推理可能较慢）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            inference_start = time.perf_counter()
            
            with torch.no_grad():
                pred_results, _ = self.model(inputs)
            
            # 确保GPU计算完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_end = time.perf_counter()
            performance_stats['inference_time'] = inference_end - inference_start
            
        finally:
            # 恢复原始forward方法
            self.model.forward = original_forward
        
        if self.enable_profiling:
            self.memory_profiler.log_memory("模型推理后")
        
        # 后处理
        postprocess_start = time.perf_counter()
        results = self.postprocess_results(pred_results)
        postprocess_end = time.perf_counter()
        performance_stats['postprocess_time'] = postprocess_end - postprocess_start
        
        if self.enable_profiling:
            gpu_mem, cpu_mem = self.memory_profiler.log_memory("后处理后")
            performance_stats['memory_usage'] = {
                'gpu_memory': gpu_mem,
                'cpu_memory': cpu_mem
            }
        
        total_end_time = time.perf_counter()
        performance_stats['total_time'] = total_end_time - total_start_time
        
        # 计算吞吐率
        performance_stats['throughput'] = 1 / performance_stats['inference_time']
        
        # FLOPs估算 - 使用实际的扩散步数
        if self.model_config:
            flops = self.compute_profiler.estimate_flops(self.model_config, self.num_points, self.diffusion_steps)
            performance_stats['flops_estimate'] = flops
            performance_stats['compute_efficiency'] = flops / 1e9 / performance_stats['inference_time']  # GFLOPs/s
        
        # 打印性能统计
        self._print_performance_stats(performance_stats)
        
        return results, preprocessed_pc, preprocessed_normals, performance_stats
    
    def predict(self, point_cloud: np.ndarray, 
                normals: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        对输入点云进行吸取点预测
        
        参数:
            point_cloud: 点云坐标 (N, 3) numpy数组
            normals: 点云法向量 (N, 3) numpy数组，可选
            
        返回:
            results: 预测结果字典，包含各种吸取评分
            preprocessed_pc: 预处理后的点云坐标
        """
        if self.enable_profiling:
            results, preprocessed_pc, _ = self.predict_with_profiling(point_cloud, normals)
            return results, preprocessed_pc
        
        print("开始进行吸取点预测...")
        start_time = time.time()
        
        # 预处理输入数据
        inputs = self.preprocess_point_cloud(point_cloud, normals)
        
        # 获取预处理后的点云坐标（用于后续的最佳点定位）
        preprocessed_pc = inputs['point_clouds'].squeeze(0).cpu().numpy()  # (num_points, 3)
        preprocessed_normals = inputs['labels']['normals'].squeeze(0).cpu().numpy()  # (num_points, 3)
        
        # 临时修改模型的pipeline调用，使用动态点数
        original_forward = self.model.forward
        
        def patched_forward(inputs_dict):
            """修补的forward函数，支持动态点数"""
            batch_size = inputs_dict['point_clouds'].shape[0]
            num_point = inputs_dict['point_clouds'].shape[1]
            
            # pointnet++提取堆叠场景点云
            input_points = inputs_dict['point_clouds']
            input_points = torch.cat((input_points, inputs_dict['labels']['normals']), dim=2)
            features, global_features = self.model.backbone(input_points)

            # 推理模式
            pred_results = self.model.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(num_point, 4),  # 使用动态点数
                features=features,
                num_inference_steps=self.model.diffusion_inference_steps,
            )
            ddim_loss = None
            return pred_results, ddim_loss
        
        # 暂时替换forward方法
        self.model.forward = patched_forward
        
        try:
            # 模型推理
            with torch.no_grad():
                pred_results, _ = self.model(inputs)
        finally:
            # 恢复原始forward方法
            self.model.forward = original_forward
        
        inference_time = time.time() - start_time
        print(f"推理完成，用时: {inference_time:.3f}秒")
        
        # 后处理结果
        results = self.postprocess_results(pred_results)
        
        return results, preprocessed_pc, preprocessed_normals
    
    def _print_performance_stats(self, stats):
        """打印性能统计信息"""
        print(f"\n{'='*60}")
        print(f"性能统计报告")
        print(f"{'='*60}")
        
        # 时间统计
        total_time = stats['total_time']
        print(f"总耗时: {total_time:.4f}秒")
        print(f"  - 数据预处理: {stats['preprocess_time']:.4f}秒 ({stats['preprocess_time']/total_time*100:.1f}%)")
        print(f"  - 模型推理: {stats['inference_time']:.4f}秒 ({stats['inference_time']/total_time*100:.1f}%)")
        print(f"  - 后处理: {stats['postprocess_time']:.4f}秒 ({stats['postprocess_time']/total_time*100:.1f}%)")
        
        # 吞吐率
        print(f"推理吞吐率: {stats['throughput']:.2f} samples/sec")
        
        # 性能分析和建议
        inference_time = stats['inference_time']
        print(f"\n推理性能分析:")
        if inference_time > 0.5:
            print(f"  ⚠️  推理时间较长 ({inference_time:.3f}s)，可能的原因:")
            print(f"      - 扩散步数过多 (当前: {self.diffusion_steps}步)")
            print(f"      - 建议减少到10-20步以提升速度")
        elif inference_time > 0.1:
            print(f"  ✓  推理时间正常 ({inference_time:.3f}s)")
        else:
            print(f"  ✅ 推理速度很快 ({inference_time:.3f}s)")
            
        # 每步扩散的平均时间
        if hasattr(self, 'diffusion_steps') and self.diffusion_steps > 0:
            time_per_step = inference_time / self.diffusion_steps
            print(f"  - 每步扩散平均时间: {time_per_step:.4f}秒")
            if time_per_step > 0.02:
                print(f"    ⚠️  每步时间较长，考虑优化模型或减少步数")
        
        # 内存使用
        if 'memory_usage' in stats and stats['memory_usage']:
            gpu_mem = stats['memory_usage'].get('gpu_memory')
            cpu_mem = stats['memory_usage'].get('cpu_memory')
            
            if gpu_mem:
                print(f"GPU内存峰值: {gpu_mem['max_allocated']:.2f}GB")
                print(f"GPU内存已分配: {gpu_mem['allocated']:.2f}GB")
            if cpu_mem:
                print(f"CPU内存使用: {cpu_mem:.2f}GB")
        
        # 计算效率
        if stats['flops_estimate'] > 0:
            print(f"估算FLOPs: {stats['flops_estimate']/1e9:.2f} GFLOPs")
            print(f"计算效率: {stats['compute_efficiency']:.2f} GFLOPs/s")
        
        print(f"{'='*60}")
    
    def postprocess_results(self, pred_results: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        后处理模型预测结果
        
        参数:
            pred_results: 模型输出张量 (1, num_points, 4)
            
        返回:
            results: 处理后的结果字典
        """
        # 移除batch维度并转换为numpy
        pred_results = pred_results.squeeze(0).cpu().numpy()  # (num_points, 4)
        
        print(f"预测结果形状: {pred_results.shape}")
        print(f"各评分范围:")
        
        results = {
            'normal_flip_mask': pred_results[:, 0],      # 密封评分
            'wrench_scores': pred_results[:, 1],    # 扭矩评分  
            'feasibility_scores': pred_results[:, 2], # 可行性评分
            'visibility_scores': pred_results[:, 3]         # 物体尺寸评分
        }
        
        # 打印各评分的统计信息
        for key, values in results.items():
            print(f"  {key}: [{values.min():.3f}, {values.max():.3f}], 均值: {values.mean():.3f}")
        
        return results
    
    def get_best_suction_points(self, 
                               results: Dict[str, np.ndarray],
                               preprocessed_point_cloud: np.ndarray,  # 使用预处理后的点云
                               preprocessed_normals: np.ndarray,  # 使用预处理后的法向量
                               top_k: int = 10) -> List[Dict]:
        """
        获取最佳吸取点
        
        参数:
            results: 预测结果字典
            preprocessed_point_cloud: 预处理后的点云坐标（与预测结果对应）
            top_k: 返回前k个最佳点
            score_weights: 各评分权重
            
        返回:
            best_points: 最佳吸取点列表
        """

        
        # 计算综合评分
        composite_score = results['wrench_scores'] * results['feasibility_scores'] * results['visibility_scores']
        normal_flip_mask = results['normal_flip_mask'] > 0.5
        
        # 翻转向量（复制一份避免修改原数据）
        flipped_normals = preprocessed_normals.copy()
        flipped_normals[normal_flip_mask] = flipped_normals[normal_flip_mask] * -1
        
        # 构建包含所有信息的数据结构，用于NMS
        # 数据格式: [composite_score, nx, ny, nz, x, y, z, wrench_score, feasibility_score, visibility_score, normal_flipped_flag]
        suction_group = np.concatenate([
            composite_score[..., np.newaxis],                      # [0] 综合评分
            flipped_normals,                                       # [1:4] 法向量 (nx, ny, nz)
            preprocessed_point_cloud,                              # [4:7] 位置 (x, y, z)
            results['wrench_scores'][..., np.newaxis],            # [7] 扭矩评分
            results['feasibility_scores'][..., np.newaxis],       # [8] 可行性评分
            results['visibility_scores'][..., np.newaxis],        # [9] 可见性评分
            normal_flip_mask.astype(np.float32)[..., np.newaxis]  # [10] 法向量是否翻转
        ], axis=-1)
        
        print(f"NMS前吸取点数据形状: {suction_group.shape}")  # 应该是 (N, 11): [score, nx, ny, nz, x, y, z, wrench, feasibility, visibility, flipped]
        
        # 使用前7列进行NMS (只需要评分、法向量、位置信息)
        suction_group_for_nms = suction_group[:, :7]
        suction_group_nms = nms_suction(suction_group_for_nms, 0.02, 181.0 / 180 * np.pi)
        
        if len(suction_group_nms) == 0:
            print("⚠️  NMS后没有剩余的吸取点")
            return []
        
        # 将NMS结果转换为numpy数组，并找到对应的完整信息
        nms_indices = []
        for nms_point in suction_group_nms:
            # 找到NMS结果中每个点在原始数据中的索引
            distances = np.sum((suction_group[:, :7] - nms_point) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            nms_indices.append(closest_idx)
        
        # 获取NMS后的完整数据 (包含所有原始评分)
        suction_group_nms_full = suction_group[nms_indices]
        print(f"吸取点NMS后数量: {suction_group_nms_full.shape[0]}")
        
        # 确保数据结构正确
        if suction_group_nms_full.shape[1] != 11:
            print(f"⚠️  NMS后数据结构异常: 期望11列，实际{suction_group_nms_full.shape[1]}列")
            return []
        
        top_k = suction_group_nms_full.shape[0]
        
        # 确保评分列存在且有效
        scores = suction_group_nms_full[:, 0]
        if np.all(np.isnan(scores)) or np.all(scores <= 0):
            print("⚠️  所有评分都无效或为零")
            return []
            
        # 获取top-k索引 (按评分从高到低排序)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        best_points = []
        
        for i, idx in enumerate(top_indices):
            try:
                suction_data = suction_group_nms_full[idx]
                
                # 解析完整的数据结构: [score, nx, ny, nz, x, y, z, wrench, feasibility, visibility, flipped]
                composite_score_val = suction_data[0]
                normal = suction_data[1:4]  # [nx, ny, nz]
                position = suction_data[4:7]  # [x, y, z]
                wrench_score = suction_data[7]
                feasibility_score = suction_data[8]
                visibility_score = suction_data[9]
                normal_flipped = bool(suction_data[10] > 0.5)
                
                # 验证数据有效性
                if np.isnan(composite_score_val) or composite_score_val <= 0:
                    print(f"跳过无效评分的点: {composite_score_val}")
                    continue
                    
                if np.any(np.isnan(normal)) or np.any(np.isnan(position)):
                    print(f"跳过含有NaN值的点")
                    continue
                
                # 构建吸取点信息字典
                point_info = {
                    'rank': i + 1,                           # 排名 (1-based)
                    'position': position.tolist(),           # 3D位置坐标 [x, y, z]
                    'normal': normal.tolist(),               # 法向量 [nx, ny, nz]
                    'composite_score': float(composite_score_val),  # 综合评分
                    'original_scores': {                     # 原始各项评分 (直接从数据中获取)
                        'wrench_score': float(wrench_score),
                        'feasibility_score': float(feasibility_score),
                        'visibility_score': float(visibility_score),
                        'normal_flipped': normal_flipped
                    },
                    'quality_metrics': {
                        'normal_magnitude': float(np.linalg.norm(normal)),    # 法向量模长
                        'score_confidence': float(composite_score_val / (1.0 + np.std(scores)))  # 评分置信度
                    }
                }
                
                best_points.append(point_info)
                
            except Exception as e:
                print(f"处理第{i+1}个吸取点时出错: {e}")
                continue
        
        # 重新排序，确保按实际评分排序
        best_points.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 更新排名
        for i, point in enumerate(best_points):
            point['rank'] = i + 1
        
        # 打印最佳吸取点信息
        print(f"\n找到 {len(best_points)} 个最佳吸取点:")
        for i, point in enumerate(best_points[:5]):  # 只打印前5个
            print(f"  第{point['rank']}名:")
            print(f"    位置: [{point['position'][0]:.4f}, {point['position'][1]:.4f}, {point['position'][2]:.4f}]")
            print(f"    法向量: [{point['normal'][0]:.4f}, {point['normal'][1]:.4f}, {point['normal'][2]:.4f}]")
            print(f"    综合评分: {point['composite_score']:.4f}")
            print(f"    原始评分: 扭矩={point['original_scores']['wrench_score']:.3f}, "
                  f"可行性={point['original_scores']['feasibility_score']:.3f}, "
                  f"可见性={point['original_scores']['visibility_score']:.3f}")
            print(f"    质量指标: 法向量模长={point['quality_metrics']['normal_magnitude']:.4f}, "
                  f"评分置信度={point['quality_metrics']['score_confidence']:.4f}")
            if i < len(best_points) - 1:
                print()

        
        return best_points
    
    def save_results(self, 
                    results: Dict[str, np.ndarray],
                    point_cloud: np.ndarray,
                    output_path: str,
                    best_points: Optional[List[Dict]] = None,
                    performance_stats: Optional[Dict] = None):
        """
        保存预测结果到文件
        
        参数:
            results: 预测结果
            point_cloud: 点云坐标
            output_path: 输出文件路径
            best_points: 最佳吸取点信息
            performance_stats: 性能统计信息
        """
        print(f"正在保存结果到: {output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为numpy格式
        np.savez(output_path, 
                point_cloud=point_cloud,
                **results)
        
        # 如果有最佳点信息，保存为JSON
        if best_points is not None:
            json_path = output_path.replace('.npz', '_best_points.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(best_points, f, indent=2, ensure_ascii=False)
            print(f"最佳吸取点信息已保存到: {json_path}")
        
        # 如果有性能统计信息，保存为JSON
        if performance_stats is not None:
            perf_path = output_path.replace('.npz', '_performance.json')
            # 序列化性能统计信息
            serializable_stats = {}
            for key, value in performance_stats.items():
                if isinstance(value, dict):
                    serializable_stats[key] = value
                else:
                    serializable_stats[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
            
            with open(perf_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            print(f"性能统计信息已保存到: {perf_path}")
        
        print("结果保存完成")


def benchmark_model(inferencer, test_cases=5):
    """模型基准测试"""
    print(f"\n开始基准测试（{test_cases}个测试案例）...")
    
    # 生成测试数据
    test_data = []
    for _ in range(test_cases):
        pc = np.random.randn(16384, 3).astype(np.float32)
        normals = np.random.randn(16384, 3).astype(np.float32)
        # 归一化法向量
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        test_data.append((pc, normals))
    
    times = []
    memory_usage = []
    
    for i, (pc, normals) in enumerate(test_data):
        print(f"\n测试案例 {i+1}/{test_cases}")
        
        start_time = time.perf_counter()
        if inferencer.enable_profiling:
            results, _, perf_stats = inferencer.predict_with_profiling(pc, normals)
            times.append(perf_stats['inference_time'])
        else:
            results, _ = inferencer.predict(pc, normals)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"基准测试结果总结")
    print(f"{'='*60}")
    print(f"平均推理时间: {np.mean(times):.4f} ± {np.std(times):.4f} 秒")
    print(f"最快推理时间: {np.min(times):.4f} 秒")
    print(f"最慢推理时间: {np.max(times):.4f} 秒")
    print(f"平均吞吐率: {1/np.mean(times):.2f} samples/sec")
    
    if memory_usage:
        print(f"平均GPU内存使用: {np.mean(memory_usage):.2f} ± {np.std(memory_usage):.2f} GB")
        print(f"最大GPU内存使用: {np.max(memory_usage):.2f} GB")
    print(f"{'='*60}")


def load_data_from_images(rgb_path: str, depth_path: str, depth_scale: str, mask_path: str, camera_info_path: str, params_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从RGB图像、深度图像、掩码图像和相机信息文件加载点云数据
    
    参数:
        rgb_path: RGB图像文件路径
        depth_path: 深度图像文件路径  
        mask_path: 掩码图像文件路径
        camera_info_path: 相机信息文件路径
        params_path: 参数配置文件路径，可选
        
    返回:
        point_cloud: 点云坐标 (N, 3)
        normals: 法向量 (N, 3)
    """
    print(f"正在从图像文件加载数据:")
    print(f"  RGB图像: {rgb_path}")
    print(f"  深度图像: {depth_path}")
    print(f"  深度单位: {depth_scale}")
    print(f"  掩码图像: {mask_path}")
    print(f"  相机信息: {camera_info_path}")
    
    # 检查文件是否存在
    for file_path in [rgb_path, depth_path, mask_path, camera_info_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 使用预处理函数生成点云和法向量
    try:
        point_cloud, normals = preprocess(rgb_path, depth_path, depth_scale, mask_path, camera_info_path, params_path)
        print(f"成功生成点云: {point_cloud.shape}, 法向量: {normals.shape}")
        return point_cloud, normals
    except Exception as e:
        raise RuntimeError(f"预处理失败: {e}")


def main():
    """主函数 - 命令行推理接口"""
    parser = argparse.ArgumentParser(description='扩散吸取网络推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点文件路径')
    
    # 输入图像参数
    parser.add_argument('--rgb', type=str, required=True,
                       help='RGB图像文件路径')
    parser.add_argument('--depth', type=str, required=True,
                       help='深度图像文件路径')
    parser.add_argument('--mask', type=str, required=True,
                       help='掩码图像文件路径')
    parser.add_argument('--camera_info', type=str, required=True,
                       help='相机信息文件路径')
    parser.add_argument('--params', type=str, 
                       help='参数配置文件路径')
    parser.add_argument('--depth_scale', type=str, default='mm', choices=['mm', 'normalized'], help='深度单位')
    parser.add_argument('--output', type=str, default='./inference_results.npz',
                       help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--num_points', type=int, default=16384,
                       help='点云采样点数')
    parser.add_argument('--diffusion_steps', type=int, default=20,
                       help='扩散推理步数')
    
    # 性能分析选项
    parser.add_argument('--enable_profiling', action='store_true',
                       help='启用详细性能分析')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    parser.add_argument('--benchmark_cases', type=int, default=5,
                       help='基准测试案例数量')
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"参数解析失败: {e}")
        parser.print_help()
        sys.exit(1)
    
    print("=" * 60)
    print("扩散吸取网络推理系统")
    print("=" * 60)
    
    # 创建推理器
    inferencer = SuctionNetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_points=args.num_points,
        diffusion_steps=args.diffusion_steps,
        enable_profiling=args.enable_profiling
    )
    
    # 从图像文件加载数据
    print(f"\n正在从图像文件加载数据...")
    point_cloud, normals = load_data_from_images(
        args.rgb, args.depth, args.depth_scale, args.mask, args.camera_info, args.params
    )
    
    # 执行推理
    print(f"\n开始推理...")
    performance_stats = None
    
    if args.enable_profiling:
        results, preprocessed_pc, preprocessed_normals, performance_stats = inferencer.predict_with_profiling(point_cloud, normals)
    else:
        results, preprocessed_pc, preprocessed_normals = inferencer.predict(point_cloud, normals)
    
    # 获取最佳吸取点
    print(f"\n计算最佳吸取点...")
    best_points = inferencer.get_best_suction_points(
        results, preprocessed_pc, preprocessed_normals
    )
    
    # 简要总结最佳吸取点（详细信息已在get_best_suction_points中打印）
    if best_points:
        print(f"\n✅ 成功找到 {len(best_points)} 个最佳吸取点")
        print(f"最高评分: {best_points[0]['composite_score']:.4f}")
        print(f"最低评分: {best_points[-1]['composite_score']:.4f}")
    else:
        print(f"\n⚠️  未找到合适的吸取点")
    
    # 保存结果
    print(f"\n保存结果...")
    inferencer.save_results(results, preprocessed_pc, args.output, best_points, performance_stats)
    
    # 保存原始点云为txt文件
    output_dir = os.path.dirname(args.output)
    output_name = os.path.splitext(os.path.basename(args.output))[0]
    point_cloud_txt = os.path.join(output_dir, f"{output_name}_original_pointcloud.txt")
    
    print(f"保存原始点云到: {point_cloud_txt}")
    # 保存格式: x y z nx ny nz (坐标 + 法向量)
    if normals is not None:
        combined_data = np.hstack([point_cloud, normals])
        header = "x y z nx ny nz"
    else:
        combined_data = point_cloud
        header = "x y z"
    
    np.savetxt(point_cloud_txt, combined_data, fmt='%.6f', header=header, comments='# ')
    print(f"原始点云已保存到: {point_cloud_txt}")
    
    # 运行基准测试
    if args.benchmark:
        benchmark_model(inferencer, test_cases=args.benchmark_cases)
    
    print(f"\n推理完成！结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
