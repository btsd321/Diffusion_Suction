# 本文件实现DiffusionSutionNet的预处理功能，主要包括以下步骤：
# 1. 读取输入RGB图像(PNG格式)、深度图像(PNG格式)、Mask图像(PNG格式)、相机信息文件(YAML格式)
# 2. 计算点云及其对应的法向量

import os
import cv2
import numpy as np
import yaml
import json
import torch
import camera_info
# PointNet2操作库，用于点云采样
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d  # 3D几何处理

INPUT_TARGET_POINT_NUM = 16384  # 输入点云的目标点数

# 默认路径，可以被函数参数覆盖
default_params_path = '/home/lixinlong/Project/pose_detect_train/example/input/parameter.json'

def _load_parameters(params_file_name):
    """
    加载配置文件
    
    参数:
        params_file_name (str): JSON配置文件路径
        
    返回:
        dict: 包含深度范围等配置的字典
    """
    params = {}
    with open(params_file_name, 'r') as f:
        config = json.load(f)
        params = config
    return params 

def _depth_to_pointcloud_optimized(cam_info, params, us, vs, zs, to_mm=False, depth_scale="normalized"):
    """
    将深度图像像素坐标转换为3D点云坐标
    
    使用相机内参进行投影变换，从2D像素坐标和深度值重建3D空间坐标。
    这是整个数据处理管道的基础步骤。
    
    参数:
        us (numpy.ndarray): u坐标数组（像素水平坐标）
        vs (numpy.ndarray): v坐标数组（像素垂直坐标）  
        zs (numpy.ndarray): 深度值数组（归一化深度）
        to_mm (bool): 是否转换为毫米单位，默认False（米单位）
        xyz_limit (list): 3D空间裁剪范围，格式[[xmin,xmax], [ymin,ymax], [zmin,zmax]]
                            用于过滤工作空间外的点
    
    返回:
        numpy.ndarray: 3D点云坐标，形状为(N, 3)
    """
    assert len(us) == len(vs) == len(zs), "坐标数组长度必须一致"
    
    # 从参数配置中获取相机内参
    fx = cam_info.intrinsic_matrix[0, 0]
    fy = cam_info.intrinsic_matrix[1, 1]
    cx = cam_info.intrinsic_matrix[0, 2]  # x方向主点坐标
    cy = cam_info.intrinsic_matrix[1, 2]  # y方向主点坐标
    clip_start = params['clip_start']  # 近裁剪面距离
    clip_end = params['clip_end']      # 远裁剪面距离
    
    # 将归一化深度值转换为真实距离（米）
    # 深度图中的值通常是归一化的，需要映射到真实距离范围
    if depth_scale == "normalized":
        Zline = clip_start + (zs/params['max_val_in_depth']) * (clip_end - clip_start)
    elif depth_scale == "mm":
        Zline = zs / 1000  # 单位转换为米
    else: 
        raise ValueError("不支持的深度单位")
    
    # 考虑透视投影的距离校正
    # 校正由于透视投影导致的距离失真
    Zcs = Zline/np.sqrt(1+ np.power((us-cx)/fx,2) + np.power((vs-cy)/fy,2))
    
    # 可选：转换为毫米单位（某些应用需要）
    if to_mm:
        Zcs *= 1000
        
    # 使用针孔相机模型进行3D重建
    # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    Xcs = (us - cx) * Zcs / fx
    Ycs = (vs - cy) * Zcs / fy
    
    # 重塑为列向量并组合成点云
    Xcs = np.reshape(Xcs, (-1, 1))
    Ycs = np.reshape(Ycs, (-1, 1))
    Zcs = np.reshape(Zcs, (-1, 1))
    points = np.concatenate([Xcs, -Ycs, -Zcs], axis=-1)
    
    
    return points

def depth_scale_to_mm(depth_img, scale, params_path=None):
    '''
    将归一化深度图像转换为毫米单位的深度图
    '''
    if params_path is None:
        params_path = default_params_path
    
    # 读取参数配置
    params = _load_parameters(params_path)
    clip_start = params['clip_start']  # 近裁剪面距离
    clip_end = params['clip_end']      # 远裁剪面距离
    max_val_in_depth = params['max_val_in_depth']  # 深度值最大值
    depth_img_array = np.array(depth_img, dtype=np.float32)
    if scale == "normalized":
        # 转换为毫米单位
        depth_img = np.floor(((depth_img_array / max_val_in_depth * (clip_end - clip_start) + clip_start) * 1000 + 0.5)).astype(np.uint16)
    elif scale == "mm":
        pass
    else:
        raise ValueError("不支持的深度单位")
    return depth_img

def depth_scale_to_normalized(depth_img, depth_scale, params_path=None):
    if params_path is None:
        params_path = default_params_path
    
    # 读取参数配置
    params = _load_parameters(params_path)
    clip_start = params['clip_start']  # 近裁剪面距离
    clip_end = params['clip_end']      # 远裁剪面距离
    max_val_in_depth = params['max_val_in_depth']  # 深度值最大值
    
    # 转换为归一化深度值
    if depth_scale == "normalized":
        pass
    elif depth_scale == "mm":
        depth_img = (depth_img / 1000. - clip_start) / (clip_end - clip_start) * max_val_in_depth
    else:
        raise ValueError("不支持的深度单位")
    return depth_img

def filter_point_cloud(points, nb_points: int = 16, filter_radius: float = 0.01, z_threshold: float = 0.7):
    try:
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 应用半径滤波
        pcd_filtered, inlier_indices_radius = pcd.remove_radius_outlier(
            nb_points, filter_radius
        )
        
        # 转换回numpy数组
        radius_filtered_points = np.asarray(pcd_filtered.points)
        
        # 创建正确的索引掩码
        inlier_mask1 = np.zeros(points.shape[0], dtype=bool)
        inlier_mask1[inlier_indices_radius] = True
        
        # print(f"半径滤波: 原始点数 {self._origin_points.shape[0]}, 过滤后点数 {radius_filtered_points.shape[0]}")
        # print(f"掩码统计: True={np.sum(inlier_mask1)}, False={np.sum(~inlier_mask1)}")
        # self._obj_ids_after_filter_1 = self._orgin_obj_ids[inlier_mask1]  # 滤波后点云中每个点对应的物体ID

        # 计算法向量（只对过滤后的点）
        radius_filter_normals = None
        if radius_filtered_points.shape[0] > 0:
            pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(radius_filtered_points))
            pc_o3d.estimate_normals(
                o3d.geometry.KDTreeSearchParamRadius(0.015), 
                fast_normal_computation=False
            )
            
            pc_o3d.normalize_normals()
            
            radius_filter_normals = np.array(pc_o3d.normals).astype(np.float32)
        else:
            radius_filter_normals = np.array([]).reshape(0, 3)

        # 直通滤波
        # 创建Z轴过滤掩码：保留Z坐标小于等于阈值的点
        z_filter_mask = radius_filtered_points[:, 2] <= z_threshold
        
        # 过滤点云和法向量
        z_filtered_points = radius_filtered_points[z_filter_mask]
        z_filtered_normals = radius_filter_normals[z_filter_mask]
        
        # 更新累积过滤掩码
        # 创建新的累积掩码，标记哪些原始点被保留
        filter_mask = np.zeros_like(inlier_mask1, dtype=bool)
        
        # 在之前被保留的点中，进一步标记通过Z轴过滤的点
        z_passed_positions = np.where(z_filter_mask)[0]
        z_passed_indices = np.array(inlier_indices_radius)[z_passed_positions]
        z_passed_indices = z_passed_indices.astype(int)
        filter_mask[z_passed_indices] = True
        
        return z_filtered_points, z_filtered_normals, filter_mask
    except Exception as e:
        print(f"半径滤波失败，使用原始点云: {e}")
        identity_mask = np.ones(points.shape[0], dtype=bool)
        dummy_normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        dummy_normals[:, 2] = -1
        return points, dummy_normals, identity_mask
    
    
def resample(points, normals, output_points_num=INPUT_TARGET_POINT_NUM):
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    if not normals.flags['C_CONTIGUOUS']:
        normals = np.ascontiguousarray(normals)
    if points.shape[0] == output_points_num:
        return points, normals
    elif points.shape[0] < 100:
        raise ValueError(f"点云数量({points.shape[0]})过小，请检查数据！")

    try:
        if points.shape[0] > output_points_num:
            points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float()
            points_transpose = points_transpose.cuda()
            # 执行最远点采样，保持点云的几何分布
            sampled_idx = furthest_point_sample(points_transpose, output_points_num).cpu().numpy().reshape(output_points_num)
            return points[sampled_idx], normals[sampled_idx]
        else:
            # 上采样，直接重复点云
            t = int(1.0 * output_points_num / points.shape[0]) + 1
            points_tile = np.tile(points, (t, 1))
            output_points = points_tile[:output_points_num]
            normals_tile = np.tile(normals, (t, 1))
            output_normals = normals_tile[:output_points_num]
            return output_points, output_normals
    except Exception as e:
        print(e)
        raise RuntimeError(f"点云重采样失败: {e}")

def preprocess(input_rgb_path, input_depth_path, depth_scale, input_mask_path, camera_info_path, params_path=None, z_threshold=0.7):
    if params_path is None:
        params_path = default_params_path
    # 读取RGB图像
    rgb_img = cv2.imread(input_rgb_path)
    # 读取深度图像
    print(f'input_depth_path: {input_depth_path}')
    depth_img = cv2.imread(input_depth_path, cv2.IMREAD_ANYDEPTH)
    # 读取Mask图像，白色为物体，其他为背景
    print(f'input_mask_path: {input_mask_path}')
    mask_img = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"无法读取Mask图像: {input_mask_path}，请检查路径和文件格式是否正确！")
    print(f'mask_img shape: {mask_img.shape}, dtype: {mask_img.dtype}')
    # # 可视化mask
    # cv2.imshow('Mask Image', mask_img)
    # cv2.waitKey(0)
    valid_mask = mask_img == 255   # 白色区域为物体
    
    # # 可视化valid_mask
    # cv2.imshow('Valid Mask', valid_mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    
    # 读取相机信息
    cam_info = camera_info.get_camera_info_from_yaml(camera_info_path)
    
    # 读取参数配置
    params = _load_parameters(params_path)
    
    # 生成点云及其法向量
    xs, ys = np.where(valid_mask)
    zs = depth_img[valid_mask]
    
    # 执行3D重建：像素坐标 + 深度 → 3D点云
    raw_points_in_camera = _depth_to_pointcloud_optimized(cam_info, params, xs, ys, zs, to_mm=False, depth_scale=depth_scale)
      
    # 半径滤波之后计算法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points_in_camera)
    
    pcd_filtered, inlier_indices_radius = pcd.remove_radius_outlier(
        nb_points=16, radius=0.01
    )
    # 转换回numpy数组
    points_in_camera = np.asarray(pcd_filtered.points)
    
    # 计算法向量（只对过滤后的点）
    normals_in_camera = None
    if points_in_camera.shape[0] > 0:
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_in_camera))
        pc_o3d.estimate_normals(
            o3d.geometry.KDTreeSearchParamRadius(0.015), 
            fast_normal_computation=False
        )
        
        pc_o3d.normalize_normals()
        
        normals_in_camera = np.array(pc_o3d.normals).astype(np.float32)
    else:
        normals_in_camera = np.array([]).reshape(0, 3)
        
    # 转换到世界坐标系
    points_in_world = None
    normals_in_world = None
    # 将3D点扩展为齐次坐标（添加第4维度为1）
    ones = np.ones((points_in_camera.shape[0], 1))
    points_homo = np.hstack((points_in_camera, ones))
    #外参矩阵为W2C矩阵即世界坐标系转相机坐标系的矩阵，因此需要先计算C2W
    c2w = np.linalg.inv(cam_info.extrinsic_matrix)
    points_world_homo = c2w @ points_homo.T # 形状: (4, N)
    points_in_world = (points_world_homo.T)[:, :3]  # 形状: (N, 3)
    # 法向量只需要旋转变换，提取旋转矩阵部分
    rotation_matrix = cam_info.extrinsic_matrix[:3, :3].T
    normals_in_world = (rotation_matrix @ normals_in_camera.T).T  # 形状: (N, 3)
    
    # 剔除points_in_world中Z轴大于z_threshold的点
    mask = points_in_world[:, 2] < z_threshold
    origin_points = points_in_world[mask]
    origin_normals = normals_in_world[mask]
    
    # 重采样到16384个点
    return resample(origin_points, origin_normals, output_points_num=INPUT_TARGET_POINT_NUM)
    
    
    

