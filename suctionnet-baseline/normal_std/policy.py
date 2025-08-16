import cv2
# import pcl
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    # 根据深度图和相机内参生成点云
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def stdFilt(img, wlen):
    '''
    计算图像的局部标准差(标准差滤波)
    :param img: 输入图像, 可以为多通道
    :param wlen: 滤波窗口大小
    :return: 标准差滤波后的图像
    '''
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    return np.sqrt(abs(wsqrmean - wmean*wmean))

def estimate_suction(depth_img, obj_mask, camera_info):
    # 估算吸取分数热力图、法向量和点云
    point_cloud = create_point_cloud_from_depth_image(depth_img, camera_info)
    # print('point_cloud:', point_cloud.shape)

    # 生成有效像素掩码(只在目标掩码包围盒区域内, 且深度不为0的像素为有效)
    valid_idx = np.zeros_like(obj_mask, dtype=np.bool)
    coord1, coord2 = np.nonzero(obj_mask)
    coord1_min, coord1_max = coord1.min(), coord1.max()
    coord2_min, coord2_max = coord2.min(), coord2.max()
    valid_idx[coord1_min:coord1_max+1, coord2_min:coord2_max+1] = 1
    valid_idx = valid_idx & (point_cloud[..., 2] != 0)
    height, width, _ = point_cloud.shape

    # 提取有效点云
    point_cloud_valid = point_cloud[valid_idx]
    # 构建Open3D点云对象
    pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_valid))
    # 估算点云法向量, KNN邻域为224
    pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(224), fast_normal_computation=False)
    # 法向量朝向统一为z轴负方向
    pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    pc_o3d.normalize_normals()
    normals = np.array(pc_o3d.normals).astype(np.float32)
    
    # 构建全图法向量图
    normal_map = np.zeros([height, width, 3], dtype=np.float32)
    normal_map[valid_idx] = normals

    # 计算法向量的局部标准差(反映表面平整度), 并取均值作为吸取分数
    # mean_normal_std = np.mean(generic_filter(normal_map, np.std, size=25), axis=2)
    mean_normal_std = np.mean(stdFilt(normal_map, 25), axis=2)
    # 归一化得到吸取分数热力图, 值越大表示越平整
    heatmap = 1 - mean_normal_std / np.max(mean_normal_std)
    heatmap[~valid_idx] = 0  # 无效区域分数设为0

    return heatmap, normal_map, point_cloud



