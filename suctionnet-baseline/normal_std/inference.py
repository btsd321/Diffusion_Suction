import os
import cv2
# import pcl
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
from policy import estimate_suction

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='数据集划分 [默认: test_seen]')
parser.add_argument('--camera', default='kinect', help='使用的相机类型 [默认: kinect]')
parser.add_argument('--save_root', default='/DATA2/Benchmark/suction/inference_results/normals_std', help='推理结果保存目录')
parser.add_argument('--dataset_root', default='/DATA2/Benchmark/graspnet', help='数据集根目录')
parser.add_argument('--save_visu', action='store_true', help='是否保存可视化结果')
FLAGS = parser.parse_args()

split = FLAGS.split
camera = FLAGS.camera
dataset_root = FLAGS.dataset_root
save_root = FLAGS.save_root

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        # 相机内参信息
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def uniform_kernel(kernel_size):
    # 生成均值卷积核
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    # center = kernel_size // 2
    kernel = kernel / kernel_size**2

    return kernel

def grid_sample(pred_score_map, down_rate=20, topk=512):
    # 对预测分数图进行网格采样, 返回topk分数及其像素坐标
    num_row = pred_score_map.shape[0] // down_rate
    num_col = pred_score_map.shape[1] // down_rate

    idx_list = []
    for i in range(num_row):
        for j in range(num_col):
            pred_score_grid = pred_score_map[i*down_rate:(i+1)*down_rate, j*down_rate:(j+1)*down_rate]
            # print('pred_score_grid:', pred_score_grid.shape)
            max_idx = np.argmax(pred_score_grid)
            
            max_idx = np.array([max_idx // down_rate, max_idx % down_rate]).astype(np.int32)
            
            max_idx[0] += i*down_rate
            max_idx[1] += j*down_rate
            # print('max_idx:', max_idx)
            idx_list.append(max_idx[np.newaxis, ...])
    
    idx = np.concatenate(idx_list, axis=0)
    # print('idx:', idx.shape)
    suction_scores = pred_score_map[idx[:, 0], idx[:, 1]]
    # print('suction_scores:', suction_scores.shape)
    sort_idx = np.argsort(suction_scores)
    sort_idx = sort_idx[::-1]

    sort_idx_topk = sort_idx[:topk]

    suction_scores_topk = suction_scores[sort_idx_topk]
    idx0_topk = idx[:, 0][sort_idx_topk]
    idx1_topk = idx[:, 1][sort_idx_topk]

    return suction_scores_topk, idx0_topk, idx1_topk

def drawGaussian(img, pt, score, sigma=1):
    """
    在输入图像上绘制二维高斯分布
    参数说明
    ----------
    img: torch.Tensor 或 numpy.ndarray
        输入图像, 形状为 (H, W)。
    pt: list or tuple
        点坐标 (x, y)。
    score: float
        高斯分布的强度(权重)。
    sigma: int
        高斯分布的标准差。
    返回
    -------
    torch.Tensor 或 numpy.ndarray
        形状为 (H, W) 的张量。
    """
    tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    tmpSize = 3 * sigma
    # 检查高斯分布是否在图像范围内
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # 如果高斯分布完全超出图像边界, 直接返回原图
        return img

    # 生成高斯分布
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # 有效高斯区域
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # 图像区域
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    img += tmp_img

def inference(scene_idx):
    # 针对指定scene的所有帧进行推理
    for anno_idx in range(256):

        rgb_file = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_file = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        segmask_file = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_file = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        # 读取深度图和分割掩码
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        seg_mask = cv2.imread(segmask_file, cv2.IMREAD_UNCHANGED).astype(np.bool)
        # 读取相机内参
        meta = scio.loadmat(meta_file)
        intrinsics = meta['intrinsic_matrix']
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        width = 1280
        height = 720
        s = 1000.0
        camera_info = CameraInfo(width, height, fx, fy, cx, cy, s)

        # 估算吸取分数热力图、法向量和点云
        heatmap, normals, point_cloud = estimate_suction(depth, seg_mask, camera_info)
        
        # 对热力图进行均值卷积平滑
        k_size = 15
        kernel = uniform_kernel(k_size)
        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        heatmap = np.pad(heatmap, k_size//2)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        heatmap = F.conv2d(heatmap, kernel).squeeze().numpy()

        # 网格采样, 选取topk吸取点
        suction_scores, idx0, idx1 = grid_sample(heatmap, down_rate=10, topk=1024)
        suction_directions = normals[idx0, idx1, :]
        suction_translations = point_cloud[idx0, idx1, :]

        # 拼接吸取点信息(分数、法向、三维坐标)
        suction_arr = np.concatenate([suction_scores[..., np.newaxis], suction_directions, suction_translations], axis=-1)

        # 保存吸取点结果
        suction_dir = os.path.join(save_root, split, 'scene_%04d'%scene_idx, camera, 'suction')
        os.makedirs(suction_dir, exist_ok=True)
        print('Saving:', suction_dir+'/%04d'%anno_idx+'.npz')
        np.savez(suction_dir+'/%04d'%anno_idx+'.npz', suction_arr)

        # 可视化保存
        if anno_idx < 3 and FLAGS.save_visu:
            # 可视化热力图叠加到RGB图像
            score_image = heatmap
            score_image *= 255
            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
            rgb_image = np.array(Image.open(rgb_file), dtype=np.float32)
            rgb_image = 0.5 * rgb_image + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)
            
            visu_dir = os.path.join(save_root, split, 'scene_%04d'%scene_idx, camera, 'visu')
            os.makedirs(visu_dir, exist_ok=True)
            print('Saving:', visu_dir+'/%04d'%anno_idx+'.png')
            im.save(visu_dir+'/%04d'%anno_idx+'.png')

            # 可视化采样吸取点
            score_image = np.zeros_like(heatmap)
            for i in range(suction_scores.shape[0]):
                drawGaussian(score_image, [idx1[i], idx0[i]], suction_scores[i], 3)
            
            score_image *= 255                        
            score_image = score_image.clip(0, 255)
            score_image = score_image.astype(np.uint8)
            score_image = cv2.applyColorMap(score_image, cv2.COLORMAP_RAINBOW)
            rgb_image = np.array(Image.open(rgb_file), dtype=np.float32)
            rgb_image = 0.5 * rgb_image + 0.5 * score_image
            rgb_image = rgb_image.astype(np.uint8)
            im = Image.fromarray(rgb_image)
            
            visu_dir = os.path.join(save_root, split, 'scene_%04d'%scene_idx, camera, 'visu')
            os.makedirs(visu_dir, exist_ok=True)
            print('Saving:', visu_dir+'/%04d_sampled'%anno_idx+'.png')
            im.save(visu_dir+'/%04d_sampled'%anno_idx+'.png')

if __name__ == "__main__":
    # 主程序入口, 根据split选择scene范围, 依次推理
    scene_list = []
    if split == 'test':
        for i in range(100, 190):
            scene_list.append(i)
    elif split == 'test_seen':
        for i in range(100, 130):
            scene_list.append(i)
    elif split == 'test_similiar':
        for i in range(130, 160):
            scene_list.append(i)
    elif split == 'test_novel':
        for i in range(160, 190):
            scene_list.append(i)
    else:
        print('invalid split')
    
    for scene_idx in scene_list:
        inference(scene_idx)
