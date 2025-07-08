# -*- coding:utf-8 -*-
"""

@author: Huang Dingtao
@checked: Huang Dingtao

"""
import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

import csv
import json
import numpy as np
import shutil
from math import radians
import math
import yaml
import csv
import cv2
CYCLE_idx_list = range(0, 100)
SCENE_idx_list = range(1, 51)  #  (1, 61)   1 2 3 4 .....59 60

OUTDIR_dir_segment_images_sigle =  os.path.join(FILE_DIR, 'segment_images_sigle')
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images')

individual_object_size =  os.path.join(FILE_DIR, 'individual_object_size')
if not os.path.exists(individual_object_size):
    os.makedirs(individual_object_size)

def render_scenes():        
    for cycle_id in CYCLE_idx_list:
        for scene_id in SCENE_idx_list:
            # 读取当前循环和场景下的分割图像
            image_ids = cv2.imread(os.path.join(os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id)),'Image0001.exr'),cv2.IMREAD_UNCHANGED)
            # 计算所有物体的掩码id
            mask_ids_all = np.round(image_ids[:,:, 1] * (scene_id - 1)).astype('int')

            areas_id = []
            for i in range(scene_id):
                # 读取当前物体的单独分割图像
                image_id = cv2.imread(os.path.join(os.path.join(OUTDIR_dir_segment_images_sigle, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id),"{:0>3}".format(scene_id)+"_{:0>3}".format(i)),'Image0001.exr'),cv2.IMREAD_UNCHANGED)
                # 获取当前物体的掩码
                mask_id = image_id[:,:, 2] == 1
                # 获取所有物体的掩码中属于当前物体的部分
                mask_ids = mask_ids_all == i
                # 只保留当前物体的掩码区域
                mask_ids[[image_ids[:,:, 2] != 1]] = 0
                if np.sum(mask_id)!=0:
                    # 计算当前物体的面积比例
                    areas_id.append(np.sum(mask_ids) / np.sum(mask_id))
                else:
                    areas_id.append(0)

            # 构建保存路径
            save_path = os.path.join(individual_object_size,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_loc = save_path + '/' + '{:0>3}'.format(scene_id) + '.csv'
            assert len(areas_id)==scene_id
            # 将面积比例写入csv文件
            with open(file_loc, 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(areas_id)

if __name__ == '__main__':
    import time
    start_time = time.time()  
    render_scenes()
    print(time.time() - start_time)
