"""
作者: Huang Dingtao
校验: Huang Dingtao
"""

import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

import csv
import cv2
import math
import numpy as np
import os
import json
import nibabel.quaternions as nq
import yaml

CYCLE_idx_list = range(0, 100)
SCENE_idx_list = range(1, 51)

CAMERA_LOCATION = [0, 0, 1.70]
CAMERA_ROTATION = [0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,6.123234e-17]
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')
GT_PATH =  os.path.join(FILE_DIR, 'gt')
if not os.path.exists(GT_PATH):
    os.makedirs(GT_PATH)    

def read_csv(csv_path):        
    with open(csv_path,'r') as csv_file:  
        all_lines=csv.reader(csv_file) 
        list_file = [i for i in all_lines]  
    array_file = np.array(list_file)[1:] # 去除表头
    obj_name = array_file[:,0]
    obj_index = array_file[:,1].astype('int')
    pose = array_file[:,2:9].astype('float32')
    return obj_name, obj_index, pose

def generate_gt(pose_world):
    ''' 生成相机坐标系下的位姿

    参数:
        pose_world: 零件在世界坐标系下的位姿

    返回:
        pose_camera: 零件在相机坐标系下的位姿
    '''  

    # 相机外参
    t_c2w = np.array(CAMERA_LOCATION).reshape(1, 3)
    quat_c2w = np.array(CAMERA_ROTATION)
    R_c2w  = nq.quat2mat(quat_c2w).reshape(3,3)

    # 世界坐标系下的平移和旋转
    t_world = pose_world[:,:3]
    quat_world = pose_world[:,3:]
    R_world = [nq.quat2mat(quat) for quat in quat_world]

    # 生成相机坐标系下的平移和旋转
    t_camera = np.array([(np.dot(R_c2w, t) + t_c2w).reshape(3) \
                      for t in t_world])
    R_camera = np.array([np.dot(R_c2w, R.reshape(3,3)).reshape(9) \
                      for R in R_world])
    pose_camera = np.concatenate((t_camera, R_camera),axis=-1)
    return pose_camera

if __name__ == "__main__":
    for cycle_id in CYCLE_idx_list:
        for scene_id in SCENE_idx_list:
            # 构建当前循环和场景的csv路径
            csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))              
            name_temp, index, pose_world = read_csv(csv_path)
            
            pose_camera = generate_gt(pose_world)
            headers = ["class_name","id","x", "y", "z", "R1", "R2", "R3", "R4","R5", "R6", "R7", "R8","R9"]
            temp = np.concatenate((name_temp.reshape(-1,1), index.reshape(-1,1)),axis=-1)
            temp = np.concatenate((temp, pose_camera),axis=-1)
            # result = np.concatenate((temp, fg_prob.reshape(-1,1)),axis=-1).tolist()
            result = temp.tolist()
            
            assert len(result[0]) == len(headers)
            # 构建保存路径
            save_path = os.path.join(GT_PATH,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_loc = save_path + '/' + '{:0>3}'.format(scene_id) + '.csv'
            
            # 写入csv文件
            with open(file_loc, 'w', newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerows(result)
    print(f'第 {cycle_id} 个循环已完成')
