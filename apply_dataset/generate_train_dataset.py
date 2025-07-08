"""

@author: Huang Dingtao
@checked: Huang Dingtao



"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import torch
# test = torch.randn(30, 3).cuda()

import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

sys.path.append(FILE_DIR)
DATASET_apply_dataset_parameter = os.path.join(FILE_DIR_generate_dataset, 'parameter.json')
    


from H5DataGenerator import *
OUT_ROOT_DIR =  os.path.join(FILE_DIR, 'h5_dataset')
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)    

TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')
if not os.path.exists( TRAIN_SET_DIR ):
    os.mkdir(TRAIN_SET_DIR)


GT_DIR = os.path.join(FILE_DIR, 'gt')
SEGMENT_DIR = os.path.join(FILE_DIR, 'segment_images')
DEPTH_DIR = os.path.join(FILE_DIR, 'depth_images')
OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')
GT_PATH =  os.path.join(FILE_DIR, 'gt')
INDIVIDUA_PATH =  os.path.join(FILE_DIR, 'individual_object_size')


if __name__ == "__main__":
    CYCLE_idx_list = range(81, 100)
    SCENE_idx_list = range(1, 51)

    # --------------------------------------------------------------------------check estimate_normals
    CYCLE_idx_list = range(0, 1)
    SCENE_idx_list = range(1, 51)
    # --------------------------------------------------------------------------check estimate_normals


    g = H5DataGenerator(DATASET_apply_dataset_parameter)
    for cycle_id in CYCLE_idx_list:
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
      
        for scene_id in SCENE_idx_list:
            # load depth images
                                
            depth_image_path = os.path.join(DEPTH_DIR,'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id),'Image0001.png')
            depth_image = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
        
            # load segment images
            seg_img_path = os.path.join(SEGMENT_DIR,'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id),'Image0001.exr')
            segment_image = cv2.imread(seg_img_path,cv2.IMREAD_UNCHANGED)

            # define gt path 
            gt_file_path = os.path.join(GT_PATH,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))+'/' + '{:0>3}'.format(scene_id) + '.csv'

            # define output path
            output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id)    )

            # 加载稠密点云
            # dense_point_path =  os.path.join(OBJ_PATH + '_{}'.format(obj_id) '_{}'.format(obj_id)+".ply")

            # 加载单个物体的抓取分数用来knn得到seal分数
            # seal_path = os.path.join(FILE_DIR, 'OBJ' + '_{}'.format(obj_id),   '_{}'.format(obj_id) + ".npz")

            # 加载像素
            individual_object_size_path =  os.path.join(INDIVIDUA_PATH, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))+'/' + '{:0>3}'.format(scene_id) + '.csv'

            g.process_train_set( depth_image, segment_image, gt_file_path, output_h5_path,individual_object_size_path)

