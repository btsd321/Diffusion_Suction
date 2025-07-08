import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)
OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')

import json
import math
import numpy as np
import cv2
import os
import time
import torch
import open3d as o3d
from torch_cluster import knn

# from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d
import h5py
import time
import csv
import random






def viewpoint_to_matrix_x(towards):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2
    return matrix

def viewpoint_to_matrix_z(towards):
    n = towards
    new_z = n
    new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
    new_y = new_y / np.linalg.norm(new_y)
    new_z = new_z / np.linalg.norm(new_z)
    new_x = np.cross(new_y, new_z)
    new_x = new_x / np.linalg.norm(new_x)
    new_x = np.expand_dims(new_x, axis=1)
    new_y = np.expand_dims(new_y, axis=1)
    new_z = np.expand_dims(new_z, axis=1)  
    rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
    return rot_matrix





class H5DataGenerator(object):
    def __init__(self, params_file_name, target_num_point = 16384):
        '''
        Input:
            params_file_name: path of parameter file ("parameter.json")
            target_num_point: target number of sampled points, default is 16384
        '''
        self.params = self._load_parameters(params_file_name)
        self.target_num_point = target_num_point

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm = False, xyz_limit=None):
        '''
        Input:
            us: np array of u coordinate
            vs: np array of v coordinate
            zs: np array of z coordinate
            to_mm: *1000.0 if True
            xyz_limit: None if no limit for xyz. Typical [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
        Output:
            x, y, z
        '''
        assert len(us) == len(vs) == len(zs)
        camera_info = self.params
        fx = camera_info['fu']
        fy = camera_info['fv']
        cx = camera_info['cu']
        cy = camera_info['cv']
        clip_start = camera_info['clip_start']
        clip_end = camera_info['clip_end']
        Zline = clip_start + (zs/camera_info['max_val_in_depth']) * (clip_end - clip_start)
        Zcs = Zline/np.sqrt(1+ np.power((us-cx)/fx,2) + np.power((vs-cy)/fy,2))
        if to_mm:
            Zcs *= 1000
        Xcs = (us - cx) * Zcs / fx
        Ycs = (vs - cy) * Zcs / fy
        Xcs = np.reshape(Xcs, (-1, 1))
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        
        
        points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)
       
        if xyz_limit is not None:
            if xyz_limit[0] is not None:
                xmin, xmax = xyz_limit[0]
                if xmin is not None:
                    idx = np.where( points[:, 0]>xmin )
                    points = points[idx]
                if xmax is not None:
                    idx = np.where( points[:, 0]<xmax )
                    points = points[idx]
            if xyz_limit[1] is not None:
                ymin, ymax = xyz_limit[1]
                if ymin is not None:
                    idx = np.where( points[:, 1]>ymin )
                    points = points[idx]
                if ymax is not None:
                    idx = np.where( points[:, 1]<ymax )
                    points = points[idx]
            if xyz_limit[2] is not None:
                zmin, zmax = xyz_limit[2]
                if zmin is not None:
                    idx = np.where( points[:, 2]>zmin )
                    points = points[idx]
                if zmax is not None:
                    idx = np.where( points[:, 2]<zmax )
                    points = points[idx]
                
        return points

    def _load_parameters(self, params_file_name):
        '''
        Input:
            params_file_name: path of parameter file ("parameter.json")
        '''
        params = {}
        with open(params_file_name,'r') as f:
            config = json.load(f)
            params = config
        return params 

    def _read_label_csv(self, file_name):
        '''
        Input:
            file_name: path of ground truth file name
        Output:
            label_trans: numpy array of shape (num_obj+1)*3. 0th pos is bg
            label_rot: numpy array of shape (num_obj+1)*9. 0th pos is bg
            label_vs: numpy array of shape (num_obj+1,). 0th pos is bg
        '''
        with open(file_name,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        array_file = np.array(list_file)[1:] # exclude titles
        num_obj = int(array_file.shape[0])
        label_trans = array_file[:,2:5].astype('float32')
        label_rot = array_file[:, 5:14].astype('float32')
        label_id = array_file[:, 1:2].astype('float32')
        label_name = array_file[:, 0]
              
        return label_trans, label_rot,label_id,label_name


    def individual_label_csv(self, file_name):
        with open(file_name,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        array_file = np.array(list_file).astype('float32')   # exclude titles
        return array_file


    def create_mesh_cylinder(self,radius, height, R, t, collision):
        cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
        # vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
        vertices = np.asarray(cylinder.vertices)
        # vertices[:, 0] += height / 2
        vertices[:, 2] += height / 2
        
        vertices = np.dot(R, vertices.T).T + t
        cylinder.vertices = o3d.utility.Vector3dVector(vertices)
        # if collision:
        #     colors = np.array([0.7, 0, 0])
        # else:
        #     colors = np.array([0, 0.7, 0])
        

        ball_colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        ball_colors[:, 0] = collision

        # colors = np.expand_dims(colors, axis=0)
        # colors = np.repeat(colors, vertices.shape[0], axis=0)
        cylinder.vertex_colors = o3d.utility.Vector3dVector(ball_colors)

        return cylinder



    def process_train_set(self,depth_img, segment_img, gt_file_path, output_file_path,individual_object_size_path,xyz_limit=None):
        start_time = time.time()    # 程序开始时间

        assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
        label_trans, label_rot, label_id,label_name = self._read_label_csv(gt_file_path)
        obj_num = label_trans.shape[0]
       
        xs = np.where(depth_img != 0)[1]
        ys = np.where(depth_img != 0)[0]
        zs = depth_img[depth_img != 0]
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)
        obj_ids = np.round(segment_img[:,:, 1][segment_img[:,:, 2] == 1] * (obj_num - 1)).astype('int')


       
        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('No foreground points!!!!!')
            return
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])
            points = points_tile[:self.target_num_point]
            obj_ids_tile = np.tile(obj_ids, [t])
            obj_ids = obj_ids_tile[:self.target_num_point]
        if num_pnt > self.target_num_point:
            points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float()
            points_transpose = points_transpose.cuda()
            sampled_idx =  furthest_point_sample(points_transpose, self.target_num_point).cpu().numpy().reshape(self.target_num_point)
            points = points[sampled_idx]
            obj_ids = obj_ids[sampled_idx]
 


        individual_object_size_lable = self.individual_label_csv(individual_object_size_path)[0]
      

        label_trans = label_trans[obj_ids]
        label_rot = label_rot[obj_ids]
        label_id = label_id[obj_ids]
        individual_object_size_lable = individual_object_size_lable[obj_ids]



        # pcd_vis = o3d.geometry.PointCloud()  
        # pcd_vis.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd_vis], window_name="3D Point Cloud", width=800, height=600)

  
        # for cluster_idx in np.unique(obj_ids):
        #     points_single_show =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[obj_ids == cluster_idx]))
        #     o3d.visualization.draw_geometries( [ points_single_show ], width=800,   height=600)
            
  
        # # vs_picked_idx = individual_object_size_lable > 0.4
        # # points_kejiandu = points[vs_picked_idx]
        # # show_points([points_kejiandu], [[2555, 0, 0]], radius=3)
        # # # 

 
        # all_model_point = []
        # all_model_color = []
        # for index in range(len(label_name)):

        #     object_sparse_point = np.load(os.path.join(OBJ_PATH,  label_name[index], "object_sparse_point.npz"))
        #     points_m_keshihua = object_sparse_point['points']
        #     if label_rot[obj_ids == index].shape[0] == 0:
        #         continue
        #     label_rot_temp = label_rot[obj_ids == index][0].reshape(3, 3)
        #     label_trans_temp = label_trans[obj_ids == index][0].reshape(1, 3)
        #     label_trans_temp = np.tile(label_trans_temp, [points_m_keshihua.shape[0], 1])
        #     points_m_keshihua = np.dot(   points_m_keshihua, label_rot_temp.T) + label_trans_temp
        
        #     pcd_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        #     points_m_keshihua = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_m_keshihua))
        #     o3d.visualization.draw_geometries( [ pcd_vis,points_m_keshihua ], width=800,   height=600)
        #     o3d.visualization.draw_geometries( [ points_m_keshihua ], width=800,   height=600)
        #     o3d.visualization.draw_geometries([pcd_vis], width=800, height=600)



 
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        # radius = 0.01  # 搜索半径
        # max_nn = 30  # 邻域内用于估算法线的最大点数
        # pc_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
        pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))# 方向是对 -1代表堆叠场景向外
        pc_o3d.normalize_normals()
        # o3d.visualization.draw_geometries([pc_o3d], window_name="法线估计",point_show_normal=True,width=800, height=600)  # 窗口高度
 
        suction_points = points
        suction_or=np.array(pc_o3d.normals).astype(np.float32)



        # # --------------------------------------------------------------------------check estimate_normals
        # from pathlib import Path
        # # output_file_path_check = output_file_path.replace('h5_dataset', 'check_h5_dataset')
        # output_file_path_check = Path(output_file_path)
        # output_file_path_check= output_file_path_check.with_suffix('.ply')
        # o3d.io.write_point_cloud(str(output_file_path_check), pc_o3d)
        # # continue
        # 
        # # --------------------------------------------------------------------------check estimate_normals








        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        show_point_temp.colors = o3d.utility.Vector3dVector([[0, 0, 1]  for i in range(points.shape[0])])
        vis_list = [  show_point_temp   ]

        for idx in range(len(suction_points[0:100])):
            suction_point = suction_points[idx]
            anno_normal = suction_or[idx]
            suction_score = individual_object_size_lable[idx]
            n = anno_normal
            new_z = n
            new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
            new_y = new_y / np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)
            new_x = new_x / np.linalg.norm(new_x)
            new_x = np.expand_dims(new_x, axis=1)
            new_y = np.expand_dims(new_y, axis=1)
            new_z = np.expand_dims(new_z, axis=1)
            rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
            ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
            vis_list.append(ball)
        o3d.visualization.draw_geometries(vis_list, width=800, height=600)
 
        import matplotlib.pyplot as plt
        individual_object_size_lable_temp = individual_object_size_lable*100
        plt.hist(individual_object_size_lable_temp.astype(int), bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()






        suction_points_normalization = np.matmul((suction_points - label_trans).reshape(16384,1,3), label_rot.reshape(-1,3,3) )
        suction_points_normalization = suction_points_normalization.reshape(16384, 3)
        suction_seal_scores  = np.zeros((16384, ))
        for index in range(len(label_name)):
 
            annotation = np.load(os.path.join(OBJ_PATH, label_name[index], "labels.npz"))
            object_sparse_point = annotation['points']
            anno_points = annotation['points']
            anno_scores = annotation['scores']

            suction_points_normalization_id = suction_points_normalization[obj_ids == index]
            if suction_points_normalization_id.shape[0] == 0:
                continue
            suction_points_normalization_id_knn = torch.from_numpy(suction_points_normalization_id).float()
            suction_points_normalization_id_knn = suction_points_normalization_id_knn.cuda()
            
            anno_points_knn = torch.from_numpy(anno_points).float()
            anno_points_knn = anno_points_knn.cuda()
            
     
            indices, dist=knn(anno_points_knn, suction_points_normalization_id_knn, k=1)
            dist=dist.cpu().numpy().reshape(dist.shape[-1])
            suction_seal_scores[obj_ids == index] = anno_scores[dist]

   
            # vis_temp1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anno_points))
            # vis_temp2 =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(  suction_points_normalization[obj_ids == index]     ))
            # o3d.visualization.draw_geometries( [ vis_temp2 ,vis_temp1], width=800,   height=600)


            # vis_temp1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anno_points))
            # suction_points_temp = suction_points_normalization[obj_ids == index]
            # anno_scores_temp = anno_scores[dist]
            # show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(suction_points_temp))
            # colors_temp = [[0, 0, 1]  for i in range(suction_points_temp.shape[0])]
            # show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
            # vis_list = [show_point_temp,vis_temp1]
            
            # for idx in range(suction_points_temp.shape[0]):
            #     suction_point = suction_points_temp[idx]
            #     suction_score = anno_scores_temp[idx]
            #     ball = o3d.geometry.TriangleMesh.create_sphere(0.001).translate(suction_point)
            #     ball_v = np.asarray(ball.vertices)
            #     ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
            #     ball_colors[:, 0] = suction_score
            #     # ball_colors[:, 2] = suction_score
            #     ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
            #     vis_list.append(ball)
            # o3d.visualization.draw_geometries(vis_list, width=800, height=600)
    
     

        # suction_points_normalization_re = np.matmul(  suction_points_normalization.reshape(16384,1,3), np.transpose(label_rot.reshape(16384,3,3), (0, 2, 1)) ).reshape(16384,3)+label_trans
        # suction_points_normalization_re_show = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(suction_points_normalization_re))
        # o3d.visualization.draw_geometries([suction_points_normalization_re_show], window_name="法线估计",point_show_normal=True,width=800, height=600)  # 窗口高度
        
   
        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        vis_list = [  show_point_temp   ]

        for idx in range(len(suction_points[0:1024*4])):
            suction_point = suction_points[idx]
            suction_score = suction_seal_scores[idx]
            ball = o3d.geometry.TriangleMesh.create_sphere(0.001).translate(suction_point)
            ball_v = np.asarray(ball.vertices)
            ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
            ball_colors[:, 0] = suction_score
            # ball_colors[:, 2] = suction_score
            ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
            vis_list.append(ball)
        
   
        for idx in range(len(suction_points[0:100])):
            suction_point = suction_points[idx]
            anno_normal = suction_or[idx]
            suction_score = suction_seal_scores[idx]
            n = anno_normal
            new_z = n
            new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
            new_y = new_y / np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)
            new_x = new_x / np.linalg.norm(new_x)
            new_x = np.expand_dims(new_x, axis=1)
            new_y = np.expand_dims(new_y, axis=1)
            new_z = np.expand_dims(new_z, axis=1)
            rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
            ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
            vis_list.append(ball)
        o3d.visualization.draw_geometries(vis_list, width=800, height=600)


        import matplotlib.pyplot as plt
        plt.hist(suction_seal_scores, bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
  
        

        k = 15.6
        k = 30
        radius = 0.01
        wrench_thre = k * radius * np.pi * np.sqrt(2)
        suction_wrench_scores = []
        for index_temp,suction_points_temp in enumerate(suction_points):
            label_trans_temp = label_trans[index_temp]
            suction_or_temp = suction_or[index_temp]
            center = label_trans_temp# (3,)
            gravity = np.array([[0, 0, 1]], dtype=np.float32) * 9.8  # 1和-1都不影响的
            # gravity = np.array([[0, 0,  1]], dtype=np.float32) * 9.8  # (1, 3)
            suction_axis = viewpoint_to_matrix_z(suction_or_temp)  # (3, 3)
            # suction_axis = viewpoint_to_matrix_x(suction_or_temp)  # (3, 3)    ????????
            suction2center = (center - suction_points_temp)[np.newaxis, :]# (1, 3)
            coord = np.matmul(suction2center, suction_axis)# (1, 3)
            gravity_proj = np.matmul(gravity, suction_axis)# (1, 3)
            torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]# scalar
            torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]# scalar
            torque = np.sqrt(torque_x**2 + torque_y**2)# scalar
            score = 1 - min(1, torque / wrench_thre)# scalar
            suction_wrench_scores.append(score)
        suction_wrench_scores = np.array(suction_wrench_scores)


        # show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        # show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        # vis_list = [  show_point_temp   ]
        # # 
        # for idx in range(len(suction_points[0:1024])):
        #     suction_point = suction_points[idx]
        #     suction_score = suction_wrench_scores[idx]
        #     ball = o3d.geometry.TriangleMesh.create_sphere(0.001).translate(suction_point)
        #     ball_v = np.asarray(ball.vertices)
        #     ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
        #     ball_colors[:, 0] = suction_score
        #     # ball_colors[:, 2] = suction_score
        #     ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
        #     vis_list.append(ball)
        
        # # 
        # for idx in range(len(suction_points[0:200])):
        #     suction_point = suction_points[idx]
        #     anno_normal = suction_or[idx]
        #     suction_score = suction_wrench_scores[idx]
        #     n = anno_normal
        #     new_z = n
        #     new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
        #     new_y = new_y / np.linalg.norm(new_y)
        #     new_x = np.cross(new_y, new_z)
        #     new_x = new_x / np.linalg.norm(new_x)
        #     new_x = np.expand_dims(new_x, axis=1)
        #     new_y = np.expand_dims(new_y, axis=1)
        #     new_z = np.expand_dims(new_z, axis=1)  
        #     rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
        #     ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
        #     vis_list.append(ball)
        
        # o3d.visualization.draw_geometries(vis_list, width=800,   height=600)
      
        # import matplotlib.pyplot as plt
        # plt.hist(suction_wrench_scores, bins=100)
        # plt.title("Histogram of Data")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        # plt.show()




        # 感觉应该根据吸盘来定,suctionnet  height = 0.1     radius = 0.01
        height = 0.1
        radius = 0.01
        scence_point = points
        suction_feasibility_scores = []
        for index_temp,suction_points_temp in enumerate(suction_points):
            suction_or_temp = suction_or[index_temp]
            grasp_poses = viewpoint_to_matrix_x(suction_or_temp)
            target = scence_point-suction_points_temp
            target = np.matmul(target, grasp_poses)
            
    
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            # org_trans_workspace_target =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
            # o3d.visualization.draw_geometries( [ org_trans_workspace_target  ,frame  ], width=800,   height=600)
     
            target_yz = target[:, 1:3]
            target_r = np.linalg.norm(target_yz, axis=-1)
            mask1 = target_r < radius
            mask2 = ((target[:,0] > 0.005) & (target[:,0] < height))
            mask = np.any(mask1 & mask2)
            suction_feasibility_scores.append(mask)
        suction_feasibility_scores = ~np.array(suction_feasibility_scores)


        # show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        # show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        # vis_list = [  show_point_temp   ]
        # # 
        # for idx in range(len(suction_points[0:100])):
        #     suction_point = suction_points[idx]
        #     anno_normal = suction_or[idx]
        #     suction_score = suction_feasibility_scores[idx]
        #     n = anno_normal
        #     new_z = n
        #     new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
        #     new_y = new_y / np.linalg.norm(new_y)
        #     new_x = np.cross(new_y, new_z)
        #     new_x = new_x / np.linalg.norm(new_x)
        #     new_x = np.expand_dims(new_x, axis=1)
        #     new_y = np.expand_dims(new_y, axis=1)
        #     new_z = np.expand_dims(new_z, axis=1)
        #     rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
        #     ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
        #     vis_list.append(ball)
        # o3d.visualization.draw_geometries(vis_list, width=800,   height=600)
   
        # import matplotlib.pyplot as plt
        # plt.hist(suction_feasibility_scores.astype(int), bins=100)
        # plt.title("Histogram of Data")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        # plt.show()

 


        score_all = suction_seal_scores * suction_wrench_scores
        score_all = suction_seal_scores * suction_wrench_scores* suction_feasibility_scores*individual_object_size_lable

        sorted_indices = np.argsort(score_all)[::-1]
        score_all_asort = score_all[sorted_indices]
        points_asort = points[sorted_indices]
        suction_points_asort = suction_points[sorted_indices]
        suction_or_asort = suction_or[sorted_indices]

  

        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_asort))
        colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        vis_list = [  show_point_temp   ]
        # 
        for idx in range(len(suction_points_asort[0:600])):
            suction_point = suction_points_asort[idx]
            anno_normal = suction_or_asort[idx]
            suction_score = score_all_asort[idx]
            n = anno_normal
            new_z = n
            new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
            new_y = new_y / np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)
            new_x = new_x / np.linalg.norm(new_x)
            new_x = np.expand_dims(new_x, axis=1)
            new_y = np.expand_dims(new_y, axis=1)
            new_z = np.expand_dims(new_z, axis=1)
            rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
            ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
            vis_list.append(ball)
        o3d.visualization.draw_geometries(vis_list, width=800,   height=600)
 
        import matplotlib.pyplot as plt
        plt.hist(score_all, bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
 


        # ------------------------------------------------------------------------------------step 5: save as h5 file
        with h5py.File(output_file_path,'w') as f:
            f['points'] = points
            f['suction_or'] = suction_or
            f['suction_seal_scores'] = suction_seal_scores
            f['suction_wrench_scores'] = suction_wrench_scores
            f['suction_feasibility_scores'] = suction_feasibility_scores
            f['individual_object_size_lable'] = individual_object_size_lable
            
            








