# -*- coding:utf-8 -*-
"""
@author: Huang Dingtao
@checked: Huang Dingtao
    
"""

CYCLE_idx_list = range(0, 100)
SCENE_idx_list = range(1, 51)  #  (1, 61)   1 2 3 4 .....59 60



import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

# sys.path.append(FILE_DIR_generate_dataset)
OBJ_folder_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"OBJ"  )
OBJ_files_and_dirs = os.listdir(OBJ_folder_path)
OBJ_name = [str(i ) for i in range(0, 113)]
OBJ_name.remove("112")
OBJ_name.remove("110")

OBJ_name.remove("90")
OBJ_name.remove("91")
OBJ_name.remove("80")
OBJ_name.remove("81")
OBJ_name.remove("70")

OBJ_name.remove("10")
OBJ_name.remove("11")
OBJ_name.remove("20")
OBJ_name.remove("21")
OBJ_name.remove("30")
OBJ_name.remove("41")

print(OBJ_name)
print(len(OBJ_name))



OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')
OUTDIR_dir =  os.path.join(FILE_DIR, 'physics_result')
if not os.path.exists(OUTDIR_dir):
    os.makedirs(OUTDIR_dir)

from tkinter.tix import ButtonBox
import pybullet
import time
import math
import pybullet_data
import csv
import numpy as np
import yaml
import random



class GenerateSimulationResult:
    def __init__(self):
        
        # choose meshScales: confirm that the object unit is meters
        unit_of_obj = 'mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # mm -> m
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

        # 800 * 600 * 500  厚50
        self.box_width  = 0.8
        self.box_length = 0.6
        self.box_thickness = 0.05
        self.box_height =  0.50
        
        # self.show_GUI = 1
        self.show_GUI = 0
        self.random_range = [0.1, 0.1, 0.13, 0.15]
        self.box_bottom_thickness = 0.01
        self.random_range.append(self.box_bottom_thickness)

    


    def scene_init(self):

        if self.show_GUI:
            _ =pybullet.connect(pybullet.GUI) 
        else:
            _ = pybullet.connect(pybullet.DIRECT)
        pybullet.setPhysicsEngineParameter(numSolverIterations=10) 
        pybullet.setTimeStep(1. / 120.)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        _= pybullet.startStateLogging(pybullet.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
        pybullet.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0)
        pybullet.setGravity(0, 0, -5)
        # set camera
        pybullet.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0.01, 0.01, 0.01])


    def random_drop_objects_single(self, mesh_scale, nums):
        multi_body = []
        OBJ_name_sample_ids = []
        
        for _ in range(nums):
            vShapedId = []
            cShapedId = []
            OBJ_name_sample_id = random.choice(range(len(OBJ_name)))
            OBJ_name_sample_ids.append(OBJ_name[OBJ_name_sample_id])
            file_path = os.path.join(OBJ_PATH, OBJ_name[OBJ_name_sample_id] ,'object.obj')
            vShapedId = pybullet.createVisualShape(shapeType=pybullet.GEOM_MESH, fileName=file_path, meshScale=mesh_scale)
            cShapedId = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH, fileName=file_path, meshScale=mesh_scale)
        
            position = []
            position.append(np.random.uniform(-self.random_range[0], self.random_range[0]))
            position.append(np.random.uniform(-self.random_range[1], self.random_range[1]))
            position.append(np.random.uniform(self.random_range[2] +  self.box_bottom_thickness,self.random_range[3] + self.box_bottom_thickness))
            rand_euler_angle = np.random.uniform(-2.0 * math.pi, 2.0 * math.pi, [3])
            rand_quat = pybullet.getQuaternionFromEuler(rand_euler_angle)

            multi_body.append(pybullet.createMultiBody(
                baseMass=1,
                #baseInertialFramePosition=config["box"]["position"],
                baseCollisionShapeIndex=cShapedId,
                baseVisualShapeIndex=vShapedId,
                basePosition=position,
                baseOrientation=rand_quat,
                useMaximalCoordinates=False))
            pybullet.changeVisualShape(multi_body[-1], -1, rgbaColor=[1, 0, 0, 1])
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        for _ in range(160):
            #print(f'加载多个物体ing{i}')
            pybullet.stepSimulation()
            time.sleep(1. / 240)
        if self.show_GUI:
            for _ in range(1000):
                #print(f'加载多个物体ing{i}')
                pybullet.stepSimulation()
                time.sleep(1. / 240)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            
        return multi_body,OBJ_name_sample_ids


    def generate_single_object(self):    
        cycle_idx_list = CYCLE_idx_list
        scene_idx_list = SCENE_idx_list
        
        for cycle_id in cycle_idx_list:
            for scene_id in scene_idx_list:
                self.scene_init()

                cube_ind_1 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX') ,'cube1.urdf'), ( 0   , self.box_length*0.5+self.box_thickness*0.5,  self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1 )
                cube_ind_2 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX') ,'cube1.urdf'), ( 0   ,-self.box_length*0.5-self.box_thickness*0.5,  self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1 )
                cube_ind_3 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX') ,'cube2.urdf'), ( self.box_width*0.5+self.box_thickness*0.5,  0 ,  self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1 )
                cube_ind_4 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX') ,'cube2.urdf'), (-self.box_width*0.5-self.box_thickness*0.5, -0 ,  self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1 )
                
                # load part   check and if there are parts out of box, then re drop
                while (1):
                    flag = 0
                    multi_body_objects_first_layer , name_list = self.random_drop_objects_single(self.meshScale, scene_id)
                    for sparepart_id in multi_body_objects_first_layer:
                        final_position, angle = pybullet.getBasePositionAndOrientation(sparepart_id)
                        if((math.fabs(final_position[2])>= self.box_height)   or (final_position[2] < 0)):    
                            flag =1
                    if flag == 1:
                        for i in multi_body_objects_first_layer:
                            pybullet.removeBody(i)
                    if flag == 0:
                        break;
                foreground_nums = len(multi_body_objects_first_layer)
                index_list = [i for i in range(foreground_nums)] 
                assert foreground_nums == scene_id
                
                if self.show_GUI:
                    for _ in range(100):
                       
                        pybullet.stepSimulation()
                        time.sleep(1. / 240)
                    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
                    


                # save the simulation results
                self.save_results(cycle_id,scene_id,multi_body_objects_first_layer,index_list, name_list)
                pybullet.disconnect()

        print('The Simulation is finished!')       


    def save_results(self, cycle_id,scene_id, multi_body_list,index_list, name_list):

        headers = ["Type", "Index", "x", "y", "z", "w", "i", "j", "k"]
        rows = []
        for _ in range(720):
            pybullet.stepSimulation()
        for i, mb in enumerate(multi_body_list):
            final_position, quat = pybullet.getBasePositionAndOrientation(mb)
            row = [name_list[i], index_list[i], \
                        final_position[0], final_position[1], \
                        final_position[2] , \
                        quat[3], quat[0], quat[1], quat[2]]
            rows.append(row)
        
        out_cycle_dir = os.path.join(OUTDIR_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
        if not os.path.exists(out_cycle_dir):
            os.makedirs(out_cycle_dir)
        file_loc = out_cycle_dir + '/' + "{:0>3}.csv".format(scene_id)
        with open(file_loc, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)
        print(  f' {cycle_id} cycle: {scene_id} scene is completed')
       
    

if __name__ == '__main__':
    import time
    start_time = time.time()  

    physics_generator = GenerateSimulationResult()
    physics_generator.generate_single_object()

    end_time = time.time()
    print(end_time-start_time )