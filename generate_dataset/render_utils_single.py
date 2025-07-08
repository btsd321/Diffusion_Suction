# -*- coding:utf-8 -*-
"""

@author: Huang Dingtao
@checked: Huang Dingtao


"""
import sys
CYCLE_idx_list = [sys.argv[4]]
SCENE_idx_list = [sys.argv[5]]
print("CYCLE_idx_list")
print(CYCLE_idx_list )
print("SCENE_idx_list")
print(SCENE_idx_list)

import logging


logger = logging.getLogger("bpy")
logger.setLevel(logging.WARNING)  # 可以选择 WARNING 或 ERROR

import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

# w10 可视化时候需要多加一句
# FILE_DIR = os.path.dirname(FILE_DIR)



import bpy
import csv
import json

import numpy as np
import shutil
from math import radians
import math
# import yaml
# from easydict import EasyDict
import csv




OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')

OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')

OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images_sigle')
if not os.path.exists(OUTDIR_dir_segment_images):
    os.makedirs(OUTDIR_dir_segment_images)



class BlenderRenderClass:
    def __init__(self, ):

        # *****Blender Setup camera*****"
        resolution = [1920, 1200]
        focal_length = 16.0
        sensor_size =  [11.25, 7.03]
        cam_location_x_y_z = [0, 0, 1.70]
        cam_rotation_qw_qx_qy_qz =  [0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,6.123234e-17]
        depth_graph_divide =  2
        depth_graph_less = 3

        self.CAMERA_RESOLUTION = resolution
        self.CAMERA_FOCAL_LEN = focal_length
        self.CAMERA_SENSOR_SIZE = sensor_size
        self.CAMERA_LOCATION = cam_location_x_y_z
        self.CAMERA_ROTATION = cam_rotation_qw_qx_qy_qz
        self.DEPTH_DIVIDE = depth_graph_divide
        self.DEPTH_LESS = depth_graph_less
        # *****Blender Setup camera*****"

        unit_of_obj='mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # mm -> m
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

    def camera_set(self):
        # difine the engine kind
        bpy.data.scenes["Scene"].render.engine = "CYCLES"

        # define the camera Internal parameter
        bpy.data.scenes["Scene"].render.resolution_x = self.CAMERA_RESOLUTION[0]
        bpy.data.scenes["Scene"].render.resolution_y = self.CAMERA_RESOLUTION[1]

        bpy.data.scenes["Scene"].render.resolution_percentage = 100

        # write the camera focal legth and sensor size. unit is mm
        bpy.data.cameras["Camera"].type = "PERSP"
        bpy.data.cameras["Camera"].lens = self.CAMERA_FOCAL_LEN
        bpy.data.cameras["Camera"].lens_unit = "MILLIMETERS"
        bpy.data.cameras["Camera"].sensor_width = self.CAMERA_SENSOR_SIZE[0]
        bpy.data.cameras["Camera"].sensor_height = self.CAMERA_SENSOR_SIZE[1]
        
        # Fitting method of sensor internal image and field of view angle: horizontal – suitable for sensor width.
        bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"
        
        # Horizontal aspect ratio - used for deformed or non square pixel output
        bpy.data.scenes["Scene"].render.pixel_aspect_x = 1.0
   
        bpy.data.scenes["Scene"].render.pixel_aspect_y = self.CAMERA_SENSOR_SIZE[1] * self.CAMERA_RESOLUTION[0] / \
                                                         self.CAMERA_RESOLUTION[1] / self.CAMERA_SENSOR_SIZE[0]
        bpy.data.scenes["Scene"].cycles.progressive = "BRANCHED_PATH"
        bpy.data.scenes["Scene"].cycles.aa_samples = 1
        bpy.data.scenes["Scene"].cycles.preview_aa_samples = 1
       
        bpy.data.objects["Camera"].location = [self.CAMERA_LOCATION[0],
                                               self.CAMERA_LOCATION[1],
                                               self.CAMERA_LOCATION[2]]
        bpy.data.objects["Camera"].rotation_mode = 'QUATERNION'
        bpy.data.objects["Camera"].rotation_quaternion = [self.CAMERA_ROTATION[0],
                                                          self.CAMERA_ROTATION[1],
                                                          self.CAMERA_ROTATION[2],
                                                          self.CAMERA_ROTATION[3]]
        # let the camera coordinate rotate 180 degree around X axis
        bpy.data.objects["Camera"].rotation_mode = 'XYZ'
        bpy.data.objects["Camera"].rotation_euler[0] = bpy.data.objects["Camera"].rotation_euler[0] + math.pi


    def read_csv(self, csv_path):      
        with open(csv_path,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            # list_file = [i for i in all_lines]  #[ ["Type", "Index", "x", "y", "z", "w", "i", "j", "k",'Layer'] ,[], []  ]
            list_file = [row for row in all_lines if any(row)]  #[ ["Type", "Index", "x", "y", "z", "w", "i", "j", "k",'Layer'] ,[], []  ]
        array_file = np.array(list_file)[1:]
        obj_name = array_file[:,0]
        obj_index = array_file[:,1].astype('int')
        pose = array_file[:,2:9].astype('float32')
        return obj_name, pose, obj_index

    def import_obj(self, obj_name, pose, instance_index):
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.select = True
            else:
                o.select = False
        bpy.ops.object.delete()
        # delete the objects in scene at the beginning
    
        for instance_index_ in instance_index:
            file_path = os.path.join(OBJ_PATH, obj_name[instance_index_] ,'object.obj')
            bpy.ops.import_scene.obj(filepath=file_path)
            instance = bpy.context.selected_objects[0]
            print(bpy.context.selected_objects)
            print(instance_index_)
            instance.pass_index = instance_index_
            instance.scale = [0.001, 0.001, 0.001]
            instance.location = [pose[instance_index_][0], pose[instance_index_][1], pose[instance_index_][2]]
            instance.rotation_mode = 'QUATERNION'
            instance.rotation_quaternion = [pose[instance_index_][3], pose[instance_index_][4], pose[instance_index_][5], pose[instance_index_][6]]
            

    # Use the node to configure the depth map and split map 
    def depth_graph(self, depth_path, segment_path):
        # edit the node compositing to use node
        bpy.data.scenes["Scene"].use_nodes = 1

        # define the compositing node
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links
        for node in nodes:
            nodes.remove(node)

        render_layers = nodes.new("CompositorNodeRLayers")
        # divide = nodes.new("CompositorNodeMath")
        # divide.operation = "DIVIDE"
        # divide.inputs[1].default_value = self.DEPTH_DIVIDE
        # less_than = nodes.new("CompositorNodeMath")
        # less_than.operation = "LESS_THAN"
        # less_than.inputs[1].default_value = self.DEPTH_LESS
        # multiply = nodes.new("CompositorNodeMath")
        # multiply.operation = "MULTIPLY"



        # # one output is for depth, the other is for label
        # output_file_depth = nodes.new("CompositorNodeOutputFile")
        # output_file_depth.base_path = depth_path
        # output_file_depth.format.file_format = "PNG"
        # output_file_depth.format.color_mode = "BW"
        # output_file_depth.format.color_depth = '16'

        output_file_label = nodes.new("CompositorNodeOutputFile")
        output_file_label.base_path = segment_path
        output_file_label.format.file_format = "OPEN_EXR"
        output_file_label.format.color_mode = "RGB"
        output_file_label.format.color_depth = '32'

        # composite = nodes.new("CompositorNodeComposite")
        # viewer = nodes.new("CompositorNodeViewer")

        # links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        # links.new(render_layers.outputs['Depth'], less_than.inputs[0])
        # links.new(render_layers.outputs['Depth'], multiply.inputs[0])

        # links.new(less_than.outputs[0], multiply.inputs[1])
        # links.new(multiply.outputs[0], divide.inputs[0])

        # links.new(divide.outputs[0], output_file_depth.inputs['Image'])
        # links.new(divide.outputs[0], viewer.inputs['Image'])
        links.new(render_layers.outputs['Image'], output_file_label.inputs['Image'])




    # define the materials(such as color) of the object, and make the objects point at the same materials
    def label_graph(self, label_number):
       
        # 遍历场景中的所有物体
        for obj in bpy.context.scene.objects:
            # 只处理网格对象（你可以根据需要调整条件）
            if obj.type == 'MESH':
                # 确保物体有材质槽
                if obj.data.materials:
                    # 清空物体的所有材质槽
                    obj.data.materials.clear()
                    print("delet object  materials")
       
       
        mymat = bpy.data.materials.get('mymat')
        if not mymat:
            mymat = bpy.data.materials.new('mymat')
            mymat.use_nodes = True

        # delete the initial nodes
        nodes = mymat.node_tree.nodes
        links = mymat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        # change the color of ColorRamp
        ColorRamp = nodes.new(type="ShaderNodeValToRGB")
        ColorRamp.color_ramp.interpolation = 'LINEAR'
        ColorRamp.color_ramp.color_mode = 'RGB'

        ColorRamp.color_ramp.elements[0].color[:3] = [1.0, 0.0, 0.0]
        ColorRamp.color_ramp.elements[1].color[:3] = [1.0, 1.0, 0.0]

        # add the stop button according to the number of objeccts
        ObjectInfo = nodes.new(type="ShaderNodeObjectInfo")
        OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
        Emission = nodes.new(type="ShaderNodeEmission")

        Math = nodes.new(type="ShaderNodeMath")
        Math.operation = "DIVIDE"
        Math.inputs[1].default_value = label_number

        links.new(ObjectInfo.outputs[1], Math.inputs[0])
        links.new(Math.outputs[0], ColorRamp.inputs[0])
        links.new(ColorRamp.outputs[0], Emission.inputs[0])
        links.new(Emission.outputs[0], OutputMat.inputs[0])

        # let the obj document point at the same materials
        objects = bpy.data.objects
        count = 0
        for obj in objects:
            if obj.type == 'MESH':
                count+=1
                if not 'mymat' in obj.data.materials:
                    obj.data.materials.append(mymat)

    def render_scenes(self):        
        for cycle_id in CYCLE_idx_list:

            for scene_id in SCENE_idx_list:
                print( 'cycle_id={}'.format(cycle_id)+'scene_id={}'.format(scene_id))
                self.camera_set()
                
                csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
                obj_names, pose, segment_indexs = self.read_csv(csv_path)

                for i in segment_indexs:
                    obj_name = []
                    obj_name.append(obj_names[i])
                    self.import_obj(obj_names, pose, [i])

                    segment_scene_path = os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id),"{:0>3}".format(scene_id)+"_{:0>3}".format(i))
                    depth_scene_path = segment_scene_path# 没有用的
                    if not os.path.exists(depth_scene_path):
                        os.makedirs(depth_scene_path)
                    if not os.path.exists(segment_scene_path):
                        os.makedirs(segment_scene_path)
                    
                    self.depth_graph(depth_scene_path, segment_scene_path)
                    # rgb图快些
                    self.label_graph(len(obj_name) - 1)
                    bpy.ops.render.render()





if __name__ == '__main__':
    import time
    start_time = time.time()


    blender_generator = BlenderRenderClass()
    blender_generator.render_scenes()
    end_time = time.time()
    print(end_time-start_time )