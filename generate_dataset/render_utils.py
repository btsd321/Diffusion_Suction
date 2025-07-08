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
print(SCENE_idx_list )




import os

FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

# # w10 可视化时候需要多加一句
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
OUTDIR_dir_depth_images =  os.path.join(FILE_DIR, 'depth_images')
if not os.path.exists(OUTDIR_dir_depth_images):
    os.makedirs(OUTDIR_dir_depth_images)
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images')
if not os.path.exists(OUTDIR_dir_segment_images):
    os.makedirs(OUTDIR_dir_segment_images)
OUTDIR_dir_rgb_images =  os.path.join(FILE_DIR, 'rgb_images')
if not os.path.exists(OUTDIR_dir_rgb_images):
    os.makedirs(OUTDIR_dir_rgb_images)

class BlenderRenderClass:
    def __init__(self, ):
        # *****Blender Setup camera*****"
        resolution = [1920, 1200]
        focal_length = 16.0
        sensor_size =  [11.25, 7.03]
        cam_location_x_y_z = [0, 0, 1.70]
        # cam_rotation_qw_qx_qy_qz = [0, -0.70710678, -0.70710678, 0] # [90. ,  0. ,180.]
        cam_rotation_qw_qx_qy_qz = [0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,6.123234e-17]  # [0. ,  0. ,180.]
        
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


        # 清除现有的灯光对象
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='LAMP')  # 修正类型为 'LAMP'
        bpy.ops.object.delete()

        # 创建平行光（Sun Light）
        bpy.ops.object.lamp_add(type='SUN', location=(0, 0, 2))  # 添加平行光源并设置其位置
        sun_light = bpy.context.object  # 获取刚创建的光源对象
        sun_light.name = "Sun_Light"  # 给光源命名
        # 设置平行光的属性
        sun_light.data.energy = 10  # 设置光源强度
        sun_light.data.color = (1, 1, 1)  # 设置光源颜色为白色 (RGB)
        sun_light.data.use_nodes = True  # 启用节点系统（如果需要控制光源的其他属性）
        
        locations_z = 1.3
        locations = [[0,0,locations_z],[0,locations_z*0.5,locations_z],[locations_z*0.5,0,locations_z],[0,-locations_z*0.5,locations_z],[-locations_z*0.5,0,locations_z]]
        for i in range(5):
            # 创建点光源 (POINT light)
            bpy.ops.object.lamp_add(type='POINT', location=locations[i])
            point_light = bpy.context.object
            point_light.name = "Point_Light"
            point_light.data.energy = 10  # 设置光源强度
            
    def read_csv(self, csv_path):      
        with open(csv_path,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  #[ ["Type", "Index", "x", "y", "z", "w", "i", "j", "k",'Layer'] ,[], []  ]
            # list_file = [row for row in all_lines if any(row)]  #[ ["Type", "Index", "x", "y", "z", "w", "i", "j", "k",'Layer'] ,[], []  ]
            
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


    def grb_graph(self, rgb_scene_path):
        # edit the node compositing to use node
        bpy.data.scenes["Scene"].use_nodes = 1

        # define the compositing node
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links
        for node in nodes:
            nodes.remove(node)
        
        output_file_rgb = nodes.new("CompositorNodeOutputFile")
        output_file_rgb.base_path = rgb_scene_path
        output_file_rgb.format.file_format = "OPEN_EXR"
        output_file_rgb.format.color_mode = "RGB"
        output_file_rgb.format.color_depth = '32'
        render_layers = nodes.new("CompositorNodeRLayers")
        links.new(render_layers.outputs['Image'], output_file_rgb.inputs['Image'])


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
        divide = nodes.new("CompositorNodeMath")
        divide.operation = "DIVIDE"
        divide.inputs[1].default_value = self.DEPTH_DIVIDE
        less_than = nodes.new("CompositorNodeMath")
        less_than.operation = "LESS_THAN"
        less_than.inputs[1].default_value = self.DEPTH_LESS
        multiply = nodes.new("CompositorNodeMath")
        multiply.operation = "MULTIPLY"



        # one output is for depth, the other is for label
        output_file_depth = nodes.new("CompositorNodeOutputFile")
        output_file_depth.base_path = depth_path
        output_file_depth.format.file_format = "PNG"
        output_file_depth.format.color_mode = "BW"
        output_file_depth.format.color_depth = '16'

        output_file_label = nodes.new("CompositorNodeOutputFile")
        output_file_label.base_path = segment_path
        output_file_label.format.file_format = "OPEN_EXR"
        output_file_label.format.color_mode = "RGB"
        output_file_label.format.color_depth = '32'

        composite = nodes.new("CompositorNodeComposite")
        viewer = nodes.new("CompositorNodeViewer")

        links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        links.new(render_layers.outputs['Depth'], less_than.inputs[0])
        links.new(render_layers.outputs['Depth'], multiply.inputs[0])

        links.new(less_than.outputs[0], multiply.inputs[1])
        links.new(multiply.outputs[0], divide.inputs[0])

        links.new(divide.outputs[0], output_file_depth.inputs['Image'])
        links.new(divide.outputs[0], viewer.inputs['Image'])
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
        times = []     
        for cycle_id in CYCLE_idx_list:
            for scene_id in SCENE_idx_list:
                start_time = time.time()  # 记录起始时间戳
                self.camera_set()
                # get the name list and pose numpy array(x, y, z, qw, qx, qy, qz)
        
                csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
                obj_name, pose, segment_index = self.read_csv(csv_path)
                # print(obj_name)  # array(['Num1', 'Num1'],dtype='<U20') 
                # print(pose)  #[[ 0.14687355  0.18479148  0.02647368  0.70635259  0.68928874  0.158188750.0303875 ][ 0.0748543   0.21340428  0.02648409  0.70603818  0.2569916   0.65876436 -0.03866476]]

                self.import_obj(obj_name, pose, segment_index)

                depth_scene_path = os.path.join(OUTDIR_dir_depth_images,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                segment_scene_path = os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                rgb_scene_path = os.path.join(OUTDIR_dir_rgb_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                if not os.path.exists(depth_scene_path):
                    os.makedirs(depth_scene_path)
                if not os.path.exists(segment_scene_path):
                    os.makedirs(segment_scene_path)
                if not os.path.exists(rgb_scene_path):
                    os.makedirs(rgb_scene_path)

                self.grb_graph(rgb_scene_path)
                bpy.ops.render.render()
                self.depth_graph(depth_scene_path, segment_scene_path)
                self.label_graph(len(obj_name) - 1)
                bpy.ops.render.render()
                times.append(time.time()-start_time)
                
        np.save('times.npy', times)
        print(times)
if __name__ == '__main__':
    import time
    
    blender_generator = BlenderRenderClass()
    blender_generator.render_scenes()

