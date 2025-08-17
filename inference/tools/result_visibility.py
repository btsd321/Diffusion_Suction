# 读取输出可视化点云和向量
# 已知点云文件中点云存储为按行存储，第一行为表头(x y z nx ny nz)，第二行开始为点云数据
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json
import argparse

def read_point_cloud(points_txt_path):
    """读取点云文件，跳过表头行"""
    points = []
    with open(points_txt_path, 'r') as f:
        # 跳过第一行表头
        next(f)
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释行
                parts = line.split()
                if len(parts) >= 6:  # 确保至少有6个值
                    x, y, z, nx, ny, nz = parts[:6]
                    try:
                        points.append([float(x), float(y), float(z), float(nx), float(ny), float(nz)])
                    except ValueError:  # 忽略无法转换的行
                        print(f"warn: 无法转换的行：{line}")
                        continue
                else:  # 忽略长度不足6的行
                    print(f"warn: 长度不足6的行：{line}")
                    continue
    points = np.array(points)
    return points

def read_result_file(result_file_path):
    """读取推理结果JSON文件"""
    if not os.path.exists(result_file_path):
        raise FileNotFoundError(f"结果文件不存在: {result_file_path}")
    
    # 检查文件扩展名
    file_ext = os.path.splitext(result_file_path)[1].lower()
    
    if file_ext != '.json':
        raise ValueError(f"只支持JSON格式文件，当前文件格式: {file_ext}")
    
    return read_best_points_json(result_file_path)


def read_best_points_json(json_file_path):
    """
    从JSON文件中提取最佳吸取点信息
    
    参数:
        json_file_path: JSON文件路径
        
    返回:
        dict: 包含吸取点信息的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            best_points = json.load(f)
        
        print(f"成功读取 {len(best_points)} 个最佳吸取点")
        
        # 提取关键信息
        result_info = {
            'total_points': len(best_points),
            'positions': [],          # 位置坐标列表
            'normals': [],           # 法向量列表
            'composite_scores': [],   # 综合评分列表
            'wrench_scores': [],     # 扭矩评分列表
            'feasibility_scores': [], # 可行性评分列表
            'visibility_scores': [],  # 可见性评分列表
            'normal_flipped': [],    # 法向量是否翻转
            'ranks': [],             # 排名列表
            'best_points_raw': best_points  # 原始数据
        }
        
        # 解析每个吸取点的信息
        for point in best_points:
            result_info['positions'].append(point['position'])
            result_info['normals'].append(point['normal'])
            result_info['composite_scores'].append(point['composite_score'])
            result_info['ranks'].append(point['rank'])
            
            # 提取原始评分
            original_scores = point['original_scores']
            result_info['wrench_scores'].append(original_scores['wrench_score'])
            result_info['feasibility_scores'].append(original_scores['feasibility_score'])
            result_info['visibility_scores'].append(original_scores['visibility_score'])
            result_info['normal_flipped'].append(original_scores['normal_flipped'])
        
        # 转换为numpy数组方便后续处理
        result_info['positions'] = np.array(result_info['positions'])
        result_info['normals'] = np.array(result_info['normals'])
        result_info['composite_scores'] = np.array(result_info['composite_scores'])
        result_info['wrench_scores'] = np.array(result_info['wrench_scores'])
        result_info['feasibility_scores'] = np.array(result_info['feasibility_scores'])
        result_info['visibility_scores'] = np.array(result_info['visibility_scores'])
        result_info['normal_flipped'] = np.array(result_info['normal_flipped'])
        result_info['ranks'] = np.array(result_info['ranks'])
        
        # 打印统计信息
        print(f"\n吸取点统计信息:")
        print(f"  总点数: {result_info['total_points']}")
        print(f"  最高综合评分: {result_info['composite_scores'].max():.4f}")
        print(f"  最低综合评分: {result_info['composite_scores'].min():.4f}")
        print(f"  平均综合评分: {result_info['composite_scores'].mean():.4f}")
        print(f"  扭矩评分范围: [{result_info['wrench_scores'].min():.3f}, {result_info['wrench_scores'].max():.3f}]")
        print(f"  可行性评分范围: [{result_info['feasibility_scores'].min():.3f}, {result_info['feasibility_scores'].max():.3f}]")
        print(f"  可见性评分范围: [{result_info['visibility_scores'].min():.3f}, {result_info['visibility_scores'].max():.3f}]")
        print(f"  法向量翻转数量: {result_info['normal_flipped'].sum()}/{result_info['total_points']}")
        
        return result_info
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except KeyError as e:
        raise ValueError(f"JSON文件缺少必要字段: {e}")
    except Exception as e:
        raise RuntimeError(f"读取JSON文件失败: {e}")


def get_top_k_points(result_info, k=10):
    """
    获取前k个最佳吸取点
    
    参数:
        result_info: 从read_best_points_json返回的结果信息
        k: 返回前k个点
        
    返回:
        dict: 前k个点的信息
    """
    if 'positions' not in result_info:
        raise ValueError("结果信息中缺少位置数据")
    
    k = min(k, result_info['total_points'])
    
    top_k_info = {
        'positions': result_info['positions'][:k],
        'normals': result_info['normals'][:k],
        'composite_scores': result_info['composite_scores'][:k],
        'wrench_scores': result_info['wrench_scores'][:k],
        'feasibility_scores': result_info['feasibility_scores'][:k],
        'visibility_scores': result_info['visibility_scores'][:k],
        'ranks': result_info['ranks'][:k]
    }
    
    print(f"提取前 {k} 个最佳吸取点")
    return top_k_info


def print_point_details(result_info, point_indices=None):
    """
    打印指定吸取点的详细信息
    
    参数:
        result_info: 结果信息字典
        point_indices: 要打印的点的索引列表，如果为None则打印前5个
    """
    if point_indices is None:
        point_indices = list(range(min(5, result_info['total_points'])))
    
    print(f"\n详细吸取点信息:")
    print("-" * 80)
    
    for i in point_indices:
        if i >= result_info['total_points']:
            continue
            
        print(f"第 {result_info['ranks'][i]} 名:")
        print(f"  位置: [{result_info['positions'][i][0]:.4f}, {result_info['positions'][i][1]:.4f}, {result_info['positions'][i][2]:.4f}]")
        print(f"  法向量: [{result_info['normals'][i][0]:.4f}, {result_info['normals'][i][1]:.4f}, {result_info['normals'][i][2]:.4f}]")
        print(f"  综合评分: {result_info['composite_scores'][i]:.4f}")
        print(f"  扭矩评分: {result_info['wrench_scores'][i]:.4f}")
        print(f"  可行性评分: {result_info['feasibility_scores'][i]:.4f}")
        print(f"  可见性评分: {result_info['visibility_scores'][i]:.4f}")
        print(f"  法向量翻转: {'是' if result_info['normal_flipped'][i] else '否'}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description='可视化推理结果')
    parser.add_argument('--points_txt_path', type=str, required=True, help='点云文件路径')
    parser.add_argument('--result_file_path', type=str, required=True, help='结果JSON文件路径')
    parser.add_argument('--top_k', type=int, default=10, help='显示前k个最佳吸取点')
    parser.add_argument('--show_details', action='store_true', help='显示详细信息')
    parser.add_argument('--save_visualization', type=str, help='保存可视化结果到指定路径')
    
    args = parser.parse_args()
    
    try:
        # 读取点云数据
        print("正在读取点云数据...")
        point_cloud_data = read_point_cloud(args.points_txt_path)
        print(f"成功读取点云: {point_cloud_data.shape}")
        
        # 读取推理结果（只支持JSON格式）
        print("正在读取推理结果...")
        result_info = read_result_file(args.result_file_path)
        
        # 获取前k个最佳点
        top_k_points = get_top_k_points(result_info, args.top_k)
        
        # 显示详细信息
        if args.show_details:
            print_point_details(result_info, list(range(args.top_k)))
        
        # 可视化结果
        print(f"\n开始可视化前 {args.top_k} 个最佳吸取点...")
        visualize_suction_points(point_cloud_data, top_k_points, args.save_visualization)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


def visualize_suction_points(point_cloud_data, suction_points, save_path=None):
    """
    使用Open3D可视化点云和吸取点
    
    参数:
        point_cloud_data: 原始点云数据 (N, 6) - [x, y, z, nx, ny, nz]
        suction_points: 吸取点信息字典
        save_path: 保存路径，如果为None则不保存
    """
    try:
        # 创建原始点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        
        # 设置点云颜色（灰色）
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 创建吸取点几何体列表
        geometries = [pcd]
        
        # 为每个吸取点创建球体和箭头
        for i in range(len(suction_points['positions'])):
            position = suction_points['positions'][i]
            normal = suction_points['normals'][i]
            score = suction_points['composite_scores'][i]
            
            # 创建球体表示吸取点
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(position)
            
            # 根据评分设置颜色（红色-低分，绿色-高分）
            color_intensity = min(1.0, score * 4)  # 假设最高分是0.25左右
            sphere.paint_uniform_color([1-color_intensity, color_intensity, 0])
            
            # 创建箭头表示法向量
            arrow_length = 0.02
            arrow_end = position + normal * arrow_length
            
            # 创建箭头几何体
            arrow = create_arrow(position, arrow_end, radius=0.001)
            arrow.paint_uniform_color([0, 0, 1])  # 蓝色箭头
            
            geometries.extend([sphere, arrow])
        
        # 可视化
        print("启动3D可视化窗口...")
        print("操作说明:")
        print("  - 鼠标左键拖拽: 旋转视角")
        print("  - 鼠标右键拖拽: 平移视角") 
        print("  - 滚轮: 缩放")
        print("  - Q键: 退出")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="吸取点可视化",
            width=1024,
            height=768
        )
        
        # 保存可视化结果
        if save_path:
            print(f"保存可视化结果到: {save_path}")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            for geom in geometries:
                vis.add_geometry(geom)
            vis.capture_screen_image(save_path)
            vis.destroy_window()
            
    except Exception as e:
        print(f"可视化失败: {e}")


def create_arrow(start_point, end_point, radius=0.001):
    """创建箭头几何体"""
    direction = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(direction)
    
    if length == 0:
        return o3d.geometry.TriangleMesh()
    
    # 创建圆柱体作为箭头主体
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length*0.8)
    
    # 创建圆锥作为箭头头部
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius*2, height=length*0.2)
    cone.translate([0, 0, length*0.8])
    
    # 合并圆柱体和圆锥
    arrow = cylinder + cone
    
    # 计算旋转矩阵
    direction = direction / length
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(direction, z_axis):
        rotation_matrix = np.eye(3)
    elif np.allclose(direction, -z_axis):
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1-c)/(s*s))
    
    # 应用旋转和平移
    arrow.rotate(rotation_matrix, center=[0, 0, 0])
    arrow.translate(start_point)
    
    return arrow


if __name__ == '__main__':
    exit(main())