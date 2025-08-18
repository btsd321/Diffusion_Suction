# 读取输出可视化点云和向量
# 已知点云文件中点云存储为按行存储，第一行为表头(x y z nx ny nz)，第二行开始为点云数据
import os
import platform
import numpy as np
import matplotlib
import json
import argparse

# 检测操作系统并配置适当的后端和字体
def configure_matplotlib_for_os():
    """根据操作系统配置matplotlib"""
    system = platform.system().lower()
    
    print(f"检测到操作系统: {platform.system()} {platform.release()}")
    
    if system == 'windows':
        # Windows环境配置
        try:
            matplotlib.use('TkAgg')
            print("Windows: 使用TkAgg后端")
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
                print("Windows: 使用Qt5Agg后端")
            except ImportError:
                matplotlib.use('Agg')
                print("Windows: 使用Agg后端（无GUI）")
                
    elif system == 'linux':
        # Linux环境配置
        # 检测是否在WSL环境中
        is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
        if is_wsl:
            print("检测到WSL环境")
            matplotlib.use('Agg')  # WSL通常使用无GUI后端
            print("WSL: 使用Agg后端（无GUI）")
        else:
            # 标准Linux环境
            if os.environ.get('DISPLAY'):
                try:
                    matplotlib.use('Qt5Agg')
                    print("Linux: 使用Qt5Agg后端")
                except ImportError:
                    try:
                        matplotlib.use('TkAgg')
                        print("Linux: 使用TkAgg后端")
                    except ImportError:
                        matplotlib.use('Agg')
                        print("Linux: 使用Agg后端（无GUI）")
            else:
                matplotlib.use('Agg')
                print("Linux: 无DISPLAY环境变量，使用Agg后端（无GUI）")
                
    elif system == 'darwin':  # macOS
        try:
            matplotlib.use('MacOSX')
            print("macOS: 使用MacOSX后端")
        except ImportError:
            try:
                matplotlib.use('TkAgg')
                print("macOS: 使用TkAgg后端")
            except ImportError:
                matplotlib.use('Agg')
                print("macOS: 使用Agg后端（无GUI）")
    else:
        # 其他系统
        matplotlib.use('Agg')
        print(f"未知系统 {system}: 使用Agg后端（无GUI）")

# 配置字体支持
def configure_fonts_for_os():
    """根据操作系统配置字体"""
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows中文字体配置
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        plt.rcParams['font.sans-serif'] = fonts + ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        print("Windows: 配置中文字体 Microsoft YaHei, SimHei 等")
        
    elif system == 'linux':
        # Linux中文字体配置
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei']
        plt.rcParams['font.sans-serif'] = fonts + ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        print("Linux: 配置中文字体 WenQuanYi, Noto Sans CJK SC 等")
        
    elif system == 'darwin':  # macOS
        # macOS中文字体配置
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
        plt.rcParams['font.sans-serif'] = fonts + ['Helvetica', 'sans-serif']
        print("macOS: 配置中文字体 PingFang SC, Heiti SC 等")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 验证字体可用性
    try:
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        found_fonts = []
        for font in plt.rcParams['font.sans-serif'][:5]:  # 检查前5个字体
            if any(font in af for af in available_fonts):
                found_fonts.append(font)
        
        if found_fonts:
            print(f"找到可用中文字体: {', '.join(found_fonts)}")
        else:
            print("⚠️  未找到合适的中文字体，可能存在中文显示问题")
            
    except ImportError:
        print("⚠️  无法检测字体可用性")

# 执行配置
configure_matplotlib_for_os()

# 导入matplotlib.pyplot（必须在配置后端之后）
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置字体
configure_fonts_for_os()

def get_environment_info():
    """获取当前环境信息"""
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'backend': matplotlib.get_backend(),
        'display': os.environ.get('DISPLAY', 'None'),
        'wsl': "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", ""),
        'gui_available': matplotlib.get_backend() != 'Agg'
    }
    return info

def print_environment_info():
    """打印环境信息"""
    info = get_environment_info()
    print("=" * 50)
    print("环境配置信息:")
    print("=" * 50)
    print(f"操作系统: {info['system']} {info['release']}")
    print(f"Matplotlib后端: {info['backend']}")
    print(f"GUI可用: {'是' if info['gui_available'] else '否'}")
    if info['system'] == 'Linux':
        print(f"WSL环境: {'是' if info['wsl'] else '否'}")
        print(f"DISPLAY变量: {info['display']}")
    print("=" * 50)

# 输入
'''
visualize_data = {
    "point_cloud": preprocessed_pc,
    "normals": flipped_normals,
    "composite_score": composite_score,
    "wrench_score": results['wrench_scores'],
    "feasibility_score": results['feasibility_scores'],
    "visibility_score": results['visibility_scores'],
    "best_points": best_points,
}
'''

# 要求：
'''
1. 完成一个可视化函数输入为上面的visualize_data字典
2. 该函数第二个输入为可视化的分数类型，可选composite_score、wrench_score、feasibility_score、visibility_score，默认为composite_score
3. 该函数第三个输入为是否显示best_points
4. 根据选择的分数对点云颜色进行着色，分数越高越红越低颜色越蓝，分数本身不需要归一化，但是显示出来的颜色需要归一化到0-1之间，添加颜色图例
5. 如果显示best_points，则在点云上绘制这些点，颜色为绿色同时显示best_points对应的向量
'''


def visualize_results(visualize_data, score_type='composite_score', show_best_points_num=1, 
                     best_point_size=8, save_path=None):
    """
    可视化推理结果（使用matplotlib）
    
    参数:
        visualize_data: 包含点云和评分信息的字典
        score_type: 可视化的分数类型，可选 'composite_score', 'wrench_score', 'feasibility_score', 'visibility_score'
        show_best_points: 是否显示最佳吸取点
        best_point_size: 最佳吸取点球体的大小（用于matplotlib中散点的size参数）
        save_path: 保存路径，如果为None则不保存
    """
    try:
        # 显示环境信息
        print_environment_info()
        
        # 提取数据
        point_cloud = visualize_data["point_cloud"]
        normals = visualize_data["normals"] 
        best_points = visualize_data.get("best_points", [])
        
        # 获取对应的评分数据
        if score_type == 'composite_score':
            scores = visualize_data["composite_score"]
            score_name = "综合评分"
        elif score_type == 'wrench_score':
            scores = visualize_data["wrench_score"]
            score_name = "扭矩评分"
        elif score_type == 'feasibility_score':
            scores = visualize_data["feasibility_score"]
            score_name = "可行性评分"
        elif score_type == 'visibility_score':
            scores = visualize_data["visibility_score"]
            score_name = "可见性评分"
        else:
            raise ValueError(f"不支持的评分类型: {score_type}")
        
        print(f"开始可视化 - {score_name}")
        print(f"点云数量: {len(point_cloud)}")
        print(f"评分范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 创建matplotlib 3D图形
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云，使用评分进行颜色映射
        scatter = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                           c=scores, cmap='coolwarm', s=4, alpha=0.8)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(f'{score_name} 范围: [{scores.min():.4f}, {scores.max():.4f}]', 
                      rotation=270, labelpad=15, fontsize=12, 
                      fontproperties='SimHei' if 'SimHei' in plt.rcParams['font.sans-serif'] else None)
        
        # 如果需要显示最佳吸取点
        if show_best_points_num > 0 and best_points:
            # 确定实际显示的点数
            show_points_num = min(show_best_points_num, len(best_points))
            print(f"显示前 {show_points_num} 个最佳吸取点（共 {len(best_points)} 个）")
            
            # 只取前show_points_num个最高分的点
            selected_points = best_points[:show_points_num]
            
            # 准备最佳点的数据
            best_positions = []
            best_normals = []
            
            for point_info in selected_points:
                position = np.array(point_info['position'])
                normal = np.array(point_info['normal'])
                best_positions.append(position)
                best_normals.append(normal)
            
            best_positions = np.array(best_positions)
            best_normals = np.array(best_normals)
            
            # 绘制最佳吸取点（绿色球体）
            # 将best_point_size转换为matplotlib的散点大小
            scatter_size = best_point_size  # 调整大小以适应matplotlib
            ax.scatter(best_positions[:, 0], best_positions[:, 1], best_positions[:, 2], 
                      c='green', s=scatter_size, alpha=1.0, edgecolors='darkgreen', linewidth=2,
                      label=f'最佳吸取点 ({show_points_num}个)')
            
            # 绘制法向量箭头
            arrow_length = 0.1  # 保持与原来的比例关系
            for i, (position, normal) in enumerate(zip(best_positions, best_normals)):
                end_point = position + normal * arrow_length
                ax.plot([position[0], end_point[0]], 
                       [position[1], end_point[1]], 
                       [position[2], end_point[2]], 
                       color='yellow', linewidth=3, alpha=0.8)
        
        # 计算点云的边界以设置合适的显示范围
        x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
        y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
        z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
        
        # 计算各轴的范围
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # 找到最大范围，作为参考
        max_range = max(x_range, y_range, z_range)
        
        # 计算各轴的中心点
        x_center = (x_max + x_min) * 0.5
        y_center = (y_max + y_min) * 0.5
        z_center = (z_max + z_min) * 0.5
        
        # 设置各轴的显示范围，保持等比例
        half_range = max_range * 0.6  # 稍微扩大一点显示范围
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)
        
        # 设置坐标轴等比例显示
        ax.set_box_aspect([1,1,1])  # 设置xyz轴的比例为1:1:1
        
        # 添加坐标轴箭头（基于实际数据范围）
        axis_length = max_range * 0.3
        ax.quiver(x_center, y_center, z_center, axis_length, 0, 0, 
                 color='red', arrow_length_ratio=0.1, linewidth=3, alpha=0.7)
        ax.quiver(x_center, y_center, z_center, 0, axis_length, 0, 
                 color='green', arrow_length_ratio=0.1, linewidth=3, alpha=0.7)
        ax.quiver(x_center, y_center, z_center, 0, 0, axis_length, 
                 color='blue', arrow_length_ratio=0.1, linewidth=3, alpha=0.7)
        
        # 添加坐标轴标签
        ax.text(x_center + axis_length * 1.1, y_center, z_center, 'X', 
               color='red', fontsize=14, fontweight='bold')
        ax.text(x_center, y_center + axis_length * 1.1, z_center, 'Y', 
               color='green', fontsize=14, fontweight='bold')
        ax.text(x_center, y_center, z_center + axis_length * 1.1, 'Z', 
               color='blue', fontsize=14, fontweight='bold')
        
        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'吸取点可视化 - {score_name}\n颜色编码: 蓝色(低分) → 红色(高分)', 
                    fontsize=14, pad=20)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例（如果显示最佳点）
        if show_best_points_num > 0 and best_points:
            ax.legend(loc='upper right', fontsize=10)
        
        # 显示操作说明（根据环境调整）
        backend = matplotlib.get_backend()
        print("\n启动3D可视化...")
        print(f"当前后端: {backend}")
        
        if backend == 'Agg':
            print("无GUI环境 - 图片将被保存到文件")
        else:
            print("交互式环境 - 操作说明:")
            print("  - 鼠标左键拖拽: 旋转视角")
            print("  - 鼠标右键拖拽: 平移视角") 
            print("  - 滚轮: 缩放")
            
        print(f"  - 颜色编码: 蓝色(低{score_name}) → 红色(高{score_name})")
        if show_best_points_num > 0 and best_points:
            print("  - 绿色点: 最佳吸取点")
            print("  - 黄色线: 吸取方向")
        
        plt.tight_layout()
        
        # 根据环境和后端选择显示方式
        backend = matplotlib.get_backend()
        system = platform.system().lower()
        
        # 保存可视化结果
        if save_path:
            print(f"保存可视化结果到: {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 智能显示策略
        if backend == 'Agg':
            # 无GUI后端，自动保存图片
            if not save_path:
                auto_save_path = f"suction_visualization_{score_type}.png"
                print(f"检测到无GUI环境，自动保存图片到: {auto_save_path}")
                plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
            print("无GUI环境，图片已保存。如需查看请打开保存的图片文件。")
        else:
            # 有GUI后端，正常显示
            try:
                plt.show()
            except Exception as e:
                print(f"显示窗口失败: {e}")
                # 降级到保存图片
                if not save_path:
                    fallback_path = f"suction_visualization_{score_type}_fallback.png"
                    print(f"降级保存图片到: {fallback_path}")
                    plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
            
    except Exception as e:
        print(f"可视化失败: {e}")
        raise


def create_colorbar(scores, score_name):
    """
    创建并显示颜色图例（已集成到主可视化函数中）
    
    参数:
        scores: 评分数组
        score_name: 评分名称
    """
    # 此函数已集成到主可视化函数中，保留此函数以保持兼容性
    print(f"颜色图例信息: {score_name} 范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print("颜色映射: 蓝色(低分) → 红色(高分)")
