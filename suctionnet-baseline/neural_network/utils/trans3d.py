from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
import numpy as np

def get_pose(pose):
    # 将4x4位姿矩阵转换为(x, y, z, alpha, beta, gamma)格式, 角度单位为度
    pos, quat = pose_4x4_to_pos_quat(pose)
    euler = np.array([quat2euler(quat)[0], quat2euler(quat)[1], quat2euler(quat)[2]])
    euler = euler * 180.0 / np.pi  # 弧度转角度
    alpha, beta, gamma = euler[0], euler[1], euler[2]
    x, y, z = pos[0], pos[1], pos[2]
    return x, y, z, alpha, beta, gamma

def get_mat(x, y, z, alpha, beta, gamma):
    """
    根据输入的位置和欧拉角, 生成4x4位姿矩阵

    参数说明:
        x, y, z: 平移分量
        alpha, beta, gamma: 欧拉角(单位：度)
    返回:
        pose: 4x4位姿矩阵
    """
    try:
        euler = np.array([alpha, beta, gamma]) / 180.0 * np.pi  # 角度转弧度
        quat = np.array(euler2quat(euler[0], euler[1], euler[2]))  # 欧拉角转四元数
        pose = pos_quat_to_pose_4x4(np.array([x, y, z]), quat)
        return pose
    except Exception as e:
        print(str(e))
        pass         

def pos_quat_to_pose_4x4(pos, quat):
    """
    将位置和四元数转换为4x4位姿矩阵

    参数说明:
        pos: 长度为3的位置向量
        quat: 长度为4的四元数
    返回:
        pose: 4x4的numpy数组, 表示位姿矩阵
    """
    pose = np.zeros([4, 4])
    mat = quat2mat(quat)  # 四元数转旋转矩阵
    pose[0:3, 0:3] = mat[:, :]
    pose[0:3, -1] = pos[:]
    pose[-1, -1] = 1
    return pose

def pose_4x4_to_pos_quat(pose):
    """
    将4x4位姿矩阵分解为位置和四元数

    参数说明:
        pose: 4x4的numpy数组, 表示位姿矩阵
    返回:
        pos: 长度为3的位置向量
        quat: 长度为4的四元数
    """
    mat = pose[:3, :3]  # 提取旋转部分
    quat = mat2quat(mat)  # 旋转矩阵转四元数
    pos = np.zeros([3])
    pos[0] = pose[0, 3]
    pos[1] = pose[1, 3]
    pos[2] = pose[2, 3]
    return pos, quat