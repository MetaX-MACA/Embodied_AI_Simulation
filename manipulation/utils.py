import numpy as np
import importlib.util

# panda机械臂默认姿态
_PANDA_HOME = np.asarray((0, 0, 0, -1.5, 0, 1.6, 1.57))
PANDA_JOINT_NUM = len(_PANDA_HOME)

def quaternion_to_euler(q):
    """
    四元数转角度
    """
    w, x, y, z = q
    # 计算俯仰角（绕X轴旋转）
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    # 计算偏航角（绕Y轴旋转）
    pitch = np.arcsin(2.0 * (w * y - z * x))
    # 计算滚转角（绕Z轴旋转）
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    # 返回欧拉角（单位是弧度）
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    return np.array([roll, pitch, yaw])

def euler_to_quaternion(q):
    """
    欧拉角转四元数
    """
    roll_deg, pitch_deg, yaw_deg = q
    
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    
    # 计算四元数
    w = cy * cr * cp + sy * sr * sp
    x = sy * cr * cp - cy * sr * sp
    y = cy * sr * cp + sy * cr * sp
    z = cy * cr * sp - sy * sr * cp
    
    return np.array([w, x, y, z])

def direction_vector_to_quaternion(v, v0 = np.array([0, 0, 1])):
    """
    方向向量转四元数

    v: 目标方向
    v0 设定参考方向, 默认为z轴, 即初始方向
    """
    # 归一化输入向量 已在外面做了
    # v = v / np.linalg.norm(v)
    
    # 计算旋转轴
    u = np.cross(v0, v)
    u_norm = np.linalg.norm(u)
    
    # 如果旋转轴长度为0，说明方向相同或相反
    if u_norm < 1e-6:
        # 方向相同
        if np.dot(v0, v) > 0:
            return np.array([1, 0, 0, 0])  # 单位四元数
        # 方向相反
        else:
            return np.array([0, 1, 0, 0])  # 180度绕x轴旋转
    
    # 归一化旋转轴
    u = u / u_norm
    
    # 计算旋转角度
    theta = np.arccos(np.dot(v0, v))
    
    # 构造四元数
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)  # 实部 w
    q[1:] = u * np.sin(theta / 2)  # 虚部 x y z
    
    return q

def get_site_quaternion(target_point, start_point):
    direction_vector = target_point - start_point
    # 归一化向量，得到单位方向向量
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # 方向向量转四元数
    quat = direction_vector_to_quaternion(direction_vector)
    # print("direction_vector:", direction_vector)
    # print("quat:", quat)
    return quat, direction_vector

def import_function_from_file(file_path, func_name):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)