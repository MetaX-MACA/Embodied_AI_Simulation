from copy import deepcopy

from manipulation.ompl_tool import OMPLTool, point_to_joints
from manipulation import approach_site_api
from manipulation.gpt_reward_api import get_eef_pos_and_quat
from manipulation.utils import _PANDA_HOME


def grasp_object_trajectories(env, ):
    # 当前姿态
    current_pos, current_quat = get_eef_pos_and_quat(env._data)
    current_pos = list(current_pos)
    current_quat = list(current_quat)
    current_pos.extend(current_quat)
    grasp_start_tra = current_pos

    # 重复步数，用于逐渐抓取目标
    steps = 50

    trajectories = []
    for _ in range(steps):
        trajectories.append(deepcopy(grasp_start_tra))

    return trajectories

def release_grasp_trajectories(env, ):
    # 当前姿态
    current_pos, current_quat = get_eef_pos_and_quat(env._data)
    current_pos = list(current_pos)
    current_quat = list(current_quat)
    current_pos.extend(current_quat)
    grasp_start_tra = current_pos

    # 重复步数，用于逐渐抓取目标
    steps = 30

    trajectories = []
    for _ in range(steps):
        trajectories.append(deepcopy(grasp_start_tra))

    return trajectories


def approach_object_trajectories(env, waypoint_func_name):
    # 机械臂的起始关节角度
    q_start = env._data.qpos[env._panda_dof_ids]
    joints_list = [q_start]

    waypoint_func = getattr(approach_site_api, waypoint_func_name)
    # 机械臂目标关节角度列表
    waypoints = waypoint_func(env)
    waypoint_num = len(waypoints)
    for item in waypoints:
        target_pos, target_quat = item[0], item[1]
        q_goal = point_to_joints(env, target_pos, target_quat)
        joints_list.append(q_goal)

    # 选择规划算法 支持: RRT, RRT-Connect, BIT*
    ompl_tool = OMPLTool(env, algorithm="BIT*")
    # 每段规划的路径包含多少个中间点
    trajectory_num = 200
    # 求解规划路径最大耗时 秒
    timeout = 10.0

    trajectories = []
    for i in range(len(joints_list)-1):
        tra_num = trajectory_num
        if waypoint_num > 1 and i == 0:
                tra_num = 100
        sub_trajectories = trajectory_plan(ompl_tool, joints_list[i], joints_list[i+1], tra_num, timeout)
        trajectories.extend(sub_trajectories)

    # 重复步数，用于机械臂运动稳定
    if len(trajectories) > 0:
        steps = 30
        for _ in range(steps):
            trajectories.append(deepcopy(trajectories[-1]))

    return trajectories

def reset_panda_trajectories(env):
    # 机械臂的起始关节角度
    q_start = env._data.qpos[env._panda_dof_ids]
    
    q_goal = deepcopy(_PANDA_HOME)
    
    # 选择规划算法 支持: RRT, RRT-Connect, BIT*
    ompl_tool = OMPLTool(env, algorithm="BIT*")
    # 每段规划的路径包含多少个中间点
    trajectory_num = 100
    # 求解规划路径最大耗时 秒
    timeout = 10.0
    
    trajectories = trajectory_plan(ompl_tool, q_start, q_goal, trajectory_num, timeout)
    
    # 重复步数，用于机械臂运动稳定
    if len(trajectories) > 0:
        steps = 30
        for _ in range(steps):
            trajectories.append(deepcopy(trajectories[-1]))

    return trajectories

def trajectory_plan(ompl_tool, q_start, q_goal, trajectory_num, timeout):
    # 最大求解次数
    max_try_times = 50

    solve_flag = False
    try_time = 0
    while not solve_flag and try_time < max_try_times:
        trajectories, solve_flag = ompl_tool.trajectory_plan(q_start, q_goal, trajectory_num, timeout)
        try_time += 1
    if not solve_flag:
        raise RuntimeError("未找到无碰撞路径")
    return trajectories