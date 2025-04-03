import copy
import numpy as np
import mujoco
from ompl import base as ob
from ompl import geometric as og
from dm_robotics.transformations import transformations as tr

from manipulation.gpt_reward_api import get_eef_pos_and_quat

PANDA_JOINTS_RANGE = [
    [-2.8973, 2.8973],  # panda_joint1
    [-1.7628, 1.7628],  # panda_joint2
    [-2.8973, 2.8973],  # panda_joint3
    [-3.0718, -0.0698], # panda_joint4
    [-2.8973, 2.8973],  # panda_joint5
    [-0.0175, 3.7525],  # panda_joint6
    [-2.8973, 2.8973],  # panda_joint7
]

class OMPLTool:
    def __init__(self, env, algorithm="BIT*", joints_range=PANDA_JOINTS_RANGE):
        self.env = env
        self.algorithm = algorithm
        self.joints_range = joints_range
        self.panda_joint_num = len(joints_range)

    def is_valid(self, state):
        """
        碰撞检查函数，当前逻辑 对比执行下一步状态前后，环境中的碰撞总数量
        """
        before_ncon = self.env.collision_info()
        
        # 从 OMPL 的 state object 提取数值
        q_state = np.array([state[i] for i in range(len(self.joints_range))])

        # 计算下一步的碰撞状态
        self.env.data.qpos[self.env._panda_dof_ids] = q_state
        mujoco.mj_forward(self.env.model, self.env.data)
        
        after_ncon = self.env.collision_info()
        
        # 没有产生新碰撞时才有效
        if after_ncon > before_ncon:
            res = False
        else:
            res = True
        # print(f"before_ncon: {before_ncon}, after_ncon: {after_ncon}, is_valid: {res}\n------------")
        return res
    
    def trajectory_plan(self, q_start, q_goal, trajectory_num=100, timeout=5.0):
        """
        给定起点和终点关节角，进行无碰撞的路径规划

        q_start: 当前各关节角度

        q_goal: 目标各关节角度
        
        trajectory_num: 规划的路径包含多少个中间点
        
        timeout: 规划最大耗时
        """
        # 预先保存环境信息
        state_qpos = self.env._data.qpos.copy() # 保存关节位置
        saved_time = self.env._data.time  # 保存当前时间
        saved_ctrl = self.env._data.ctrl.copy()  # 保存控制信号

        # 定义状态空间（机械臂有 7 个关节）
        space = ob.RealVectorStateSpace(self.panda_joint_num)
        # 设置关节范围
        bounds = ob.RealVectorBounds(self.panda_joint_num)
        for i in range(self.panda_joint_num):
            bounds.setLow(i, self.joints_range[i][0])  # 下限
            bounds.setHigh(i, self.joints_range[i][1])  # 上限
        # 将边界应用到状态空间
        space.setBounds(bounds)

        # 设定状态有效性检查器
        validity_checker = ob.StateValidityCheckerFn(self.is_valid)

        # 将碰撞检测器应用到状态空间
        space_info = ob.SpaceInformation(space)
        space_info.setStateValidityChecker(validity_checker)

        # 设定起始和目标状态
        start = ob.State(space)
        goal = ob.State(space)
        for i in range(self.panda_joint_num):
            start[i] = q_start[i]
            goal[i] = q_goal[i]

        # 设置问题
        problem = ob.ProblemDefinition(space_info)
        problem.setStartAndGoalStates(start, goal)

        # 设置规划器
        if self.algorithm == "RRT":
            planner = og.RRT(space_info)
        elif self.algorithm == "RRT-Connect":
            planner = og.RRTConnect(space_info)
        elif self.algorithm == "BIT*":
            planner = og.BITstar(space_info)
        else:
            raise ValueError("不支持的规划器")
        
        planner.setProblemDefinition(problem)

        # 进行路径搜索
        angle_trajectories = []
        solve_flag = planner.solve(timeout)
        if solve_flag:
            print("找到可行路径")
            path = problem.getSolutionPath() # 只返回较少的关键状态点
            path.interpolate(trajectory_num)  # 自适应插值让路径更平滑

            for i in range(path.getStateCount()):
                tra = []
                for j in range(self.panda_joint_num):
                    tra.append(path.getState(i)[j])
                angle_trajectories.append(tra)

        # 恢复之前状态
        self.env._data.qpos = state_qpos
        self.env._data.time = saved_time
        self.env._data.ctrl = saved_ctrl
        mujoco.mj_forward(self.env._model, self.env._data)

        # 将关节角度值转为机械臂末端的位置与四元数
        trajectories = []
        for joint_angles in angle_trajectories:
            self.env._data.qpos[self.env._panda_dof_ids] = joint_angles
            mujoco.mj_forward(self.env._model, self.env._data)
            # 获取当前位置和姿态
            current_pos, current_quat = get_eef_pos_and_quat(self.env._data)
            current_pos = list(current_pos)
            current_quat = list(current_quat)
            current_pos.extend(current_quat)
            trajectories.append(list(current_pos))
        
        # 恢复之前状态
        self.env._data.qpos = state_qpos
        self.env._data.time = saved_time
        self.env._data.ctrl = saved_ctrl
        mujoco.mj_forward(self.env._model, self.env._data)

        return trajectories, solve_flag


def point_to_joints(env, target_pos, target_quat, joints_range=PANDA_JOINTS_RANGE):
    """
    计算IK逆运动学, 找到使机器人末端接近目标位置和姿态时，机械臂各关节的角度
    """
    state_qpos = env._data.qpos.copy() # 保存关节位置
    saved_time = env._data.time  # 保存当前时间
    saved_ctrl = env._data.ctrl.copy()  # 保存控制信号

    max_iter=200
    tol=1e-4

    for i in range(max_iter):
        # 获取当前位置和姿态
        current_pos, current_quat = get_eef_pos_and_quat(env._data)

        # 计算位置误差
        pos_err = target_pos - current_pos
        
        # 计算位姿误差之前，修正四元数符号
        if np.dot(current_quat, target_quat) < 0.0:
            target_quat *= -1.0
        
        # 计算位姿四元数误差
        quat_err = tr.quat_diff_active(source_quat=current_quat, target_quat=target_quat)            
        ori_err = tr.quat_to_axisangle(quat_err) # 转换为 3 维轴角误差
        
        # 计算雅可比矩阵
        J_pos = np.zeros((3, env._model.nv))
        J_rot = np.zeros((3, env._model.nv))
        mujoco.mj_jacSite(env._model, env._data, J_pos, J_rot, env._pinch_site_id)
        # 只提取目标机械臂的关节部分
        J_pos = J_pos[:, env._panda_dof_ids]
        J_rot = J_rot[:, env._panda_dof_ids]
        
        # 组合误差
        error = np.concatenate([pos_err, ori_err],  axis=0) # (6,)
        Jac = np.vstack([J_pos, J_rot])  # 6×n 的雅可比矩阵

        # 使用伪逆雅可比计算关节角变化量
        dq = np.linalg.pinv(Jac) @ error  
        
        # 更新关节角
        current_joints = copy.deepcopy(env._data.qpos[env._panda_dof_ids])
        new_joints = current_joints + dq

        # 逐个关节处理，确保角度在范围内
        for j in range(len(env._panda_dof_ids)):
            new_joints[j] = normalize_angle(new_joints[j], joints_range[j][0], joints_range[j][1])
            new_joints[j] = np.clip(new_joints[j], joints_range[j][0], joints_range[j][1])
            env._data.qpos[env._panda_dof_ids[j]] = new_joints[j]

        mujoco.mj_forward(env._model, env._data)  # 更新仿真状态
        
        # 误差足够小则停止
        if np.linalg.norm(error) < tol:
            break

    q_goal = copy.deepcopy(env._data.qpos[env._panda_dof_ids])

    # current_pos, current_quat = get_eef_pos_and_quat(env._data)
    # print("target_pos: ", target_pos)
    # print("target_quat: ", target_quat)
    # print("current_pos: ", current_pos)
    # print("current_quat: ", current_quat)
    # print("q_goal: ", q_goal)

    # 恢复碰撞检测之前状态
    env._data.qpos = state_qpos
    env._data.time = saved_time
    env._data.ctrl = saved_ctrl
    mujoco.mj_forward(env._model, env._data)

    return q_goal


def normalize_angle(angle, min_angle, max_angle):
    """将角度归一化到 [min_angle, max_angle] 之间"""
    while angle < min_angle:
        angle += 2 * np.pi
    while angle > max_angle:
        angle -= 2 * np.pi
    return angle
