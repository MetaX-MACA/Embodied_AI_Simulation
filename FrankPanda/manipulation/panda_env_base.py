import copy
import mujoco
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Tuple, Dict
from gymnasium import spaces

from manipulation.gpt_reward_api import *
from manipulation.utils import _PANDA_HOME, PANDA_JOINT_NUM
from mujoco_base import MujocoGymEnv
from panda_controller import operation_space


class PandaEnv(MujocoGymEnv):
    def __init__(self, xml_path, action_scale= np.asarray([0.1, 1]), max_step=1000,
                 seed=0, control_dt=0.02, physics_dt=0.002, time_limit=10.0, render_mode="rgb_array", 
                 image_obs=False, save_video=False, video_path="./panda_env.mp4", asset_joint_list=[]):
        self._action_scale = action_scale
        self.dz = max_step
        self.init_step = max_step
        super().__init__(
            xml_path=xml_path, 
            seed=seed, 
            control_dt=control_dt, 
            physics_dt=physics_dt, 
            time_limit=time_limit, 
            render_mode=render_mode, 
            image_obs=image_obs,
            save_video=save_video,
            video_path=video_path,
        )

        # 关节id
        self._panda_dof_ids = np.asarray(
            [get_joint_id_from_name(self._model, f"panda_joint{i}") for i in range(1, PANDA_JOINT_NUM+1)]
        )
        # 左夹具joint_id
        self._panda_left_finger_joint_id = self._model.joint(f"left_driver_joint").id
        # 右夹具joint_id
        self._panda_right_finger_joint_id = self._model.joint(f"right_driver_joint").id
        
        # 速度控制器
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, PANDA_JOINT_NUM+1)]
        )
        # 拇指控制ID
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        # 看结构树应该是执行器末端参考坐标系
        self._pinch_site_id = get_site_id_from_name(self._model, "pinch")
        
        # 强化学习模型控制的关节id
        self._rl_joint_ids = []
        self._rl_joint_ids.extend(self._panda_dof_ids.tolist())
        self._rl_joint_ids.append(self._panda_left_finger_joint_id)
        self._rl_joint_ids.append(self._panda_right_finger_joint_id)
        
        # 强化学习模型控制的力控制器id
        self._rl_ctrl_ids = []
        self._rl_ctrl_ids.extend(self._panda_ctrl_ids.tolist())
        self._rl_ctrl_ids.append(self._gripper_ctrl_id)
        
        # 7个关节角 + 2个手指 + 夹具位置（End-effector position）+ 门角度 + 夹具状态  共7+2+3+1+1=13
        self.obs_shape = len(self._rl_joint_ids) + 5
        # for better training, we use 13 dims as observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        # 8 个力控制器, from panda.xml
        self.action_space = spaces.Box(
            low=np.array([-87, -87, -87, -87, -12, -12, -12, 0]),  # 7维位姿 + 夹具状态
            high=np.array([87, 87, 87, 87, 12, 12, 12, 1]),  # 夹具状态范围为 [0, 1]
            dtype=np.float32)
        
        self.asset_joint_list = asset_joint_list
        
        self.reward_func = None
        self.reward_joint_name = None
        self.store_env_pos = None
        return

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.dz = self.init_step
        # Reset the environment
        mujoco.mj_resetData(self._model, self._data)
        
        if self.store_env_pos is None:
            # 初始化机械臂位置
            self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
            mujoco.mj_forward(self._model, self._data)
            # 设置部分资产joint的默认值
            self.set_assets_init_joint()
        else:
            self._data.qpos = copy.deepcopy(self.store_env_pos)
            mujoco.mj_forward(self._model, self._data)

        obs, other_info = self._compute_observation()
        return obs, other_info

    def set_assets_init_joint(self):
        """
        设置部分资产joint的默认值
        """
        if len(self.asset_joint_list) > 0:
            for item in self.asset_joint_list:
                joint_name, joint_value = item[0], item[1]
                joint_id = get_joint_id_from_name(self._model, joint_name)
                self._data.qpos[joint_id] = joint_value
            mujoco.mj_forward(self._model, self._data)
        return

    def primitive_step(self, action: np.ndarray):
        """
        take a step in the environment.
        Params:
            action: np.ndarray   表示执行器末端的7维位姿表达和夹具状态值

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """        
        self.dz -= 1
        x, y, z, u, v, w, r, grasp = action
        
        new_pos = np.asarray([x, y, z])
        new_quat = np.asarray([u, v, w, r])
        
        dg = grasp * self._action_scale[1]
        ng = np.clip(dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255
        
        # 计算逆运动学
        for _ in range(self._n_substeps):
            tau = operation_space(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=new_pos,          # 目标位置
                ori=new_quat,
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs, other_info = self._compute_observation()
        terminated = self.time_limit_exceeded()
        return obs, terminated, other_info

    def step(self, action: np.ndarray):
        """
        强化学习模型使用的step
        """
        for _ in range(self._n_substeps): 
            self._data.ctrl[self._rl_ctrl_ids] = action
            
            grasp = action[-1]
            self._data.ctrl[self._gripper_ctrl_id] = grasp * 255
            mujoco.mj_step(self._model, self._data)
        
        obs, other_info = self._compute_observation()
        reward, success = self._compute_reward()
        terminated = self.time_limit_exceeded()
        return obs, reward, success, terminated, other_info

    def _compute_observation(self):
        # 7个关节角 + 2个手指 + 夹具位置（End-effector position）+ 门角度 + 夹具状态  共7+2+3+1+1=13
        obs = np.zeros(self.obs_shape)
        cnt = 0
        
        # 7个关节角 + 2个手指
        for idx in self._rl_joint_ids:
            obs[cnt] = self._data.qpos[idx]
            cnt += 1
        # 夹具位置
        eef_pos = self._data.site("pinch").xpos  # Assuming 'pinch' is the end effector's site
        obs[cnt:cnt+3] = eef_pos
        cnt += 3

        # 关节角度
        joint_angle = 0.0
        if self.reward_joint_name is not None:
            joint_angle = self._data.joint(self.reward_joint_name).qpos[0]
        obs[cnt] = joint_angle
        cnt += 1
        
        # 夹具状态 (whether the gripper is open or closed)
        gripper_state = self._data.ctrl[self._gripper_ctrl_id]  # Assuming 255 for closed
        obs[cnt] = gripper_state
        cnt += 1

        all_qpos = self._data.qpos.copy()
        pinch_pos, pinch_quat = get_eef_pos_and_quat(self._data)
        all_ctrl = np.array(self._data.ctrl, dtype=np.float32)
        
        other_info = {}
        other_info["qpos"] = all_qpos.astype(np.float32)
        other_info["pinch_pos"] = pinch_pos.astype(np.float32)
        other_info["pinch_quat"] = pinch_quat.astype(np.float32)        
        other_info["ctrl_list"] = all_ctrl

        if self.image_obs:
            other_info["images"] = {}
            for name in (self.camera_names):
                image = self.cam_rgb_render(name)
                # depth = self.cam_depth_render(name)
                # seg = self.cam_seg_render(name)
                other_info["images"][name] = image
        return obs, other_info
    
    def _compute_reward(self):
        return self.reward_func(self)[:2]
    
    def __call__(self):
        return self
    
    def collision_info(self, verbose=False):
        num_contacts = self._data.ncon
        if verbose:
            print(f"当前碰撞数量: {num_contacts}")
        
        # 临时办法
        # 忽略的碰撞对列表
        ignore_col_pair_list = [
            [['mug_col'], ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]],
        ]

        num = num_contacts
        for i in range(num_contacts):
            contact = self._data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_name = self._model.geom(geom1_id).name
            geom2_name = self._model.geom(geom2_id).name
            # 忽略一些碰撞
            for pair in ignore_col_pair_list:
                if (geom1_name in pair[0] and geom2_name in pair[1]) or \
                    (geom1_name in pair[1] and geom2_name in pair[0]):
                    num -= 1
            if verbose:
                print(f"碰撞实体对(id) {i + 1}: {geom1_id} <-> {geom2_id}")
                print(f"碰撞实体对(name) {i + 1}: {geom1_name} <-> {geom2_name}")
                print(f"Contact Point: {contact.pos}")
                print(f"Contact Normal: {contact.frame[:3]}")
                print(f"Contact Distance: {contact.dist}")
        return num
    
    def get_start_status(self):
        start_pos, start_quat = get_eef_pos_and_quat(self._data)
        action = list(copy.deepcopy(start_pos))
        action.extend(list(start_quat))
        grasp = 0.0 # 末端最大张开
        action.append(grasp)
        return action
