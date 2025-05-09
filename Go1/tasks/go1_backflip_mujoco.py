import mujoco
import torch
import numpy as np
import cv2
import os
import gymnasium as gym
from gymnasium import spaces


def quat_mul(q1, q2):
    """
    Quaternion product: q1 * q2
    Args:
    q1, q2: Quaternions in [w, x, y, z] order
    Shape: Single quaternion (4,) or batch (N, 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1) if q1.ndim > 1 else torch.tensor([w, x, y, z], device=q1.device)


class Env(gym.Env):
    def __init__(self, xml_file="./assets/go1/xml/go1_correct.xml", device='cuda' if torch.cuda.is_available() else 'cpu', cfg=None):
        """
            raw_obs: 
                base orn (3), joint pos (12), joint vel (12), prev action (12), command (3)
            observation: raw_obs * history_len
            state: 
                linear velocity (3), angular vel (3), com height (1), foot contact (4), 
                gravity (3), friction (1), restitution (1), stage (5)
        """
        self.video_writer = None
        self.frame_count = 0
        self.cfg = cfg
        self.raw_obs_dim = 3 + 12*3 + 3
        self.history_len = self.cfg["env"]["history_len"]
        self.cfg['env']['numObservations'] = self.raw_obs_dim * self.history_len
        self.cfg['env']['numStates'] = 3 + 3 + 1 + 4 + 3 + 1 + 1 + 5 
        self.sim_dt = self.cfg["sim"]["sim_dt"]
        self.control_dt = self.cfg["sim"]["con_dt"]
        self.num_actions = 12
        self.num_dofs = 12
        self.num_obs = self.cfg['env']['numObservations']
        # sigle env in mujoco
        self.num_envs = 1 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # for randomization
        self.is_randomized = self.cfg["env"]["randomize"]["is_randomized"]
        self.rand_period_motor_strength_s = self.cfg["env"]["randomize"]["rand_period_motor_strength_s"]
        self.rand_period_gravity_s = self.cfg["env"]["randomize"]["rand_period_gravity_s"]
        self.rand_period_motor_strength = int(self.rand_period_motor_strength_s/self.control_dt + 0.5)
        self.rand_period_gravity = int(self.rand_period_gravity_s/self.control_dt + 0.5)
        self.rand_range_body_mass = self.cfg["env"]["randomize"]["rand_range_body_mass"]
        self.rand_range_com_pos_x = self.cfg["env"]["randomize"]["rand_range_com_pos_x"]
        self.rand_range_com_pos_y = self.cfg["env"]["randomize"]["rand_range_com_pos_y"]
        self.rand_range_com_pos_z = self.cfg["env"]["randomize"]["rand_range_com_pos_z"]
        self.rand_range_dof_pos = self.cfg["env"]["randomize"]["rand_range_init_dof_pos"]
        self.rand_range_root_vel = self.cfg["env"]["randomize"]["rand_range_init_root_vel"]
        self.rand_range_motor_strength = self.cfg["env"]["randomize"]["rand_range_motor_strength"]
        self.rand_range_gravity = self.cfg["env"]["randomize"]["rand_range_gravity"]
        self.rand_range_friction = self.cfg["env"]["randomize"]["rand_range_friction"]
        self.rand_range_restitution = self.cfg["env"]["randomize"]["rand_range_restitution"]
        self.rand_range_motor_offset = self.cfg["env"]["randomize"]["rand_range_motor_offset"]
        self.noise_range_dof_pos = self.cfg["env"]["randomize"]["noise_range_dof_pos"]
        self.noise_range_dof_vel = self.cfg["env"]["randomize"]["noise_range_dof_vel"]
        self.noise_range_body_orn = self.cfg["env"]["randomize"]["noise_range_body_orn"]
        self.n_lag_action_steps = self.cfg["env"]["randomize"]["n_lag_action_steps"]
        self.n_lag_imu_steps = self.cfg["env"]["randomize"]["n_lag_imu_steps"]
        self.common_step_counter = 0
        super(Env, self).__init__()
        
        # load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # save to video 
        if self.cfg["test"]: 
            model_num = self.cfg["model_num"]
            self.video_path = "output_videos/"
            os.makedirs(self.video_path, exist_ok=True)
            self.video_filename = os.path.join(self.video_path, "backflip_{}.mp4".format(model_num))
            # self.width, self.height = 1280, 960
            self.width, self.height = 640, 480
            self.video_writer = cv2.VideoWriter(
                self.video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 200, (self.width, self.height)
            )
            self.video_count = 0
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(126,), dtype=np.float32),
            "states": spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32),
        })
        self.num_states = 3 + 3 + 1 + 4 + 3 + 1 + 1 + 5
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32)

        # get the body and idx in MuJoCo 3.2.7 or higher vision
        body_names = []
        names_str = self.model.names
        for i in range(self.model.nbody):
            name_adr = self.model.name_bodyadr[i]
            if name_adr >= 0:
                next_null = names_str.find(b'\0', name_adr)
                if next_null == -1:
                    next_null = len(names_str)
                name = names_str[name_adr:next_null].decode('utf-8')
                body_names.append(name)
        
        self.body_name_map = {name: i for i, name in enumerate(body_names)} 
        self.trunk_id = self.body_name_map.get("trunk", -1)
        if self.trunk_id == -1:
            raise ValueError("Body 'trunk' not found in the model")
        base_names = ['trunk']
        hip_names = [s for s in self.body_name_map if 'hip' in s]
        thigh_names = [s for s in self.body_name_map if 'thigh' in s]
        calf_names = [s for s in self.body_name_map if 'calf' in s]
        foot_names = [s for s in self.body_name_map if 'foot' in s]
        terminate_touch_names = base_names + hip_names
        undesired_touch_names = thigh_names + calf_names
        
        # find foot & knee & hip & base's index
        self.hip_indices = torch.zeros(
            len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.body_name_map[hip_names[i]]
            
        self.thigh_indices = torch.zeros(
            len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.body_name_map[thigh_names[i]]
            
        self.calf_indices = torch.zeros(
            len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.body_name_map[calf_names[i]]
            
        self.foot_indices = torch.zeros(
            len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.body_name_map[foot_names[i]]
        
        self.terminate_touch_indices = torch.zeros(
            len(terminate_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(terminate_touch_names)):
            self.terminate_touch_indices[i] = self.body_name_map[terminate_touch_names[i]]
            
        self.undesired_touch_indices = torch.zeros(
            len(undesired_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(undesired_touch_names)):
            self.undesired_touch_indices[i] = self.body_name_map[undesired_touch_names[i]]
        
        self.base_index = self.trunk_id
        self.max_episode_length_s = self.cfg["env"]["learn"]["episode_length_s"]
        self.max_episode_length = int(self.max_episode_length_s/self.control_dt + 0.5)
        self.reward_names = self.cfg["env"]["reward_names"]
        self.cost_names = self.cfg["env"]["cost_names"]
        self.stage_names = self.cfg["env"]["stage_names"]
        self.num_rewards = len(self.reward_names)
        self.num_costs = len(self.cost_names)
        self.num_stages = len(self.stage_names)
        self.action_smooth_weight = self.cfg["env"]["control"]["action_smooth_weight"]
        self.action_scale = self.cfg["env"]["control"]["action_scale"]
        self.action_scale = 1.0
        
        # allocate buffers
        self.rew_buf = torch.zeros((self.num_envs, self.num_rewards), dtype=torch.float32, device=self.device)
        self.cost_buf = torch.zeros((self.num_envs, self.num_costs), dtype=torch.float32, device=self.device)
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages), dtype=torch.float32, device=self.device)
        # init stage is standing
        self.stage_buf[0, 0] = 1.0  
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        # check the robot tumbling
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.land_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # init the state tenso
        self.dof_positions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        self.base_quaternions = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vels = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vels = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_positions = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        # sync state
        self._sync_state_from_mujoco()
        
        # get default base frame pose
        base_init_state = []
        base_init_state += self.cfg["env"]["init_base_pose"]["pos"]
        base_init_state += self.cfg["env"]["init_base_pose"]["quat"]
        base_init_state += self.cfg["env"]["init_base_pose"]["lin_vel"]
        base_init_state += self.cfg["env"]["init_base_pose"]["ang_vel"]
        self.base_init_state = torch.tensor(
            base_init_state, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # get default joint (DoF) position
        self.named_default_joint_positions = self.cfg["env"]["default_joint_positions"]
        self.dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        self.default_dof_positions = torch.zeros_like(
            self.dof_positions, dtype=torch.float32, device=self.device, requires_grad=False)
        for i, name in enumerate(self.named_default_joint_positions.keys()):
            self.default_dof_positions[0][i] = self.named_default_joint_positions[self.dof_names[i]]

        # for inner variables
        self.world_x = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_y = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_z = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_x[:, 0] = 1.0
        self.world_y[:, 1] = 1.0
        self.world_z[:, 2] = 1.0
        self.joint_targets = torch.zeros(
            (self.num_envs, self.num_dofs), 
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_dofs), 
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.lag_joint_target_buffer = [torch.zeros_like(self.dof_positions, device=self.device) for _ in range(self.n_lag_action_steps + 1)]
        self.lag_imu_buffer = [torch.zeros_like(self.world_z) for _ in range(self.n_lag_imu_steps + 1)]
        self.prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.prev_prev_joint_targets = torch.zeros_like(self.joint_targets)
        self.gravity = torch.tensor(self.cfg['sim']['gravity'], dtype=torch.float32, device=self.device, requires_grad=False)

        # for dof limits
        self.default_dof_pos_lower_limits = torch.tensor([-0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965, -0.8029, -1.0472, -2.6965], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_pos_upper_limits = torch.tensor([0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163,  0.8029,  4.1888, -0.9163], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_vel_upper_limits = torch.tensor([50., 28., 28., 50., 28., 28., 50., 28., 28., 50., 28., 28.], dtype=torch.float32, device=self.device, requires_grad=False)
        dof_pos_lower_limits = []
        dof_pos_upper_limits = []
        for joint_name in ['hip', 'thigh', 'calf']:
            joint_dict = self.cfg["env"]["learn"][f"{joint_name}_joint_limit"]
            dof_pos_lower_limits.append(joint_dict['lower'] if 'lower' in joint_dict.keys() else -np.inf)
            dof_pos_upper_limits.append(joint_dict['upper'] if 'upper' in joint_dict.keys() else np.inf)
        self.dof_pos_lower_limits = torch.tensor(
            dof_pos_lower_limits*4, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_lower_limits = torch.maximum(self.dof_pos_lower_limits, self.default_dof_pos_lower_limits)
        self.dof_pos_upper_limits = torch.tensor(
            dof_pos_upper_limits*4, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_upper_limits = torch.minimum(self.dof_pos_upper_limits, self.default_dof_pos_upper_limits)
        self.dof_vel_upper_limits = torch.tensor(
            self.cfg["env"]["learn"]["joint_vel_upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_vel_upper_limits = torch.minimum(self.dof_vel_upper_limits, self.default_dof_vel_upper_limits)
        self.dof_torques_upper_limits = torch.tensor([200] * 12, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # for noise observation
        self.est_base_body_orns = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.est_dof_positions = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.est_dof_velocities = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device, requires_grad=False)    

        # for observation and action symmetric matrix
        self.joint_sym_mat = torch.zeros((self.num_dofs, self.num_dofs), device=self.device, dtype=torch.float32, requires_grad=False)
        self.joint_sym_mat[:3, 3:6] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[0, 3] = -1.0
        self.joint_sym_mat[3:6, :3] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[3, 0] = -1.0
        self.joint_sym_mat[6:9, 9:12] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[6, 9] = -1.0
        self.joint_sym_mat[9:12, 6:9] = torch.eye(3, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[9, 6] = -1.0
        self.obs_sym_mat = torch.zeros((self.num_obs, self.num_obs), device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat = torch.eye(self.raw_obs_dim, device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat[1, 1] = -1.0
        for i in range(3):
            raw_obs_sym_mat[(3+self.num_dofs*(i)):(3+self.num_dofs*(i+1)), (3+self.num_dofs*(i)):(3+self.num_dofs*(i+1))] = self.joint_sym_mat.clone()
        raw_obs_sym_mat[3+3*self.num_dofs:, 3+3*self.num_dofs:] = torch.eye(3, device=self.device, dtype=torch.float32)
        for i in range(self.history_len):
            self.obs_sym_mat[(self.raw_obs_dim*i):(self.raw_obs_dim*(i+1)), (self.raw_obs_dim*i):(self.raw_obs_dim*(i+1))] = raw_obs_sym_mat.clone()
        self.state_sym_mat = torch.eye(self.num_states - self.num_stages, device=self.device, dtype=torch.float32, requires_grad=False)
        self.state_sym_mat[1, 1] = -1.0
        self.state_sym_mat[3, 3] = -1.0
        self.state_sym_mat[5, 5] = -1.0
        self.state_sym_mat[7:11, 7:11] = 0
        self.state_sym_mat[7, 8] = 1.0
        self.state_sym_mat[8, 7] = 1.0
        self.state_sym_mat[9, 10] = 1.0
        self.state_sym_mat[10, 9] = 1.0
        self.state_sym_mat[12, 12] = -1.0
        self.extras = {}
        self.friction_coeffs = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.clip_obs = self.cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg["env"].get("clipActions", np.Inf)
        self.obs_dict = {}
        self.INITPOS = self.data.qpos
        self.contact_forces = torch.zeros((self.num_envs, len(self.body_name_map), 3), dtype=torch.float, device=self.device)
        
    def _sync_state_from_mujoco(self):
        self.dof_positions[:] = torch.from_numpy(self.data.qpos[7:]).to(self.device)[np.newaxis, :]
        self.dof_velocities[:] = torch.from_numpy(self.data.qvel[6:]).to(self.device)[np.newaxis, :]
        self.base_quaternions[:] = torch.from_numpy(self.data.qpos[3:7]).to(self.device)[np.newaxis, :]
        self.base_lin_vels[:] = torch.from_numpy(self.data.qvel[0:3]).to(self.device)[np.newaxis, :]
        self.base_ang_vels[:] = torch.from_numpy(self.data.qvel[3:6]).to(self.device)[np.newaxis, :]
        self.base_positions[:] = torch.from_numpy(self.data.qpos[0:3]).to(self.device)[np.newaxis, :]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # reset robot's base frame pose
        self.data.qpos[7:] = torch.zeros(12, device=self.device).cpu().numpy()
        self.data.qvel[6:] = torch.zeros(12, device=self.device).cpu().numpy()
        self.data.qpos[:7] = torch.tensor([0, 0, 0.35, 1, 0, 0, 0], device=self.device).cpu().numpy()
        self.data.qvel[:6] = torch.zeros(6, device=self.device).cpu().numpy()
        self._sync_state_from_mujoco()
        # reset all thing to start state
        positions_offset = torch.ones((1, self.num_dofs), dtype=torch.float32, device=self.device)
        self.dof_positions[0] = torch.tensor(self.default_dof_positions.cpu().numpy() * positions_offset[0].cpu().numpy(), dtype=torch.float32, device=self.device)
        self.dof_velocities[0] = 0.0
        # reset inner variables
        self.joint_targets.zero_()
        self.joint_targets[:] = self.default_dof_positions
        self.prev_joint_targets[0] = self.joint_targets[0].clone()
        self.prev_prev_joint_targets[0] = self.joint_targets[0].clone()
        self.prev_actions.zero_()
        self.prev_actions[0] = (self.joint_targets[0] - self.default_dof_positions[0])/self.action_scale
        # reset buffers
        self.progress_buf.zero_()
        self.reset_buf.zero_()
        self.fail_buf.zero_()
        self.stage_buf.zero_()
        self.stage_buf[0, 0] = 1.0
        self.is_half_turn_buf.zero_()
        self.is_one_turn_buf.zero_()
        self.start_time_buf.zero_()
        self.start_time_buf[0] = torch.tensor(1.0).to(self.device)
        self.cmd_time_buf.zero_()
        self.land_time_buf.zero_()
        for i in range(len(self.lag_joint_target_buffer)):
            self.lag_joint_target_buffer[i][:] = self.joint_targets
        for i in range(len(self.lag_imu_buffer)):
            self.lag_imu_buffer[i][:] = self.est_base_body_orns
        
        # estimate observations
        self.est_base_body_orns[0] = self._quat_rotate_inverse(self.base_quaternions[0], self.world_z[0]).clone()
        self.est_dof_positions[0] = self.dof_positions[0].clone()
        self.est_dof_velocities[0] = self.dof_velocities[0].clone()
        # calculate commands
        commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf[0] == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf[0]*self.control_dt < self.cmd_time_buf[0] + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2
        
        # reset observation
        obs = self.jit_compute_observations(
            self.est_base_body_orns, self.est_dof_positions, self.est_dof_velocities, 
            self.prev_actions, commands)
        for history_idx in range(self.history_len):
            self.obs_buf[0, history_idx*self.raw_obs_dim:(history_idx+1)*self.raw_obs_dim] = obs
        
        # reset state
        contact_forces = self.data.cfrc_ext
        foot_contact_forces = contact_forces[self.foot_indices.cpu().numpy(), :3]
        calf_contact_forces = contact_forces[self.calf_indices.cpu().numpy(), :3]
        self.states_buf[0] = self.jit_compute_states(
            self.base_quaternions, self.base_lin_vels, self.base_ang_vels, self.base_positions,
            torch.from_numpy(foot_contact_forces).to(self.device), torch.from_numpy(calf_contact_forces).to(self.device), self.gravity, 
            self.friction_coeffs[0], self.restitution_coeffs[0], self.stage_buf[0])
        if self.cfg["is_uniform_rollout"]:
            self.progress_buf = torch.randint_like(self.progress_buf, low=0, high=self.max_episode_length)
        # reset the pose and other sate
        self.data.qpos = self.INITPOS
        self.data.qpos[7:] = self.dof_positions[0].cpu().numpy()
        mujoco.mj_forward(self.model, self.data)
        self.contact_forces.fill_(0.0)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf[0], -self.clip_obs, self.clip_obs).cpu().numpy()
        self.obs_dict["states"] = torch.clamp(self.states_buf[0], -self.clip_obs, self.clip_obs).cpu().numpy()
        info = {"reset_time": 0.0}
        
        return self.obs_dict, info

    def step(self, action):
        action_tensor = torch.from_numpy(action[np.newaxis, :]).to(self.device)
        action = torch.clamp(action_tensor,  -self.clip_actions, self.clip_actions)  # clip action
        self.pre_physics_step(action)
        self._sync_state_from_mujoco()
        self.post_physics_step()
        self.obs_dict["obs"] = torch.clamp(self.obs_buf[0], -self.clip_obs, self.clip_obs).cpu().numpy()
        self.obs_dict["states"] = torch.clamp(self.states_buf[0], -self.clip_obs, self.clip_obs).cpu().numpy()
        truncated = False
        return self.obs_dict, 1, self.reset_buf.cpu().numpy(), truncated, self.extras

    def pre_physics_step(self, actions):
        self.prev_prev_joint_targets[:] = self.prev_joint_targets
        self.prev_joint_targets[:] = self.joint_targets
        actions[:, 3] = -actions[:, 0]  # FR_hip = -FL_hip
        actions[:, 9] = -actions[:, 6]  # RR_hip = -RL_hip
        actions[:, [1, 4]] = actions[:, [1, 4]].mean(dim=1, keepdim=True)  # FL_thigh = FR_thigh
        actions[:, [2, 5]] = actions[:, [2, 5]].mean(dim=1, keepdim=True)  # FL_calf = FR_calf
        actions[:, [7, 10]] = actions[:, [7, 10]].mean(dim=1, keepdim=True)  # RL_thigh = RR_thigh
        actions[:, [8, 11]] = actions[:, [8, 11]].mean(dim=1, keepdim=True)  # RL_calf = RR_calf
        self.prev_actions[:] = actions

        # use PD control
        smooth_weight = self.action_smooth_weight
        self.joint_targets[:] = (smooth_weight * (actions * self.action_scale + self.default_dof_positions) + (1.0 - smooth_weight) * self.joint_targets)
        for i in range(4):
            # calculate torques using PD control
            self.lag_joint_target_buffer = self.lag_joint_target_buffer[1:] + [self.joint_targets]
            joint_targets = self.lag_joint_target_buffer[0]
            current_dof_positions = torch.from_numpy(self.data.qpos[7:]).to(self.device)
            current_dof_velocities = torch.from_numpy(self.data.qvel[6:]).to(self.device)
            if self.stage_buf[0, 2] == 1.0:  
                stiffness = 135
                damping = 0.2
            else:
                stiffness = 30
                damping = 0.5
            if self.stage_buf[0, 4] == 1.0:
                stiffness = 5
                damping = 10
            torques = stiffness*(joint_targets - current_dof_positions) - damping*current_dof_velocities
            torques = torch.clip(torques*self.motor_strengths, -self.dof_torques_upper_limits, self.dof_torques_upper_limits)
            self.data.ctrl[:] = torques.cpu().numpy()
            mujoco.mj_step(self.model, self.data)
            if self.cfg["test"]:
                self.renderer.update_scene(self.data, camera="track")
                frame = self.renderer.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_count = self.video_count + 1
                print("write frame to video, count:", self.video_count)
                self.video_writer.write(frame)

    def post_physics_step(self):
        self.progress_buf += 1
        self.common_step_counter += 1
        torso_pos = torch.from_numpy(self.data.xpos[self.trunk_id]).to(self.device)
        torso_quat = torch.from_numpy(self.data.xquat[self.trunk_id]).to(self.device)
        torso_vel = torch.from_numpy(self.data.qvel[0:3]).to(self.device)
        torso_ang_vel = torch.from_numpy(self.data.qvel[3:6]).to(self.device)
        # stage 0: stand, stage 1: down, stage 2: jump, stage 3: back turn, stage 4: land
        # =================== calculate rewards =================== #
        # 1. CoM
        com_height = torso_pos[2]
        self.rew_buf[:, 0] =  self.stage_buf[:, 0]*(-torch.abs(com_height - 0.35))
        self.rew_buf[:, 0] += self.stage_buf[:, 1]*(-torch.abs(com_height - 0.2))
        self.rew_buf[:, 0] += self.stage_buf[:, 2]*(torch.where(com_height <= 1.0, com_height * 2, torch.tensor(0.0, device=self.device)))
        self.rew_buf[:, 0] += self.stage_buf[:, 3]*(torch.where(com_height <= 1.0, com_height * 2, torch.tensor(0.0, device=self.device)))
        self.rew_buf[:, 0] += self.stage_buf[:, 4]*(-torch.abs(com_height - 0.35))
        # 2. balance
        body_z = torch.unsqueeze(self._quat_rotate_inverse(torso_quat, self.world_z[0]).to(self.device), dim=0).to(dtype=torch.float32)
        self.rew_buf[:, 1] =  self.stage_buf[:, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 1]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 2]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 3]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 4]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        # 3. vel
        base_lin_vels = torch.unsqueeze(torso_vel, dim=0)
        base_ang_vels = torch.unsqueeze(torso_ang_vel, dim=0)
        vel_penalty = torch.square(base_lin_vels[:, 0]) + torch.square(base_lin_vels[:, 1]) + torch.square(base_ang_vels[:, 2])
        base_ang_vel_y = base_ang_vels[:, 1]
        self.rew_buf[:, 2] =  self.stage_buf[:, 0]*(-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 1]*(-vel_penalty)
        self.rew_buf[:, 2] += self.stage_buf[:, 2]*(1.0 - self.is_one_turn_buf)*(-base_ang_vel_y) * 4
        self.rew_buf[:, 2] += self.stage_buf[:, 3]*(1.0 - self.is_one_turn_buf)*(-base_ang_vel_y) * 4
        self.rew_buf[:, 2] += self.stage_buf[:, 4]*(-vel_penalty) * 5
        # 4. enery
        torques = torch.from_numpy(self.data.ctrl).to(self.device)
        self.rew_buf[:, 3] = -torch.mean(torch.square(torques))/1000
        # 5. style
        dof_pos = torch.from_numpy(self.data.qpos[7:]).to(self.device)
        self.rew_buf[:, 4] = -torch.mean(torch.square(dof_pos - self.default_dof_positions))
        # ========================================================= #
        # ==================== calculate costs ==================== #
        foot_contact_threshold = 0.25
        self.contact_forces.fill_(0.0)
        n_contacts = len(self.data.contact)
        if n_contacts > 0:
            for i in range(n_contacts):
                contact = self.data.contact[i]
                contact_idx = contact.geom2
                body_idx = self.model.geom_bodyid[contact_idx]
                force = self._extract_contact_force(i)
                self.contact_forces[:, body_idx] = torch.from_numpy(force).to(self.device)     
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        calf_contact_forces = self.contact_forces[:, self.calf_indices, :]
        foot_contact = ((torch.norm(foot_contact_forces, dim=2) > 10.0) | (torch.norm(calf_contact_forces, dim=2) > 10.0)).type(torch.float)
        # 1. high cost
        self.cost_buf[:, 0] =  self.stage_buf[:, 0]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 1]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 2]*(1.0 - (foot_contact[:, 2] + foot_contact[:, 3])/2.0)
        self.cost_buf[:, 0] += self.stage_buf[:, 3]*(foot_contact_threshold)
        self.cost_buf[:, 0] += self.stage_buf[:, 4]*(foot_contact_threshold)
        # 2. body contact cost
        term_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1)
        self.cost_buf[:, 1] =  self.stage_buf[:, 0]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 2]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 3]*undesired_contact.type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 4]*undesired_contact.type(torch.float)
        # 3. dof cost
        self.cost_buf[:, 2] = torch.mean((
            (self.dof_positions < self.dof_pos_lower_limits) | (self.dof_positions > self.dof_pos_upper_limits)
            ).to(torch.float), dim=-1)
        # 4.  dof vel cost
        self.cost_buf[:, 3] = torch.mean(
            (torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        # 5. dof torques cost
        torques = torch.from_numpy(self.data.ctrl).to(self.device)
        self.cost_buf[:, 4] = torch.mean(
            (torch.abs(torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)
        # ========================================================= #
        # update stage
        # have to handle in the following order: N -> N-1 -> N-2 ... -> 1 -> 0.
        from3_to4 = torch.logical_and(
            self.stage_buf[:, 3] == 1.0, torch.logical_and(
                foot_contact.mean(dim=-1) > 0.0,
                self.is_one_turn_buf
            )
        ).type(torch.float32)
        self.stage_buf[:, 3] = (1.0 - from3_to4)*self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4)*self.stage_buf[:, 4]
        from2_to3 = torch.logical_and(
            self.stage_buf[:, 2] == 1.0, 
            foot_contact.mean(dim=-1) < 0.1
        ).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to3)*self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3)*self.stage_buf[:, 3]
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0, torch.logical_and(
                com_height <= 0.24, 
                foot_contact.mean(dim=-1) >= 0.9
            )
        ).type(torch.float32)
        self.stage_buf[:, 1] = (1.0 - from1_to2)*self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2)*self.stage_buf[:, 2]
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0, torch.logical_and(
                self.progress_buf*self.control_dt > self.start_time_buf, torch.logical_and(
                    com_height >= 0.3, 
                    self.is_half_turn_buf == 0
                )
            )
        ).type(torch.float32)
        self.stage_buf[:, 0] = (1.0 - from0_to1)*self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1)*self.stage_buf[:, 1]
        # check the robot tumbling
        self.is_half_turn_buf[:] = torch.logical_or(
            self.is_half_turn_buf, torch.logical_and(
                body_z[:, 0] < 0, body_z[:, 2] < 0)).type(torch.long)
        self.is_one_turn_buf[:] = torch.logical_or(
            self.is_one_turn_buf, torch.logical_and(
                self.is_half_turn_buf, torch.logical_and(
                    body_z[:, 0] >= -0.15, body_z[:, 2] >= 0.98))).type(torch.long)
        land_masks = torch.logical_and(self.land_time_buf == 0, self.stage_buf[:, 4] == 1).type(torch.float32)
        self.land_time_buf[:] = land_masks*(self.progress_buf*self.control_dt) + (1.0 - land_masks)*self.land_time_buf
        cmd_masks = torch.logical_and(self.cmd_time_buf == 0, self.stage_buf[:, 1] == 1).type(torch.float32)
        self.cmd_time_buf[:] = cmd_masks*(self.progress_buf*self.control_dt) + (1.0 - cmd_masks)*self.cmd_time_buf
        body_contacts = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        landing_wo_turns = torch.logical_and(self.stage_buf[:, 3] == 1.0, torch.logical_and(foot_contact.mean(dim=-1) > 0.0, 1 - self.is_half_turn_buf))
        self.fail_buf[:] = torch.logical_or(body_contacts, landing_wo_turns).type(torch.long)

        # calculate reset buffer
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length, 
            torch.ones_like(self.reset_buf), self.fail_buf
        )

        # estimate observations
        est_base_body_orns = self._quat_rotate_inverse(self.base_quaternions[0], self.world_z[0]).clone()
        self.est_dof_positions = self.dof_positions.clone()
        self.est_dof_velocities = self.dof_velocities.clone()
        self.lag_imu_buffer = self.lag_imu_buffer[1:] + [est_base_body_orns]
        self.est_base_body_orns[:] = self.lag_imu_buffer[0]
        
        # calculate commands
        commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf*self.control_dt < self.cmd_time_buf + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2
        self._sync_state_from_mujoco()

        # update observation buffer
        obs = self.jit_compute_observations(
            self.est_base_body_orns, self.est_dof_positions, self.est_dof_velocities, 
            self.prev_actions, commands)
        self.obs_buf[:, :-self.raw_obs_dim] = self.obs_buf[:, self.raw_obs_dim:].clone()
        self.obs_buf[:, -self.raw_obs_dim:] = obs

        self.states_buf[:] = self.jit_compute_states(
                self.base_quaternions, self.base_lin_vels, self.base_ang_vels, self.base_positions,
                foot_contact_forces[0], calf_contact_forces[0], self.gravity, 
                self.friction_coeffs[0], self.restitution_coeffs[0], self.stage_buf[0])
        # return extra
        self.extras['costs'] = self.cost_buf.clone()
        self.extras['fails'] = self.fail_buf.clone()
        self.extras['next_obs'] = self.obs_buf.clone()
        self.extras['next_states'] = self.states_buf.clone()
        self.extras['dones'] = self.reset_buf.clone()
        self.extras['reward'] = self.rew_buf.clone()

        # reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: self.reset()
        
    def jit_compute_observations(self, body_orns, dof_pos, dof_vel, prev_actions, commands):
        obs = torch.cat([body_orns, dof_pos, dof_vel, prev_actions, commands], dim=1)
        return obs

    def jit_compute_states(self,
        base_quaternions, base_lin_vels, base_ang_vels, base_positions, 
        foot_contact_forces, calf_contact_forces, gravity, friction_coeffs, restitution_coeffs, stages,
    ):
        bb_lin_vels = self._quat_rotate_inverse(base_quaternions[0], base_lin_vels[0])
        bb_ang_vels = self._quat_rotate_inverse(base_quaternions[0], base_ang_vels[0])
        com_height = base_positions[0, 2:3]
        foot_contacts = ((torch.norm(foot_contact_forces, dim=1) > 1.0) \
                        | (torch.norm(calf_contact_forces, dim=1) > 1.0)).type(torch.float)
        gravities = gravity.unsqueeze(0).repeat(base_quaternions.shape[0], 1)[0]
        states = torch.cat([
            bb_lin_vels, bb_ang_vels, com_height, foot_contacts, 
            gravities, friction_coeffs, restitution_coeffs, stages], dim=-1)
        return states

    def _quat_rotate_inverse(self, quat, vec):
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        q_conj = torch.stack([w, -x, -y, -z], dim=-1)
        vec_quat = torch.tensor([0, vec[0], vec[1], vec[2]], device=self.device)
        result = quat_mul(q_conj, quat_mul(vec_quat, quat))
        return result[1:4]

    def _extract_contact_force(self, contact_id):
        # create a numpy to restore result
        result = np.zeros((6, 1), dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_id, result)  # get the contact force
        contact = self.data.contact[contact_id]
        normal = contact.frame[:3]
        normal_force = result[0]
        global_force = normal_force * normal
        return global_force
    
    
    def render(self, mode="human"):
        if not hasattr(self, "viewer"):
            import mujoco_viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()
        return None

    def close(self):
        if self.cfg["test"]:
            self.video_writer.release()
    

