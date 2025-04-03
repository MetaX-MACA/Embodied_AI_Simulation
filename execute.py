import copy
import mujoco
import os
import time, datetime
import yaml

from manipulation.panda_env_base import PandaEnv
from manipulation import gpt_primitive_api
from manipulation.gpt_primitive_api import reset_panda_trajectories
from manipulation.utils import import_function_from_file, _PANDA_HOME
from manipulation.assets_joint_config import ASSET_JOINT_RANGE
from manipulation.rl_learn import RLTrainer

from stable_baselines3.common.vec_env import SubprocVecEnv


class Executer(object):
    def __init__(self, task_config_path:str=None, gui=False, n_envs=10):
        """
        task_config_path: str, 任务配置文件路径
        gui: bool, 是否可视化查看任务仿真过程, 这需要运行服务器有桌面环境以及显示器
        n_envs: int, 强化学习训练环境的数量, 受服务器内存限制, 64G内存最大支持10
        """
        self.gui = gui
        self.n_envs = n_envs
        if not self.gui:
            # OSMesa：一种离屏渲染库，使用 CPU 进行渲染，适合无图形界面的环境（如远程无图形界面的服务器）
            os.environ["MUJOCO_GL"] = "osmesa"
            # 若不生效，则在执行脚本前，手动设置以下环境变量
            # export MUJOCO_GL=osmesa
        
        with open(task_config_path, 'r') as f:
            task_config = yaml.safe_load(f)
        self.task_config = task_config
        
        # 初始化相关参数
        self.task_name = None
        self.task_description = None
        self.solution_path = None
        self.scene_xml_path = None
        self.asset_joint_list = []
        for obj in task_config:
            if "solution_path" in obj:
                self.solution_path = obj["solution_path"]
            if 'task_name' in obj:
                self.task_name = obj['task_name'].strip().replace(" ", "_")
                self.task_description = obj['task_description'].strip().replace("\n", " ")
            if "scene_xml_path" in obj:
                self.scene_xml_path = os.path.abspath(obj["scene_xml_path"]) # 转为绝对路径，否则mujoco加载资产会触发路径错误
        
        for obj in task_config:
            for key, value in obj.items():
                if "joint" in key:
                    self.asset_joint_list.append([key, get_joint_value(key, value)])
        
        substep_file_path = os.path.join(self.solution_path, "substeps.txt")
        with open(substep_file_path, 'r') as f:
            substeps = f.readlines()
        self.substeps = [item.strip().replace("\n", "").replace(" ", "_") for item in substeps]
            
        
        substep_types_file_path = os.path.join(self.solution_path, "substep_types.txt")
        with open(substep_types_file_path, 'r') as f:
            substep_types = f.readlines()
        self.substep_types = [item.strip().replace("\n", "") for item in substep_types]
        
        self.substep_config_files = []
        for index, substep_type in enumerate(self.substep_types):
            substep = self.substeps[index]
            if substep_type == "primitive":
                file_path = os.path.join(self.solution_path, f"{substep}.yaml")
                self.substep_config_files.append(file_path)
            elif substep_type == "reward":
                file_path = os.path.join(self.solution_path, f"{substep}.py")
                self.substep_config_files.append(file_path)
            else:
                raise ValueError("Invalid substep type")
        
        time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        video_path = os.path.join(self.solution_path, self.task_name + f"_{time_string}.mp4")
        
        # 创建仿真环境
        self.env = PandaEnv(xml_path=self.scene_xml_path, save_video=True, video_path=video_path, 
                            asset_joint_list=self.asset_joint_list)
        # 初始化环境
        self.env.reset()
        # 稳定环境
        for _ in range(10):     
            self.env.primitive_step(self.env.get_start_status()) 

        return

    def run(self):
        if self.gui:
            # 使用 GPU 进行渲染，适合有图形界面的环境
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=key_callback) as viewer:
                self._execute(viewer)
        else:
            self._execute()
        return
    
    def _execute(self, viewer=None):
        pre_substep_type = None
        for step_idx, (substep, substep_type, config_file) in enumerate(zip(self.substeps, self.substep_types, self.substep_config_files)):
            print("======================================================")
            print(f"step_idx: {step_idx+1}, substep: {substep}, substep_type: {substep_type}")
            # Motion planning
            if substep_type == "primitive":
                actions = self.execute_primitive(config_file)
                self._execute_primitive_actions(actions, viewer)
            
            # Reinforcement learning
            elif substep_type == "reward":
                if "knob" in substep:
                    print("Skip rotate knob") # because it often performs poorly
                    continue
                rl_model = self.execute_rl(substep, config_file)
                self._execute_reward_action(rl_model, viewer)
                
            else:
                raise ValueError("Invalid substep type")
            
            # 每次释放物体后，对机械臂进行复位。降低机械臂下一次运动规划的复杂度
            # 这是妥协做法，因为使用的机械臂资产没有可运动的底座，经过几次操作后容易出现关节锁死
            if "release" in substep:
                print("reset panda position")
                if pre_substep_type != "reward" and  "knob" not in substep:
                    self.reset_panda(viewer)
                else:
                    self.reset_panda_hard(viewer)

            pre_substep_type = substep_type
            
        self.env.close()
        return
    
    def execute_primitive(self, primitive_config_file_path):
        with open(primitive_config_file_path, 'r') as f:
            substep_config = yaml.safe_load(f)
        
        primitive_func_name = None
        waypoint_func_name = None
        gripper_closure = None
        for obj in substep_config:
            if "primitive_func_name" in obj:
                primitive_func_name = obj["primitive_func_name"]
            if 'waypoint_func_name' in obj:
                waypoint_func_name = obj['waypoint_func_name']
            if 'gripper_closure' in obj:
                gripper_closure = float(obj['gripper_closure'])

        primitive_func = getattr(gpt_primitive_api, primitive_func_name)
        if primitive_func_name == "approach_object_trajectories":
            trajectories = primitive_func(self.env, waypoint_func_name)
        else:
            trajectories = primitive_func(self.env)

        actions = []
        for tra in trajectories:
            tra.append(gripper_closure)
            actions.append(tra)

        return actions
    
    def _execute_primitive_actions(self, actions, viewer=None):
        """
        批量执行元动作
        """
        act_num = len(actions)
        max_act_num = act_num + 30
        i = 0
        act = None
        flag = True
        if viewer is not None:
            flag = viewer.is_running()
        
        while flag:
            step_start = time.time()
            if i < act_num:
                act = actions[i]
            self.env.primitive_step(act)
            self.env.render_frame(self.env.camera_names[0])
            
            if viewer is not None:
                viewer.sync()
                flag = viewer.is_running()
            time_until_next_step = self.env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            i += 1
            if i>= max_act_num:
                break
        return
    
    def execute_rl(self, sub_step_name, reward_function_file_path):
        reward_function = import_function_from_file(reward_function_file_path, "compute_reward")
        self.env.reward_func = reward_function
        self.env.reward_joint_name = reward_function(self.env)[-1]
        self.env.store_env_pos = copy.deepcopy(self.env.data.qpos)
        
        rl_model_dir = os.path.join(os.path.dirname(reward_function_file_path), "rl_checkpoints")
        os.makedirs(rl_model_dir, exist_ok=True)
        exp_name = sub_step_name
        
        best_model_path = os.path.join(rl_model_dir, exp_name, "best_model.zip")
        
        if not os.path.exists(best_model_path):
            envs = self.make_env()
            rl_trainer = RLTrainer(envs, exp_name, rl_model_dir)
            model = rl_trainer.train()
        else:
            model = RLTrainer.load_model(best_model_path)
        
        return model
    
    def make_env(self,):
        """
        创建多个环境，用于加速训练
        """
        print("Creating multiple environments for RL parallel training, this will take several minutes...")
        envs = []
        for _ in range(self.n_envs):
            env = PandaEnv(xml_path=self.scene_xml_path, asset_joint_list=self.asset_joint_list)
            env.reward_func = self.env.reward_func
            env.reward_joint_name = self.env.reward_joint_name
            env.store_env_pos = copy.deepcopy(self.env.store_env_pos)
            env.reset()
            envs.append(env)
        
        envs = SubprocVecEnv(envs)
        
        return envs
    
    def _execute_reward_action(self, model, viewer=None):
        obs, other_info = self.env.reset()
        terminated = False
        is_success = False
        flag = True
        if viewer is not None:
            flag = viewer.is_running()
            
        while flag and not terminated and not is_success:
            step_start = time.time()
            # predict
            action, _states = model.predict(obs, deterministic=True)
            # step
            obs, reward, is_success, terminated, info = self.env.step(action)
            
            # 打印奖励和状态信息
            print(f"Reward: {reward}, Is_success: {is_success}")
            
            self.env.render_frame(self.env.camera_names[0])
            
            if viewer is not None:
                viewer.sync()
                flag = viewer.is_running()    
            time_until_next_step = self.env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # reset
        self.env.reward_func = None
        self.env.reward_joint_name = None
        self.env.store_env_pos = None
    
    def reset_panda(self, viewer):
        """
        机械臂复位
        """
        trajectories = reset_panda_trajectories(self.env)
        gripper_closure = 0.0
        actions = []
        for tra in trajectories:
            tra.append(gripper_closure)
            actions.append(tra)
        self._execute_primitive_actions(actions, viewer)
        return
    
    def reset_panda_hard(self, viewer):
        """
        对机械臂强行复位
        原因:执行rl模型预测的操作、或操作旋钮后, 机械臂容易出现关节锁死现象, 使用运动规划无法复原
        """
        self.env._data.qpos[self.env._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self.env._model, self.env._data)
        
        # 稳定环境
        for _ in range(20):     
            self.env.primitive_step(self.env.get_start_status()) 
            self.env.render_frame(self.env.camera_names[0])
            if viewer is not None:
                viewer.sync()
        return 



def key_callback(keycode):
    if keycode == 32:
        global reset
        reset = True

def get_joint_value(joint_name, value):
    value = float(value)
    range = ASSET_JOINT_RANGE[joint_name]
    
    joint_limit_low, joint_limit_high = range[0], range[1]
    min_joint_angle = joint_limit_low
    max_joint_angle = joint_limit_high
    # Avoid joint_limit_low is a negative value
    if abs(joint_limit_low) > abs(joint_limit_high):
        min_joint_angle = joint_limit_high
        max_joint_angle = joint_limit_low
    
    joint_value = 0.0
    if value == 0.0:
        joint_value = min_joint_angle
    elif value == 1.0:
        joint_value = max_joint_angle
        # 控制微波炉门开启幅度，太大会导致固定机械臂够不到门把手
        if joint_name == "microdoorroot_joint":
            joint_value *= 0.65
    else:
        raise ValueError(f"Not support joint value, joint name: {joint_name}, value: {value}")

    return joint_value
        
def main(args):
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Not found file: {args.config}")
    executer = Executer(task_config_path=args.config, gui=args.gui, n_envs=args.envs)
    executer.run()
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="", 
                        help="path to task yaml file generated by run_kitchen.py")
    parser.add_argument('--gui', action="store_true", default=False, 
                        help="Default False, means using cpu to render image, because c500 not support rendering.")
    parser.add_argument('--envs', type=int, default=10, 
                        help="Number of environments for RL parallel training. The number is limited by the server's memory. A 64GB memory configuration can support a maximum of 10 envs.")
    args = parser.parse_args()
    
    main(args)
