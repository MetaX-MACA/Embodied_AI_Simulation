import os
from copy import deepcopy

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

class RLTrainer(object):
    def __init__(self, envs, exp_name, model_dir, eval_freq=10000,):
        self.envs = envs
        self.exp_name = exp_name
        self.model_dir = model_dir
        self.eval_freq = eval_freq
    
        self.model = SAC(
            "MlpPolicy",
            self.envs,
            verbose=1,
            tensorboard_log=os.path.join(model_dir, exp_name, "tensorboard_log"),
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
        )

        # 设置 EvalCallback 以保存最佳模型
        self.eval_callback = EvalCallback(
            self.envs,
            best_model_save_path=os.path.join(model_dir, exp_name),  # 保存最佳模型的路径
            log_path=os.path.join(model_dir, exp_name, "eval_log"),  # 评估日志路径
            eval_freq=eval_freq,  # 每 10000 步评估一次
            deterministic=True,  # 使用确定性动作进行评估
            render=False,  # 是否渲染评估环境
            n_eval_episodes=5,  # 每次评估的 episode 数量
        )
        
        self.best_model_path = os.path.join(model_dir, exp_name, "best_model.zip")
        
        # 设置 CheckpointCallback 以定期保存模型
        self.checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=os.path.join(model_dir, exp_name, "checkpoints")
        )
        
        return
        
    def train(self):
        # 不存在则进行训练
        if not os.path.exists(self.best_model_path):
            self.model.learn(
                total_timesteps=10_000_000,
                callback=[self.eval_callback, self.checkpoint_callback],
                progress_bar=True,
            )
        
        model = SAC.load(self.best_model_path)
        
        return model
    
    @staticmethod
    def load_model(model_path):
        model = SAC.load(model_path)
        return model