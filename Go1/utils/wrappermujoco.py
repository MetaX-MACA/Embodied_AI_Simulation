import torch
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv


class EnvWrapperMujoco:
    def __init__(self, env_fn, n_envs):
        
        self._env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_envs = n_envs
        self.obs_dim = self._env.observation_space["obs"].shape[0]
        self.state_dim = self._env.get_attr("state_space")[0].shape[0]
    
    def reset(self, **kwargs):
        obs_dict = self._env.reset(**kwargs)
        obs = torch.as_tensor(obs_dict['obs'], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(obs_dict['states'], dtype=torch.float32, device=self.device)
        return obs, state
    
    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        obs = torch.tensor(obs_dict['obs'], device=self.device)
        state = torch.tensor(obs_dict['states'], device=self.device)
        merged_dict = {}
        for key in info[0].keys():
            if isinstance(info[0][key], torch.Tensor):
                merged_dict[key] = torch.cat([d[key] for d in info], dim=0)
            else:
                merged_dict[key] = info[0][key]
    
        return obs, state, reward, done, merged_dict

    def close(self):
        self._env.close()
    
    @property
    def unwrapped(self):
        return self._env