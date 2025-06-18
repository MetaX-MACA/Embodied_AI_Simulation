# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import os
import time
import logging
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 (REINFORCE) for HalfCheetah-v4")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

# ------------------------- Logging Setup ------------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"td3_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=-1)
        return self.q1(xu), self.q2(xu)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

# ------------------------- Evaluation ------------------------- #
def evaluate_policy(actor, eval_env, device, eval_episodes=5):
    returns = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_return = 0.0
        while not done:
            obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy().flatten()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_return += reward
        returns.append(total_return)
    return np.mean(returns), np.std(returns)

# ------------------------- Checkpoint Utils ------------------------- #
def save_model(agent, save_dir, filename):
    torch.save(agent.state_dict(), os.path.join(save_dir, filename))

def load_model(agent, save_dir, filename):
    agent.load_state_dict(torch.load(os.path.join(save_dir, filename)))

def manage_checkpoints(save_dir, max_files=10):
    ckpts = sorted([f for f in os.listdir(save_dir) if f.endswith(".pt") and f != "best.pt"],
                   key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
    while len(ckpts) > max_files:
        os.remove(os.path.join(save_dir, ckpts.pop(0)))

def train_td3(args):
    logger = setup_logging(args.device)
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, max_action).to(args.device)
    actor_target = Actor(obs_dim, act_dim, max_action).to(args.device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)

    critic = Critic(obs_dim, act_dim).to(args.device)
    critic_target = Critic(obs_dim, act_dim).to(args.device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    obs, _ = env.reset(seed=args.seed)
    
    total_time = 0
    best_return = float('-inf')
    save_dir = os.path.join("checkpoints", args.device)
    os.makedirs(save_dir, exist_ok=True)

    for t in range(args.total_timesteps):
        start_time = time.time()

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(args.device)
            action = actor(obs_tensor).cpu().data.numpy().flatten()
            action += np.random.normal(0, args.expl_noise, size=act_dim)
            action = np.clip(action, -max_action, max_action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, done)

        obs = next_obs

        if done:
            obs, _ = env.reset()

        if t >= args.start_timesteps:
            batch = replay_buffer.sample_batch(args.batch_size)
            obs_b = torch.FloatTensor(batch['obs']).to(args.device)
            act_b = torch.FloatTensor(batch['acts']).to(args.device)
            rew_b = torch.FloatTensor(batch['rews']).to(args.device)
            obs2_b = torch.FloatTensor(batch['obs2']).to(args.device)
            done_b = torch.FloatTensor(batch['done']).to(args.device)

            with torch.no_grad():
                noise = (torch.randn_like(act_b) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                next_action = (actor_target(obs2_b) + noise).clamp(-max_action, max_action)
                target_Q1, target_Q2 = critic_target(obs2_b, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rew_b + (1 - done_b) * args.gamma * target_Q

            current_Q1, current_Q2 = critic(obs_b, act_b)
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            
            if t % args.policy_freq == 0:
                # actor_loss = -critic.q1(obs_b, actor(obs_b)).mean()
                actor_loss = -critic(obs_b, actor(obs_b))[0].mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                with torch.no_grad():
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            total_time += time.time() -start_time
        if t % args.save_interval == 0:
            save_name = f"td3_{args.env_id}_step{t}.pt"
            save_model(actor, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(actor, eval_env,args.device, args.eval_episodes)
            logger.info(f"Update {t:03d}: Eval mean return = {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(actor, save_dir, "best.pt")
                logger.info(f"New best model saved with return {mean_return:.2f}")
            
    # ---------------- Final Summary ---------------- #
    fps = (t - args.start_timesteps) * args.batch_size / total_time
    # Load best model for final eval
    load_model(actor, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(actor, eval_env, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {t}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_td3(args)
