# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import os
import time
import logging
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Normal
from typing import Tuple

# ------------------------- Args ------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="SAX (REINFORCE) for HalfCheetah-v4")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=3000000)
    parser.add_argument("--start_steps", type=int, default=25000)
    parser.add_argument("--update_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

# ------------------------- Logging ------------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"sac_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# ------------------------- Utils ------------------------- #
def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    act=self.acts_buf[idxs],
                    rew=self.rews_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    done=self.done_buf[idxs])
        
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
        
# ------------------------- Networks ------------------------- #
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256], activation=nn.ReLU)
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -2, 0

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist

    def sample(self, obs):
        dist = self(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, 256, 256, 1])
        self.q2 = mlp([obs_dim + act_dim, 256, 256, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

# ------------------------- SAC Training ------------------------- #
def evaluate_policy(env, actor, device, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_ret = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _ = actor.sample(obs_tensor.unsqueeze(0))
            action = action.cpu().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_ret += reward
        returns.append(total_ret)
    return np.mean(returns), np.std(returns)

def main():
    args = parse_args()
    logger = setup_logging(args.device)
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_opt = optim.Adam(critic.parameters(), lr=args.learning_rate)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)

    obs, _ = env.reset()
    episode_reward = 0
    best_return = float('-inf')
    total_steps = 0
    training_time = 0
    save_dir = os.path.join("checkpoints", device)
    os.makedirs(save_dir, exist_ok=True)

    while total_steps < args.total_timesteps:
        start_time = time.time()
        if total_steps < args.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                action, _ = actor.sample(obs_tensor.unsqueeze(0))
                action = action.cpu().numpy()[0]
                action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward
        total_steps += 1

        if done:
            obs, _ = env.reset()
            episode_reward = 0

        if total_steps >= args.start_steps and total_steps % args.update_every == 0:
            for _ in range(args.update_every // 10):
                batch = replay_buffer.sample_batch(args.batch_size)
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32).to(device)
                act_b = torch.tensor(batch["act"], dtype=torch.float32).to(device)
                rew_b = torch.tensor(batch["rew"], dtype=torch.float32).to(device).unsqueeze(-1)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32).to(device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32).to(device).unsqueeze(-1)

                with torch.no_grad():
                    next_action, log_prob = actor.sample(next_obs_b)
                    next_q1, next_q2 = target_critic(next_obs_b, next_action)
                    next_q = torch.min(next_q1, next_q2) - args.alpha * log_prob.unsqueeze(-1)
                    target_q = rew_b + args.gamma * (1 - done_b) * next_q

                q1, q2 = critic(obs_b, act_b)
                critic_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                pi, logp_pi = actor.sample(obs_b)
                q1_pi, q2_pi = critic(obs_b, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (args.alpha * logp_pi.unsqueeze(-1) - min_q_pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(critic, target_critic, args.tau)
        training_time += time.time() - start_time
        if total_steps % args.save_interval == 0:
            save_name = f"sac_{args.env_id}_step{total_steps}.pt"
            save_model(actor, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(eval_env, actor, device, args.eval_episodes)
            logger.info(f"Update {total_steps:03d}: Eval mean return = {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(actor, save_dir, "best.pt")
                logger.info(f"New best model saved with return {mean_return:.2f}")

    # ---------------- Final Summary ---------------- #
    fps = total_steps / training_time
    # Load best model for final eval
    load_model(actor, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, actor, device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()