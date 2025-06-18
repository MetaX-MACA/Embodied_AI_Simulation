# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import os
import time
import random
from collections import deque
import logging
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------- Args ------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v4')
    parser.add_argument('--total_steps', type=int, default=10000000)
    parser.add_argument('--steps_per_epoch', type=int, default=2048)
    parser.add_argument('--update_iters', type=int, default=10)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--eval_interval', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

# ------------------------- Logging ------------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"ppo_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

# ------------------------- Networks ------------------------- #
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp([obs_dim, 64, 64, act_dim * 2])
        self.act_limit = act_limit

    def forward(self, obs):
        mu_log_std = self.net(obs)
        mu, log_std = torch.chunk(mu_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def act(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        return torch.tanh(action) * self.act_limit


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v_net = mlp([obs_dim, 64, 64, 1])

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)

# ------------------------- Utils ------------------------- #
class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return [
            torch.as_tensor(self.obs_buf, dtype=torch.float32),
            torch.as_tensor(self.act_buf, dtype=torch.float32),
            torch.as_tensor(self.adv_buf, dtype=torch.float32),
            torch.as_tensor(self.ret_buf, dtype=torch.float32),
            torch.as_tensor(self.logp_buf, dtype=torch.float32),
        ]

    @staticmethod
    def discount_cumsum(x, discount):
        return np.array([sum(discount ** i * x[j] for i, j in enumerate(range(t, len(x)))) for t in range(len(x))])

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


def compute_log_probs(dist, action):
    logp = dist.log_prob(action).sum(axis=-1)
    logp -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)  # Tanh correction
    return logp


def evaluate_policy(env, actor, device, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_ret = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = actor.act(obs_tensor.unsqueeze(0), deterministic=True).cpu().numpy().squeeze()
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
    act_limit = env.action_space.high[0]

    
    actor = Actor(obs_dim, act_dim, act_limit).to(device)
    critic = Critic(obs_dim).to(device)
    pi_optimizer = optim.Adam(actor.parameters(), lr=args.pi_lr)
    vf_optimizer = optim.Adam(critic.parameters(), lr=args.vf_lr)

    buffer = PPOBuffer(obs_dim, act_dim, args.steps_per_epoch, args.gamma, args.lam)

    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0
    
    best_return = float('-inf')
    training_time = 0
    used_step = 0
    save_dir = os.path.join("checkpoints", device)
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    for t in range(1, args.total_steps + 1):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
        dist = actor(obs_tensor.unsqueeze(0))
        action = dist.rsample()
        logp = compute_log_probs(dist, action).cpu().item()
        clipped_action = torch.tanh(action).detach().cpu().numpy().squeeze() * act_limit

        value = critic(obs_tensor).cpu().item()

        next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated
        ep_ret += reward
        ep_len += 1

        # buffer.store(obs, action.cpu().numpy().squeeze(), reward, value, logp)
        buffer.store(obs, action.detach().cpu().numpy().squeeze(), reward, value, logp)

        obs = next_obs

        if done or t % args.steps_per_epoch == 0:
            if not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                last_val = critic(obs_tensor).cpu().item()
            else:
                last_val = 0
            buffer.finish_path(last_val)
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0
        if t % args.steps_per_epoch == 0:
            used_step += 1
            data = buffer.get()
            obs_b, act_b, adv_b, ret_b, logp_old_b = data

            for _ in range(args.update_iters):
                dist = actor(obs_b.to(device))
                logp = compute_log_probs(dist, act_b.to(device))
                ratio = torch.exp(logp - logp_old_b.to(device))
                clip_adv = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv_b.to(device)
                pi_loss = -(torch.min(ratio * adv_b.to(device), clip_adv)).mean()

                pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_optimizer.step()

                value = critic(obs_b.to(device))
                v_loss = F.mse_loss(value, ret_b.to(device))

                vf_optimizer.zero_grad()
                v_loss.backward()
                vf_optimizer.step()
            
            
            if t % args.eval_interval == 0:
                save_name = f"sac_{args.env_id}_step{t}.pt"
                save_model(actor, save_dir, save_name)
                manage_checkpoints(save_dir)
                mean_return, std_return = evaluate_policy(eval_env, actor, device)
                logger.info(f"Update {t:03d}: Eval mean return = {mean_return:.2f} ± {std_return:.2f}")
                if mean_return > best_return:
                    best_return = mean_return
                    save_model(actor, save_dir, "best.pt")
                    logger.info(f"New best model saved with return {mean_return:.2f}")

    # ---------------- Final Summary ---------------- #
    training_time += time.time() - start_time
    fps = used_step * args.steps_per_epoch * args.update_iters / training_time
    # Load best model for final eval
    load_model(actor, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, actor, device)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {used_step * args.steps_per_epoch * args.update_iters}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")
    
if __name__ == '__main__':
    main()
