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


# ----------------- Argument Parsing ----------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="DDPG for HalfCheetah-v4")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_noise", type=float, default=0.1)
    parser.add_argument("--exploration_noise", type=float, default=0.05)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

# ----------------- Logging ----------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"ddpg_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# ----------------- Replay Buffer ----------------- #
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.acts[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.rews[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[idx], dtype=torch.float32).to(self.device)
        )

# ----------------- Model Definitions ----------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 400)), nn.ReLU(),
            layer_init(nn.Linear(400, 300)), nn.ReLU(),
            layer_init(nn.Linear(300, act_dim), std=0.01), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, 400)), nn.ReLU(),
            layer_init(nn.Linear(400, 300)), nn.ReLU(),
            layer_init(nn.Linear(300, 1))
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))

# ----------------- Evaluation ----------------- #
def evaluate_policy(env, actor, device, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=1)
        done, total_return = False, 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
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

# ----------------- Training ----------------- #
def main():
    args = parse_args()
    logger = setup_logging(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    max_action = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, max_action).to(args.device)
    actor_target = Actor(obs_dim, act_dim, max_action).to(args.device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(obs_dim, act_dim).to(args.device)
    critic_target = Critic(obs_dim, act_dim).to(args.device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate)

    replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, args.device)

    obs, _ = env.reset(seed=args.seed)
    total_steps = 0
    best_return = -float("inf")
    save_dir = os.path.join("checkpoints", args.device)
    os.makedirs(save_dir, exist_ok=True)
    total_time = 0

    while total_steps < args.total_timesteps:
        start_time = time.time()
        if total_steps < args.start_steps:
            action = env.action_space.sample()
        else:
            action = actor(torch.tensor(obs, dtype=torch.float32).to(args.device).unsqueeze(0)).cpu().data.numpy().flatten()
            noise = np.random.normal(0, args.exploration_noise, size=act_dim)
            action += noise
            action = np.clip(action, -max_action, max_action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        total_steps += 1

        if done:
            obs, _ = env.reset()

        if replay_buffer.size >= args.batch_size:
            for _ in range(1):
                obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(args.batch_size)
                # target Q
                with torch.no_grad():
                    target_act = actor_target(next_obs_b)
                    target_Q = critic_target(next_obs_b, target_act)
                    target_Q = rew_b + args.gamma * (1 - done_b) * target_Q

                # update critic
                current_Q = critic(obs_b, act_b)
                critic_loss = nn.MSELoss()(current_Q, target_Q)
                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optim.step()

                # update actor
                actor_loss = -critic(obs_b, actor(obs_b)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_optim.step()

                # soft update
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        total_time += time.time() - start_time
        
        if total_steps % args.save_interval == 0:
            save_name = f"ddpg_{args.env_id}_step{total_steps}.pt"
            save_model(actor, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(eval_env, actor, args.device, args.eval_episodes)
            logger.info(f"[Step {total_steps}] Eval return: {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(actor, save_dir, "best.pt")
                logger.info("New best model saved!")
    
    # ---------------- Final Summary ---------------- #
    fps = (total_steps - args.batch_size) * args.batch_size / total_time if total_time > 0 else 0
    load_model(actor, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, actor, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}, {best_std}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()