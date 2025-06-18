# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import logging
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# --------------------- Argument Parsing --------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="C51 for CartPole-v0")
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=500)
    parser.add_argument("--target_update_freq", type=int, default=500)
    parser.add_argument("--train_start", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_atoms", type=int, default=101)
    parser.add_argument("--v_min", type=float, default=-100)
    parser.add_argument("--v_max", type=float, default=100)
    return parser.parse_args()

# --------------------- Logging Setup --------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"c51_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# --------------------- Q Network --------------------- #
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim * n_atoms),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, self.n, self.n_atoms)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        pmfs = torch.softmax(logits, dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action if action is not None else action]

# --------------------- Replay Buffer --------------------- #
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --------------------- Epsilon Greedy Policy --------------------- #
def get_epsilon(step, start, end, decay):
    return end + (start - end) * np.exp(-1.0 * step / decay)

# --------------------- Evaluation --------------------- #
def evaluate_policy(env, q_net, device, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = q_net.get_action(state_tensor)
                action = action.item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

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

# --------------------- Main Training --------------------- #
def main():
    args = parse_args()
    logger = setup_logging(args.device)

    # Set seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim, args.n_atoms, args.v_min, args.v_max).to(args.device)
    target_net = QNetwork(obs_dim, act_dim, args.n_atoms, args.v_min, args.v_max).to(args.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    buffer = ReplayBuffer(args.buffer_size)

    state, _ = env.reset(seed=0)
    total_steps = 0
    training_time = 0
    best_return = -float("inf")
    save_dir = os.path.join("checkpoints", args.device)
    os.makedirs(save_dir, exist_ok=True)

    while total_steps < args.total_timesteps:
        start_time = time.time()
        total_steps += 1
        epsilon = get_epsilon(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(args.device)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = q_net.get_action(state_tensor)
                action = action.item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, done)

        state = next_state if not done else env.reset(seed=0)[0]

        # Train
        if len(buffer) > args.train_start and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(args.device)
            actions = torch.tensor(batch.action).to(args.device).long()
            rewards = torch.tensor(batch.reward, dtype=torch.float32).to(args.device).unsqueeze(-1)
            next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(args.device)
            dones = torch.tensor(batch.done, dtype=torch.float32).to(args.device).unsqueeze(-1)

            with torch.no_grad():
                _, next_pmfs = target_net.get_action(next_states)
                next_atoms = rewards + args.gamma * target_net.atoms.unsqueeze(0) * (1 - dones)
                delta_z = target_net.atoms[1] - target_net.atoms[0]
                tz = next_atoms.clamp(args.v_min, args.v_max)
                b = (tz - args.v_min) / delta_z
                l = b.floor().clamp(0, args.n_atoms - 1)
                u = b.ceil().clamp(0, args.n_atoms - 1)
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros(args.batch_size, args.n_atoms).to(args.device)
                for i in range(args.batch_size):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            _, old_pmfs = q_net.get_action(states, actions)
            loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time += time.time() - start_time

        # Target network update
        if total_steps % args.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Save and Evaluate
        if total_steps % args.save_interval == 0:
            save_name = f"c51_{args.env_id}_step{total_steps}.pt"
            save_model(q_net, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(eval_env, q_net, args.device, args.eval_episodes)
            logger.info(f"Step {total_steps}: Eval Return = {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(q_net, save_dir, "best.pt")
                logger.info(f"New best model saved with return {mean_return:.2f}")

    # ---------------- Final Summary ---------------- #
    step_pre = max(args.train_start, args.batch_size)
    fps = (total_steps - step_pre) * args.batch_size / training_time if training_time > 0 else 0
    load_model(q_net, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, q_net, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()