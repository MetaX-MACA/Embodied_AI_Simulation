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
    parser = argparse.ArgumentParser(description="Double Deep Q-Network (Double DQN) for CartPole-v0")
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--total_timesteps", type=int, default=400000)
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
    return parser.parse_args()

# --------------------- Logging Setup --------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"double_dqn_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# --------------------- Q Network --------------------- #
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

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
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()
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
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim).to(args.device)
    target_net = QNetwork(obs_dim, act_dim).to(args.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
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
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, done)

        state = next_state if not done else env.reset()[0]

        # Train
        if len(buffer) > args.train_start and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            states = torch.tensor(batch.state, dtype=torch.float32).to(args.device)
            actions = torch.tensor(batch.action).unsqueeze(1).to(args.device)
            rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(args.device)
            next_states = torch.tensor(batch.next_state, dtype=torch.float32).to(args.device)
            dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(args.device)

            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                # Double DQN: Use q_net to select the best action, target_net to evaluate it
                next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = target_net(next_states).gather(1, next_actions)
                target_q = rewards + (1 - dones) * args.gamma * next_q_values
            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        training_time += time.time() - start_time

        # Target network update
        if total_steps % args.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Save and Evaluate
        if total_steps % args.save_interval == 0:
            save_name = f"double_dqn_{args.env_id}_step{total_steps}.pt"
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
    fps = (total_steps - step_pre) * args.batch_size  / training_time
    # Load best model for final eval
    load_model(q_net, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, q_net, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()