# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import logging
import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# --------------------- Argument Parsing --------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Policy Gradient (REINFORCE) for CartPole-v0")
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--total_timesteps", type=int, default=40000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--episode_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_interval", type=int, default=100)
    return parser.parse_args()

# --------------------- Logging Setup --------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"pg_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# --------------------- Policy Network --------------------- #
class PolicyNetwork(nn.Module):
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
        logits = self.net(x)
        return logits

    def get_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

# --------------------- Evaluation --------------------- #
def evaluate_policy(env, policy_net, device, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _ = policy_net.get_action(state_tensor)
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

    policy_net = PolicyNetwork(obs_dim, act_dim).to(args.device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    total_steps = 0
    training_time = 0
    best_return = -float("inf")
    save_dir = os.path.join("checkpoints", args.device)
    os.makedirs(save_dir, exist_ok=True)

    while total_steps < args.total_timesteps:
        start_time = time.time()
        total_steps += 1

        # Collect trajectory
        state, _ = env.reset(seed=0)
        log_probs = []
        rewards = []
        episode_steps = 0
        done = False

        while not done and episode_steps < args.episode_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(args.device)
            action, log_prob = policy_net.get_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            episode_steps += 1
            
        # Compute discounted rewards (returns)
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(args.device)

        # Normalize returns (optional, for stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        training_time += time.time() - start_time

        # Save and Evaluate
        if total_steps % args.save_interval == 0:
            save_name = f"pg_{args.env_id}_step{total_steps}.pt"
            save_model(policy_net, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(eval_env, policy_net, args.device, args.eval_episodes)
            logger.info(f"Step {total_steps}: Eval Return = {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(policy_net, save_dir, "best.pt")
                logger.info(f"New best model saved with return {mean_return:.2f}")

    # ---------------- Final Summary ---------------- #
    fps = total_steps / training_time
    # Load best model for final eval
    load_model(policy_net, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, policy_net, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()