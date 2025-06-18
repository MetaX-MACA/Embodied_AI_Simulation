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


# --------------------- Argument Parsing --------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Actor-Critic (AC) for CartPole-v0")
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--total_timesteps", type=int, default=400000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_interval", type=int, default=1000)
    return parser.parse_args()

# --------------------- Logging Setup --------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"ac_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# --------------------- Actor-Critic Network --------------------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

# --------------------- Evaluation --------------------- #
def evaluate_policy(env, model, device, episodes=5):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = model(state_tensor)
                action = torch.argmax(probs, dim=-1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# --------------------- Checkpoint Utils --------------------- #
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

    model = ActorCritic(obs_dim, act_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    state, _ = env.reset(seed=0)
    total_steps = 0
    training_time = 0
    best_return = -float("inf")
    save_dir = os.path.join("checkpoints", args.device)
    os.makedirs(save_dir, exist_ok=True)

    while total_steps < args.total_timesteps:
        start_time = time.time()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(args.device)
        probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(args.device)
        with torch.no_grad():
            _, next_value = model(next_state_tensor)

        target = reward + (1 - done) * args.gamma * next_value.item()
        advantage = target - value.item()

        policy_loss = -log_prob * advantage
        value_loss = nn.MSELoss()(value, torch.tensor([[target]], dtype=torch.float32).to(args.device))
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state if not done else env.reset()[0]
        total_steps += 1
        training_time += time.time() - start_time

        if total_steps % args.save_interval == 0:
            save_name = f"ac_{args.env_id}_step{total_steps}.pt"
            save_model(model, save_dir, save_name)
            manage_checkpoints(save_dir)
            mean_return, std_return = evaluate_policy(eval_env, model, args.device, args.eval_episodes)
            logger.info(f"Step {total_steps}: Eval Return = {mean_return:.2f} ± {std_return:.2f}")
            if mean_return > best_return:
                best_return = mean_return
                save_model(model, save_dir, "best.pt")
                logger.info(f"New best model saved with return {mean_return:.2f}")

    # ---------------- Final Summary ---------------- #
    fps = total_steps / training_time
    load_model(model, save_dir, "best.pt")
    best_mean, best_std = evaluate_policy(eval_env, model, args.device, args.eval_episodes)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
