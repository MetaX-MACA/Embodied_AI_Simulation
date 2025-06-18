# Copyright 2025 MetaX Integrated Circuits (Shanghai) Co.,Ltd.
import argparse
import logging
import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# --------------------- Argument Parsing --------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-threaded A3C for CartPole-v0")
    parser.add_argument("--env_id", type=str, default="CartPole-v0")
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--update_global_iter", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy_beta", type=float, default=0.0005)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

# --------------------- Logging Setup --------------------- #
def setup_logging(device):
    log_file = f"a3c_training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

# --------------------- A3C Network --------------------- #
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, 2)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x)
        return logits, value

    def choose_action(self, s):
        self.eval()
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
        logits, _ = self.forward(s)
        logits = torch.clamp(logits, -20, 20)
        prob = F.softmax(logits, dim=1).cpu().detach().numpy().flatten()
        if np.any(np.isnan(prob)) or np.sum(prob) == 0:
            prob = np.ones_like(prob) / len(prob)
        device = next(self.parameters()).device
        return np.random.choice(len(prob), p=prob), device

    def loss_func(self, s, a, v_t, entropy_beta):
        self.train()
        device = next(self.parameters()).device
        s, a, v_t = s.to(device), a.to(device), v_t.to(device) 
        logits, values = self.forward(s)
        td = v_t - values.squeeze()
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        a_log_probs = log_probs[range(len(a)), a]
        a_loss = -a_log_probs * td.detach()
        total_loss = (c_loss + a_loss - entropy_beta * entropy).mean()
        return total_loss

# --------------------- Shared Optimizer --------------------- #
class SharedRMSProp(torch.optim.RMSprop):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['square_avg'] = torch.zeros_like(p.data)
                state['momentum_buffer'] = torch.zeros_like(p.data)
                p.share_memory_()

# --------------------- Worker Process --------------------- #
class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, args, name, device):
        super().__init__()
        self.name = f"worker_{name}"
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.args = args
        self.lnet = Net().to(device)
        self.env = gym.make(args.env_id)

    def run(self):
        total_step = 1
        while self.g_ep.value < self.args.max_episodes:
            s, _ = self.env.reset()
            s = np.clip(s, -5, 5)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                a, device = self.lnet.choose_action(s)
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                s_ = np.clip(s_, -5, 5)
                ep_r += r

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % self.args.update_global_iter == 0 or done:
                    v_s_ = 0 if done else self.lnet.forward(torch.tensor(s_, dtype=torch.float32).unsqueeze(0).to(device))[1].item()

                    buffer_v_target = []
                    for rwd in buffer_r[::-1]:
                        v_s_ = rwd + self.args.gamma * v_s_
                        buffer_v_target.insert(0, v_s_)

                    s_batch = torch.tensor(buffer_s, dtype=torch.float32)
                    a_batch = torch.tensor(buffer_a, dtype=torch.int64)
                    v_batch = torch.tensor(buffer_v_target, dtype=torch.float32)

                    loss = self.lnet.loss_func(s_batch, a_batch, v_batch, self.args.entropy_beta)
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.lnet.parameters(), self.args.max_grad_norm)

                    # push local gradients to global network
                    for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                        if gp.grad is None:
                            gp.grad = lp.grad.clone()
                        else:
                            gp.grad += lp.grad

                    self.opt.step()
                    self.lnet.load_state_dict(self.gnet.state_dict())

                    buffer_s, buffer_a, buffer_r = [], [], []
                s = s_
                total_step += 1
                if done:
                    with self.g_ep.get_lock():
                        self.g_ep.value += 1
                    with self.g_ep_r.get_lock():
                        if self.g_ep_r.value == 0.:
                            self.g_ep_r.value = ep_r
                        else:
                            self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_r * 0.01
                    self.res_queue.put(self.g_ep_r.value)
                    break

# --------------------- Evaluation --------------------- #
def evaluate_policy(net, env_id, device, episodes=5):
    env = gym.make(env_id)
    rewards = []
    net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            s, _ = env.reset()
            ep_r = 0
            done = False
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = net(s_tensor)
                prob = F.softmax(logits, dim=1)
                a = prob.argmax(dim=1).item()
                s, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                ep_r += r
            rewards.append(ep_r)
    net.train()
    return np.mean(rewards), np.std(rewards)

# --------------------- Checkpoint Utils --------------------- #
def save_model(net, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(save_dir, filename))

def load_model(net, save_dir, filename):
    net.load_state_dict(torch.load(os.path.join(save_dir, filename)))

def manage_checkpoints(save_dir, max_files=10):
    ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.endswith(".pth") and f != "best.pth"],
        key=lambda x: os.path.getmtime(os.path.join(save_dir, x)),
    )
    while len(ckpts) > max_files:
        os.remove(os.path.join(save_dir, ckpts.pop(0)))

# --------------------- Main Training --------------------- #
def main():
    args = parse_args()
    device = args.device
    logger = setup_logging(device)
    

    gnet = Net().to(device)
    gnet.share_memory()
    opt = SharedRMSProp(gnet.parameters(), lr=args.lr)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, args, i, device) for i in range(args.num_workers)]
    [w.start() for w in workers]

    rewards = []
    best_mean_return = -float('inf')
    
    save_dir = os.path.join("checkpoints", device)
    os.makedirs(save_dir, exist_ok=True)
    training_time = 0
    while True:
        start_time = time.time()
        r = res_queue.get()
        if r is not None:
            rewards.append(r)
            mean_last_100 = np.mean(rewards[-100:])
            logger.info(f"Episode: {global_ep.value}, Avg Reward Last 100: {mean_last_100:.2f}")
            # Save checkpoint and eval
            if global_ep.value % args.save_interval == 0:
                save_name = f"a3c_{args.env_id}_step{global_ep}.pt"
                save_model(gnet, save_dir, f"checkpoint_ep{global_ep.value}.pth")
                manage_checkpoints(save_dir)
                mean_return, std_return = evaluate_policy(gnet, args.env_id, device, episodes=5)
                logger.info(f"Eval after {global_ep.value} episodes: Mean Return = {mean_return:.2f} ± {std_return:.2f}")
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    save_model(gnet, save_dir, "best.pth")
                    logger.info(f"New best model saved with mean return {best_mean_return:.2f}")

            if global_ep.value >= args.max_episodes:
                logger.info("Training finished.")
                for w in workers:
                    w.terminate()
                break
    # ---------------- Final Summary ---------------- #
    training_time = time.time() - start_time
    fps = global_ep.value  / training_time
    # Load best model for final eval
    load_model(gnet, save_dir, "best.pth")
    best_mean, best_std = evaluate_policy(gnet, args.env_id, device, episodes=5)
    logger.info(f"--- Training Summary ({args.device}) ---")
    logger.info(f"Total steps: {global_ep.value}")
    logger.info(f"Total time: {training_time:.2f}s")
    logger.info(f"Best Eval Mean Return: {best_mean:.2f}")
    logger.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
