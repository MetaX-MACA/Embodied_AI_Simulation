# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import argparse
import numpy as np
import logging
import d3rlpy

# --------------------- Logging Setup --------------------- #
def setup_logging(device: str) -> logging.Logger:
    log_file = f"training_{device}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.info(f"Using device: {device}")
    return logger

def evaluate_policy(algo, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            obs_batch = np.array([obs])
            action = algo.predict(obs_batch)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns), np.std(returns)

def epoch_callback(algo, epoch, total_step, total_epochs, env, logger=None):
    mean_return, std_return = evaluate_policy(algo, env)
    if logger:
        logger.info(f"Epoch {epoch}/{total_epochs}: Step {total_step}: Mean return = {mean_return:.2f} ± {std_return:.2f}")
    else:
        print(f"Epoch {epoch}/{total_epochs}: Step {total_step}: Mean return = {mean_return:.2f} ± {std_return:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    logger = setup_logging(args.device)

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
    ).create(device=args.device)

    total_epochs = 500000 // 1000
    cql.fit(
        dataset,
        # n_steps=500000,
        n_steps=2000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"{args.device}/CQL_{args.dataset}_{args.seed}",
        epoch_callback=lambda algo, epoch, total_step: epoch_callback(algo, epoch, total_step, total_epochs, env, logger),
    )

if __name__ == "__main__":
    main()