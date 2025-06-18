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

    td3 = d3rlpy.algos.TD3PlusBCConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        alpha=2.5,
        update_actor_interval=2,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
    ).create(device=args.device)

    total_epochs = 500000 // 1000
    td3.fit(
        dataset,
        # n_steps=500000,
        n_steps=2000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"{args.device}/TD3PlusBC_{args.dataset}_{args.seed}",
        epoch_callback=lambda algo, epoch, total_step: epoch_callback(algo, epoch, total_step, total_epochs, env, logger),
    )


if __name__ == "__main__":
    main()