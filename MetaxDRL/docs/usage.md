# Usage
Please refer to [install.md](install.md) for environment setup.


## Overview
The pipeline of this project consists of two main modules:  
1. Training and Testing the online Reinforcement Learning.
2. Training and Testing the offline Reinforcement Learning.

## Support Tasks
- online Reinforcement Learning
    - a2c
    - a3c
    - ac
    - c51
    - ddpg
    - double_dqn
    - dqn
    - dueling_dqn
    - policy_gradient
    - ppo
    - sac
    - td3

- offline Reinforcement Learning
    - awac
    - bcq
    - bear
    - cql
    - crr
    - decision_trans
    - iql
    - plas
    - td3
    - td3_plus_bc

### Actual Running
``` shell
# chose online a2c for example
cd ./example/online/a2c/
# training on C500 or GPU
python train.py --device cuda
# training on CPU
python train.py --device cpu
# Checkpoints will be saved automatically, and the log file will record the entire training and evaluation process.
```

### Further Work
In the future, we plan to extend support to a broader range of reinforcement learning algorithms.