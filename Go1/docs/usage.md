# Usage
Please refer to [install.md](install.md) for environment setup.


## Overview
The pipeline of this project consists of two main modules:  
1. Training go1 robot to backflip in MuJoCo using multi-objective RL.
2. Testing learned skills in MuJoCo.

## Support Tasks
- GO1 Robot (Quadruped from Unitree)
    - Back-Flip

## Training and Evaluation
For simplicity, we only train and test the teacher model.

### Teacher Learning
cd ./Stage-Wise-CMORL/
- training: `python main_teacher_mujoco.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --wandb --seed 1`
- test: `python main_teacher_mujoco.py --task_cfg_path tasks/{task_name}.yaml --algo_cfg_path algos/comoppo/{task_name}.yaml --test --render --seed 1 --model_num {saved_model_num}`

### Actual Running
cd ./Stage-Wise-CMORL/
- training: `python main_teacher_mujoco.py --task_cfg_path tasks/go1_backflip_mujoco.yaml --algo_cfg_path algos/comoppo/go1_backflip_mujoco.yaml --wandb --seed 1`
- test: `python main_teacher_mujoco.py --task_cfg_path tasks/go1_backflip_mujoco.yaml --algo_cfg_path algos/comoppo/go1_backflip_mujoco.yaml --test --render --seed 1 --model_num {saved_model_num}`

### Trained models
Pretrained models are provided in the /result folder.
``` shell
# Execute for MuJoCo rendering (outputs to ./output_videos)
cp -r ./results ./Stage-Wise-CMORL/
cd ./Stage-Wise-CMORL/
python main_teacher_mujoco.py --task_cfg_path tasks/go1_backflip_mujoco.yaml --algo_cfg_path algos/comoppo/go1_backflip_mujoco.yaml --test --render --seed 1 --model_num 2400000
```


### Further Work
Our future developments will incorporate more complex skills including robotic(Go1 & H1 platforms) backflips , side-flips in Mujoco simulation.