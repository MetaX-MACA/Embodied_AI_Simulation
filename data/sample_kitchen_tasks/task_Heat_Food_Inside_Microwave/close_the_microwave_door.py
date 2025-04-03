
import numpy as np
from manipulation.gpt_reward_api import *

## substep: close_the_microwave_door

def compute_reward(env):
    # This reward encourages the end-effector to stay near door to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    door_pos = get_site_position(env._data, 'microhandle_site')
    reward_near = np.linalg.norm(eef_pos - door_pos)
    
    # Get the joint state of the door. We know from the semantics and the articulation tree that microdoorroot_joint connects microdoorroot and is the joint that controls the rotation of the door.
    reward_joint_name = "microdoorroot_joint"
    joint_angle = get_joint_state(env._data, reward_joint_name)
    
    # The reward is the negative distance between the current joint angle and the joint angle when the microwave door is fully close (low limit).
    joint_limit_low, joint_limit_high = get_joint_limit(env._model, reward_joint_name)
    min_joint_angle = joint_limit_low
    
    # Avoid joint_limit_low is a negative value.
    min_joint_angle = joint_limit_high if np.abs(joint_limit_low) > np.abs(joint_limit_high) else joint_limit_low
    
    reward_close = np.abs(joint_angle - min_joint_angle)
    
    reward = -reward_near - 5 * reward_close
    success = reward_close < 0.05 * np.abs(joint_limit_high - joint_limit_low)  # for closing door, we think 95 percent is enough
    
    return reward, success, reward_joint_name

