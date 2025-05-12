
import numpy as np
from manipulation.gpt_reward_api import *

## substep: open_the_slide_cabinet_door

def compute_reward(env):
    # This reward encourages the end-effector to stay near door to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    door_pos = get_site_position(env._data, 'slidehandle_site')
    reward_near = np.linalg.norm(eef_pos - door_pos)
    
    # Get the joint state of the door. We know from the semantics and the articulation tree that slidedoor_joint connects slidedoor and is the joint that controls the rotation of the door.
    reward_joint_name = "slidedoor_joint"
    joint_angle = get_joint_state(env._data, reward_joint_name)
    
    # The reward is the negative distance between the current joint angle and the joint angle when the slide cabinet door is fully open (high limit).
    joint_limit_low, joint_limit_high = get_joint_limit(env._model, reward_joint_name)
    max_joint_angle = joint_limit_high
    
    # Avoid joint_limit_low is a negative value.
    max_joint_angle = joint_limit_low if np.abs(joint_limit_low) > np.abs(joint_limit_high) else joint_limit_high
    
    reward_open = np.abs(joint_angle - max_joint_angle)
    
    reward = -reward_near - 5 * reward_open
    success = reward_open < 0.25 * np.abs(joint_limit_high - joint_limit_low)  # for opening door, we think 75 percent is enough
    
    return reward, success, reward_joint_name

