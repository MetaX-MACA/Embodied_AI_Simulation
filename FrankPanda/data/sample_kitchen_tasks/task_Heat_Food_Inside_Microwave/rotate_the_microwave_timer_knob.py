
import numpy as np
from manipulation.gpt_reward_api import *

## substep: rotate_the_microwave_timer_knob

def compute_reward(env):
    # this reward encourages the end-effector to stay near knob to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    knob_pos = get_site_position(env._data, 'micro_timer_knob_site')
    reward_near = np.linalg.norm(eef_pos - knob_pos)
    
    # Get the joint state of the knob. We know from the semantics and the articulation tree that micro_timer_joint connects micro_timer_knob and is the joint that controls the rotation of the knob.
    reward_joint_name = "micro_timer_joint"
    joint_angle = get_joint_state(env._data, reward_joint_name) 
    
    # The reward is the negative distance between the current joint angle and the joint angle when the knob is fully rotated (upper limit).
    joint_limit_low, joint_limit_high = get_joint_limit(env._model, reward_joint_name)
    max_joint_angle = joint_limit_high
    
    # Avoid joint_limit_low is a negative value.
    max_joint_angle = joint_limit_low if np.abs(joint_limit_low) > np.abs(joint_limit_high) else joint_limit_high
    
    reward_angle = np.abs(joint_angle - max_joint_angle)
    
    reward = -reward_near - 10 * reward_angle
    
    success = reward_angle < 0.5 * np.abs(joint_limit_high - joint_limit_low) # for rotating knob, we think 50 percent is enough
    
    return reward, success, reward_joint_name

