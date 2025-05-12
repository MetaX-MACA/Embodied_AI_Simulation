import numpy as np

def get_eef_pos_and_quat(env_data):
    pos = env_data.sensor("2f85/pinch_pos").data
    quat = env_data.sensor("2f85/pinch_quat").data
    return pos, quat

def get_joint_id_from_name(env_model, joint_name):
    return env_model.joint(joint_name).id

def get_site_id_from_name(env_model, site_name):
    return env_model.site(site_name).id

def get_joint_state(env_data, joint_name):
    joint_value = env_data.joint(joint_name).qpos[0]
    return joint_value

def get_joint_limit(env_model, joint_name):
    joint_limit_low, joint_limit_high = env_model.joint(joint_name).range
    return joint_limit_low, joint_limit_high

def get_site_position(env_data, site_name):
    site_pos = env_data.site(site_name).xpos
    return site_pos