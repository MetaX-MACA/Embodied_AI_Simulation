import numpy as np
from copy import deepcopy

from gpt_reward_api import get_eef_pos_and_quat
from utils import get_site_quaternion


def move_to_mug_in_slidecabinet(env):
    end_pos = env._data.sensor("mug_center_pos").data

    inner_behind_pos = env._data.site("slide_inner_behind_site").xpos
    inner_pos = env._data.site("slide_inner_site").xpos
    quat, _ = get_site_quaternion(inner_behind_pos, inner_pos)
    
    waypoints = [[end_pos, quat],]
    return waypoints

def put_mug_in_microwave(env):
    start_pos, start_quat = get_eef_pos_and_quat(env._data)

    # 先向上抬起，避免底部摩擦
    gap = 0.2
    start_pos[-1] += gap
    waypoints = [[start_pos, start_quat],]

    inner_behind_pos = env._data.site("micro_inner_behind_site").xpos
    end_pos = env._data.site("micro_inner_site").xpos
    quat, direction_vector = get_site_quaternion(inner_behind_pos, end_pos)
    # 沿着法向量增加一个小偏移，作为附近的点
    near_pos = deepcopy(end_pos)
    gap = 0.01
    near_pos = np.array(near_pos) + -1 * np.array(direction_vector) * gap
    waypoints.append([near_pos, quat])

    # 慢慢放下，避免摔倒
    gap = 0.02
    end_pos2 = deepcopy(near_pos)
    end_pos2[-1] -= gap
    waypoints.append([end_pos2, quat])

    return waypoints

def retrieve_the_mug_from_slidecabinet(env):
    start_pos, start_quat = get_eef_pos_and_quat(env._data)

    # 先向上抬起，避免底部摩擦
    gap = 0.2
    start_pos[-1] += gap
    waypoints = [[start_pos, start_quat],]

    end_pos = env._data.site("table_place_site").xpos
    quat = deepcopy(start_quat)
    waypoints.append([end_pos, quat])

    # 慢慢放下，避免摔倒
    # gap = 0.02
    # end_pos2 = deepcopy(end_pos)
    # end_pos2[-1] -= gap
    # waypoints.append([end_pos2, quat])
    return waypoints

def move_to_slidecabinet_handle(env):
    behind_pos = env._data.site("slidehandle_behind_site").xpos
    end_pos = env._data.site("slidehandle_site").xpos
    quat, _ = get_site_quaternion(behind_pos, end_pos)
    waypoints = [[end_pos, quat],]
    return waypoints

def move_to_microwave_handle(env):
    end_pos = env._data.site("microhandle_site").xpos
    # behind_pos = env._data.site("microhandle_behind_site").xpos
    # quat, _ = get_site_quaternion(behind_pos, end_pos)

    behind_pos = env._data.site("micro_behind_site").xpos
    front_pos = env._data.site("micro_front_site").xpos
    quat, direction_vector = get_site_quaternion(behind_pos, front_pos)
    # 沿着法向量增加一个小偏移，作为附近的点
    near_pos = deepcopy(end_pos)
    gap = 0.01
    near_pos = np.array(near_pos) + -1 * np.array(direction_vector) * gap

    waypoints = [[near_pos, quat],]
    return waypoints

def move_to_microwave_timer_knob(env):
    end_pos = env._data.site("micro_timer_knob_site").xpos
    behind_pos = env._data.site("micro_behind_site").xpos
    front_pos = env._data.site("micro_front_site").xpos
    quat, direction_vector = get_site_quaternion(behind_pos, front_pos)

    # 沿着法向量增加一个小偏移，作为附近的点
    near_pos = deepcopy(end_pos)
    gap = 0.01
    near_pos = np.array(near_pos) + -1 * np.array(direction_vector) * gap

    waypoints = [[near_pos, quat], ]
    return waypoints

def move_to_microwave_temperature_knob(env):
    end_pos = env._data.site("micro_temperature_knob_site").xpos
    behind_pos = env._data.site("micro_behind_site").xpos
    front_pos = env._data.site("micro_front_site").xpos
    quat, direction_vector = get_site_quaternion(behind_pos, front_pos)
    # 沿着法向量增加一个小偏移，作为附近的点
    near_pos = deepcopy(end_pos)
    gap = 0.005
    near_pos = np.array(near_pos) + -1 * np.array(direction_vector) * gap

    waypoints = [[near_pos, quat], ]
    return waypoints
