import copy
import os
import yaml
from gpt.qwen_query import qwen_query

user_contents = [
"""
A robotic arm is trying to solve some household object manipulation tasks to learn corresponding skills in a simulator.

We will provide with you the task description, the initial scene configurations of the task, which contains the objects in the task and certain information about them. 
Your goal is to decompose the task into executable sub-steps for the robot, and for each substep, you should either call a primitive action that the robot can execute, or design a reward function for the robot to learn, to complete the substep. 

Common substeps include moving towards a location, grasping an object, and interacting with the joint of an articulated object.

An example task:
Task Name: Adjust Microwave Temperature
Description: The robotic arm will turn the microwaves temperature knob to set a desired temperature
Initial config:
```yaml
-   use_table: true
-   center: (0.8, 0.0, 1.6)
    lang: a common microwave
    movable: false
    name: microwave
    type: xml
```

I will also give you the articulation tree and semantics file of the articulated object in the task. Such information will be useful for writing the reward function/the primitive actions, for example, when the reward requires accessing the joint value of a joint in the articulated object, or the position of a body in the articulated object, or when the primitive needs to access a name of the object.
```Microwave articulation tree:
bodies:
microroot
microdoorroot
micro_timer_knob
micro_temperature_knob

joints: 
joint_name: microdoorroot_joint joint_type: hinge parent_body: microroot child_body: microdoorroot
joint_name: micro_timer_joint joint_type: hinge parent_body: microroot child_body: micro_timer_knob
joint_name: micro_temperature_joint joint_type: hinge parent_body: microroot child_body: micro_temperature_knob
```

```Microwave semantics
microroot heavy microwave_body
microdoorroot hinge door
micro_timer_knob hinge timer_knob
micro_temperature_knob hinge temperature_knob
```

I will also give you the bodies and joints of the articulated object that will be used for completing the task:
Bodies:
- micro_temperature_knob: We know from the semantics that micro_temperature_knob is a hinge knob. It is assumed to be the knob that controls the temperature of the microwave. The robot needs to actuate this knob to set the temperature of the microwave.

Joints:
- micro_temperature_joint: from the articulation tree, micro_temperature_joint connects micro_temperature_knob and is a hinge joint. Therefore, the robot needs to actuate micro_temperature_joint to turn micro_temperature_knob, which is the knob.


For each substep, you should decide whether the substep can be achieved by using the provided list of primitives. If not, you should then write a reward function for the robot to learn to perform this substep. 
If you choose to write a reward function for the substep, you should also specify the action space of the robot when learning this reward function. 
There are 2 options for the action space: "delta-translation", where the action is the delta translation of the robot end-effector, suited for local movements; and "normalized-direct-translation", where the action specifies the target location the robot should move to, suited for moving to a target location.

Here is a list of primitives the robot can do. The robot is equipped with a suction gripper, which makes it easy for the robot to grasp an object. 
- grasp_object_trajectories(env): the robot arm will apply force to its end effector to grasp an object. 
- release_grasp_trajectories(env): the robot arm will release the grasped object. 
- approach_object_trajectories(env, waypoint_func_name): the robot arm will move to the target position following the waypoints provided by the waypoint_func_name. 
Note that all primitives will return a trajectory list which represents the motion trajectory for performing that action. 

The waypoint_func_name represents a function that produces a sequence of 'waypoints', defining the robot arm's trajectory through space. 
Here is a list of available waypoint_func functions that the approach_object_trajectories can use. You are required to select the appropriate waypoint_func based on its functionality and the specific details of the substep. Note that your selection must be limited to the functions within this list.
- move_to_mug_in_slidecabinet: generate waypoints to move the robot arm end effector near the mug which inside the sliding cabinet.
- put_mug_in_microwave: generate waypoints for the robot arm end effector to put the mug into the microwave. Use this function to place the mug into the microwave regardless of whether the mug is in the slidecabinet or on the table. For example, if the mug is in the slidecabinet, you can directly use this function to put mug into microwave.
- move_to_slidecabinet_handle: generate waypoints to move the robot arm end effector near the sliding cabinet door handle.
- move_to_microwave_handle: generate waypoints to move the robot arm end effector near the microwave door handle.
- move_to_microwave_timer_knob: generate waypoints to move the robot arm end effector near the microwave timer knob.
- move_to_microwave_temperature_knob: generate waypoints to move the robot arm end effector near the microwave temperature knob.
- retrieve_the_mug_from_slidecabinet: generate waypoints to retrieve the mug from slidecabinet and put it on the table.

Here is a list of available robot arm end effector closure degrees.It is used to determine the closure degree of the end effector when the robotic arm executes above primitives.
- 0.0: End effector closure for robotic arm simple motions or when releasing an object.
- 0.38: End effector closure for robotic arm when grasping a big object (like a mug) or carrying big object (like a mug) while moving.
- 0.8: End effector closure for robotic arm when grasping a small object (like door handle or knob).

You should decide the primitive function name, waypoint_func_name and pinch_grasp, return the result in the following format:
```primitive
- primitive_func_name: some_primitive_function
- waypoint_func_name: some_waypoint_function or None
- gripper_closure: some_closure_degrees
```

Here is a list of helper functions that you can use for designing the reward function or the success condition:
- get_site_position(env_data, site_name): get the position of center of mass of object with site_name. Currently, site_name supports 'slidehandle_site', 'microhandle_site', 'micro_timer_knob_site' and 'micro_temperature_knob_site'. 'slidehandle_site' represents the center point of the sliding cabinet door handle. 'microhandle_site' represents the center point of the microwave door handle. 'micro_timer_knob_site' represents the center point of the microwave timer knob. 'micro_temperature_knob_site' represents the center point of the microwave temperature knob.Please choose the appropriate site_name based on the substep description.
- get_joint_state(env_data, joint_name): get the joint angle value of a joint in an object.
- get_joint_limit(env_model, joint_name): get the lower and upper joint angle limit of a joint in an object, returned as a 2-element tuple.
- get_eef_pos_and_quat(env_data): returns the position, orientation of the robot end-effector as a 2-element tuple.

You can assume that for objects, the joint value: 0 corresponds to their natural state, e.g., a box is closed with the lid joint being 0, and a lever is unpushed when the joint angle is 0.

For the above task "Adjust Microwave Temperature", it can be decomposed into the following substeps, primitives, and reward functions:

substep 1: move to the microwave temperature knob
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: move_to_microwave_temperature_knob
- gripper_closure: 0.0 # chose 0.0 for simple motions
```

substep 2: grasp the microwave temperature knob
```primitive
- primitive_func_name:  grasp_object_trajectories
- waypoint_func_name: None
- gripper_closure: 0.8 # chose 0.8 for grasping a small object (like door handle or knob)
```

substep 3: rotate the microwave temperature knob
```reward
def compute_reward(env):
    # this reward encourages the end-effector to stay near knob to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    knob_pos = get_site_position(env._data, 'micro_temperature_knob_site')
    reward_near = np.linalg.norm(eef_pos - door_pos)

    # Get the joint state of the knob. We know from the semantics and the articulation tree that micro_temperature_joint connects micro_temperature_knob and is the joint that controls the rotation of the knob.
    reward_joint_name = "micro_temperature_joint"
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
```

```action space
delta-translation
```

substep 4: release the microwave temperature knob
```primitive
- primitive_func_name:  release_grasp_trajectories
- waypoint_func_name: None
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```

I will give some more examples of decomposing the task. Reply yes if you understand the goal.

Note that, If the decomposed sub-steps include taking an object out of Container A and placing it into Container B or on table, **it can be assumed that the doors of A and B are already open**, allowing the robotic arm to directly grasp the object and place it into B, and close the door of B. For example, if moving a mug from the sliding cabinet to the microwave, the robot arm candirectly grasp the mug and place it into microwave, and close microwave door. The correct sequence of sub-steps is as follows:
```
substep 1: move to the mug inside the sliding cabinet
substep 2: grasp the mug
substep 3: put the mug into microwave
substep 4: release the mug
substep 5: move to the microwave door
substep 6: grasp the microwave door
substep 7: close the microwave door
substep 8: release the microwave door
```
""",

"""
Another example:
Task Name: Retrieve Mug from Slide Cabinet
Description: The robotic arm will reach slidecabinet inside to grab mug, place it on the table
Initial config:
```yaml
-   use_table: true
-   center: (0.82, 0.16, 2.3)
    lang: a common slide cabinet
    movable: false
    name: slidecabinet
    type: xml
-   center: (0.6, 0.4, 2.141)
    lang: a common mug
    movable: true
    name: mug
    type: xml
-   center: (0.5, -0.6, 1.6)
    lang: a common microwave
    movable: false
    name: microwave
    type: xml
```

```slidecabinet articulation tree:
bodies:
slide
slidedoor

joints: 
joint_name: slidedoor_joint joint_type: slide parent_body: slide child_body: slidedoor
```

```slidecabinet semantics
slide heavy slidecabinet_body
slidedoor slide door
```

```Mug articulation tree:
bodies:
mug_base

joints: 
None
```

```Mug semantics
mug_base heavy mug_body
```

```Microwave articulation tree:
bodies:
microroot
microdoorroot
micro_timer_knob
micro_temperature_knob

joints: 
joint_name: microdoorroot_joint joint_type: hinge parent_body: microroot child_body: microdoorroot
joint_name: micro_timer_joint joint_type: hinge parent_body: microroot child_body: micro_timer_knob
joint_name: micro_temperature_joint joint_type: hinge parent_body: microroot child_body: micro_temperature_knob
```

```Microwave semantics
microroot heavy microwave_body
microdoorroot hinge door
micro_timer_knob hinge timer_knob
micro_temperature_knob hinge temperature_knob
```

Bodies:
- slide: the robot needs to approach slide, which is the slide cabinet body, to retrieve the mug.
- slidedoor: from the semantics, this is the door of the slide cabinet. The robot needs to approach this door in order to open it.

Joints: 
- slidedoor_joint: from the articulation tree, this is the slide joint that connects slidedoor. Therefore, the robot needs to actuate this joint for opening the door.

This task can be decomposed as follows:

substep 1: move to the mug inside the sliding cabinet
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: move_to_mug_in_slidecabinet
- gripper_closure: 0.0 # chose 0.0 for simple motions
```

substep 2: grasp the mug
```primitive
- primitive_func_name:  grasp_object_trajectories
- waypoint_func_name: None
- gripper_closure: 0.38 # chose 0.38 for grasping a big object
```

substep 3: retrieve the mug from slidecabinet
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: retrieve_the_mug_from_slidecabinet
- gripper_closure: 0.38 # chose 0.38 for carrying big object while moving
```

substep 4: release the mug
```primitive
- primitive_func_name:  release_grasp_trajectories
- waypoint_func_name: None
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```


I will provide more examples in the following messages. Please reply yes if you understand the goal.
""",

"""
Here is another example:

Task Name:  Adjust Microwave Temperature
Description: The robotic arm will turn the microwaves temperature knob to set the desired heating power.
Initial config:
```yaml
-   use_table: true
-   center: (0.8, 0.0, 1.6)
    lang: a common microwave
    movable: false
    name: Microwave
    type: xml
-   center: (0.82, 0.16, 2.3)
    lang: a wooden slidecabinet
    movable: false
    name: Slidecabinet
    type: xml
-   center: (0.6, 0.4, 2.141)
    lang: a ceramic mug
    movable: true
    name: Mug
    type: xml
```
```slidecabinet articulation tree:
bodies:
slide
slidedoor

joints: 
joint_name: slidedoor_joint joint_type: slide parent_body: slide child_body: slidedoor
```

```slidecabinet semantics
slide heavy slidecabinet_body
slidedoor slide door
```

```Mug articulation tree:
bodies:
mug_base

joints: 
None
```

```Mug semantics
mug_base heavy mug_body
```

```Microwave articulation tree:
bodies:
microroot
microdoorroot
micro_timer_knob
micro_temperature_knob

joints: 
joint_name: microdoorroot_joint joint_type: hinge parent_body: microroot child_body: microdoorroot
joint_name: micro_timer_joint joint_type: hinge parent_body: microroot child_body: micro_timer_knob
joint_name: micro_temperature_joint joint_type: hinge parent_body: microroot child_body: micro_temperature_knob
```

```Microwave semantics
microroot heavy microwave_body
microdoorroot hinge door
micro_timer_knob hinge timer_knob
micro_temperature_knob hinge temperature_knob
```

Bodies:
- micro_temperature_knob: from the semantics, this is the temperature knob of the microwave. The robot needs to approach this knob in order to adjust it.

Joints: 
- micro_temperature_joint: from the articulation tree, this is the microroot joint that connects micro_temperature_knob. Therefore, the robot needs to actuate this joint for adjust the temperature.

This task can be decomposed as follows:

substep 1: move to the microwave temperature knob
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: move_to_microwave_temperature_knob
- gripper_closure: 0.0 # chose 0.0 for simple motions
```

substep 2: grasp the microwave temperature knob
```primitive
- primitive_func_name:  grasp_object_trajectories
- waypoint_func_name: None
- gripper_closure: 0.8 # chose 0.8 for grasping a small object (like door handle or knob)
```

substep 3: rotate the microwave temperature knob
```reward
def compute_reward(env):
    # this reward encourages the end-effector to stay near knob to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    knob_pos = get_site_position(env._data, 'micro_temperature_knob_site')
    reward_near = np.linalg.norm(eef_pos - door_pos)

    # Get the joint state of the knob. We know from the semantics and the articulation tree that micro_temperature_joint connects micro_temperature_knob and is the joint that controls the rotation of the knob.
    reward_joint_name = "micro_temperature_joint"
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
```

```action space
delta-translation
```

substep 4: release the microwave temperature knob
```primitive
- primitive_func_name:  release_grasp_trajectories
- waypoint_func_name: None
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```


I will provide more examples in the following messages. Please reply yes if you understand the goal.
""",
"""
Here is another example:

Task Name:  Open slidecabinet Door
Description: The robotic arm will open the slidecabinet door.
Initial config:
```yaml
-   use_table: true
-   center: (0.8, 0.0, 1.6)
    lang: a common microwave
    movable: false
    name: microwave
    type: xml
-   center: (0.82, 0.16, 2.3)
    lang: a common slide cabinet
    movable: false
    name: slidecabinet
    type: xml
-   center: (0.6, 0.4, 2.141)
    lang: a common mug
    movable: true
    name: mug
    type: xml
```

```slidecabinet articulation tree:
bodies:
slide
slidedoor

joints: 
joint_name: slidedoor_joint joint_type: slide parent_body: slide child_body: slidedoor
```

```slidecabinet semantics
slide heavy slidecabinet_body
slidedoor slide door
```

```Mug articulation tree:
bodies:
mug_base

joints: 
None
```

```Mug semantics
mug_base heavy mug_body
```

```Microwave articulation tree:
bodies:
microroot
microdoorroot
micro_timer_knob
micro_temperature_knob

joints: 
joint_name: microdoorroot_joint joint_type: hinge parent_body: microroot child_body: microdoorroot
joint_name: micro_timer_joint joint_type: hinge parent_body: microroot child_body: micro_timer_knob
joint_name: micro_temperature_joint joint_type: hinge parent_body: microroot child_body: micro_temperature_knob
```

```Microwave semantics
microroot heavy microwave_body
microdoorroot hinge door
micro_timer_knob hinge timer_knob
micro_temperature_knob hinge temperature_knob
```

Bodies:
- slidedoor: from the semantics, this is the door of the slidecabinet. The robot needs to approach this door in order to open it.

Joints: 
- slidedoor_joint: from the articulation tree, this is the slide joint that connects slidedoor. Therefore, the robot needs to actuate this joint for opening the door.

This task can be decomposed as follows:

substep 1: move to the slidecabinet door
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: move_to_slidecabinet_handle
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```

substep 2: grasp the slidecabinet door
```primitive
- primitive_func_name:  grasp_object_trajectories
- waypoint_func_name: None
- gripper_closure: 0.8 # chose 0.8 for grasping a small object (like door handle or knob)
```

substep 3: open the slidecabinet door
```reward
def compute_reward(env):
    # This reward encourages the end-effector to stay near door to grasp it.
    eef_pos, eef_quat = get_eef_pos_and_quat(env._data)
    door_pos = get_site_position(env._data, 'slidehandle_site')
    reward_near = np.linalg.norm(eef_pos - door_pos)

    # Get the joint state of the door. We know from the semantics and the articulation tree that slidedoor_joint connects slidedoor and is the joint that controls the rotation of the door.
    reward_joint_name = "slidedoor_joint"
    joint_angle = get_joint_state(env._data, reward_joint_name)

    # The reward is the negative distance between the current joint angle and the joint angle when the slidecabinet is fully open (upper limit).
    joint_limit_low, joint_limit_high = get_joint_limit(env._model, reward_joint_name)
    max_joint_angle = joint_limit_high

    # Avoid joint_limit_low is a negative value.
    max_joint_angle = joint_limit_low if np.abs(joint_limit_low) > np.abs(joint_limit_high) else joint_limit_high
    
    reward_open = np.abs(joint_angle - max_joint_angle)

    reward = -reward_near - 5 * reward_open
    success = reward_open < 0.25 * np.abs(joint_limit_high - joint_limit_low)  # for opening door, we think 75 percent is enough

    return reward, success, reward_joint_name
```

```action space
delta-translation
```

substep 4: release the slidecabinet door
```primitive
- primitive_func_name:  release_grasp_trajectories
- waypoint_func_name: None
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```


I will provide more examples in the following messages. Please reply yes if you understand the goal.
""",

"""
Here is another example:

Task Name:  Close Microwave Door
Description: The robotic arm will close the microwave door after put the mug in it.
Initial config:
```yaml
-   use_table: true
-   center: (0.8, 0.0, 1.6)
    lang: a common microwave
    movable: false
    name: microwave
    type: xml
-   center: (0.82, 0.16, 2.3)
    lang: a common slide cabinet
    movable: false
    name: slidecabinet
    type: xml
-   center: (0.6, 0.4, 2.141)
    lang: a common mug
    movable: true
    name: mug
    type: xml
```
```slidecabinet articulation tree:
bodies:
slide
slidedoor

joints: 
joint_name: slidedoor_joint joint_type: slide parent_body: slide child_body: slidedoor
```

```slidecabinet semantics
slide heavy slidecabinet_body
slidedoor slide door
```

```Mug articulation tree:
bodies:
mug_base

joints: 
None
```

```Mug semantics
mug_base heavy mug_body
```

```Microwave articulation tree:
bodies:
microroot
microdoorroot
micro_timer_knob
micro_temperature_knob

joints: 
joint_name: microdoorroot_joint joint_type: hinge parent_body: microroot child_body: microdoorroot
joint_name: micro_timer_joint joint_type: hinge parent_body: microroot child_body: micro_timer_knob
joint_name: micro_temperature_joint joint_type: hinge parent_body: microroot child_body: micro_temperature_knob
```

```Microwave semantics
microroot heavy microwave_body
microdoorroot hinge door
micro_timer_knob hinge timer_knob
micro_temperature_knob hinge temperature_knob
```

Bodies:
- microdoorroot: from the semantics, this is the door of the microwave. The robot needs to approach this door in order to close it.

Joints: 
- microdoorroot_joint: from the articulation tree, this is the microroot joint that connects microdoorroot. Therefore, the robot needs to actuate this joint for closing the door.

This task can be decomposed as follows:

substep 1: move to the microwave door
```primitive
- primitive_func_name:  approach_object_trajectories
- waypoint_func_name: move_to_microwave_handle
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```

substep 2: grasp the microwave door
```primitive
- primitive_func_name:  grasp_object_trajectories
- waypoint_func_name: None
- gripper_closure: 0.8 # chose 0.8 for grasping a small object (like door handle or knob)
```

substep 3: close the microwave door
```reward
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
```

```action space
delta-translation
```

substep 4: release the microwave door
```primitive
- primitive_func_name:  release_grasp_trajectories
- waypoint_func_name: None
- gripper_closure: 0.0 # chose 0.0 for releasing an object
```

Please decompose the following task into substeps. For each substep, write a primitive yaml config or a reward function, and the action space if the reward is used. 

The primitives you can call:
- grasp_object_trajectories(env): the robot arm will apply force to its end effector to grasp an object. 
- release_grasp_trajectories(env): the robot arm will release the grasped object. 
- approach_object_trajectories(env, waypoint_func_name): the robot arm will move to the target position following the waypoints provided by the waypoint_func_name. 
Note that all primitives will return a trajectory list which represents the motion trajectory for performing that action. 

The waypoint_func_name represents a function that produces a sequence of 'waypoints', defining the robot arm's trajectory through space. 
Here is a list of available waypoint_func functions that the approach_object_trajectories can use. You are required to select the appropriate waypoint_func based on its functionality and the specific details of the substep. Note that your selection must be limited to the functions within this list.
- move_to_mug_in_slidecabinet: generate waypoints to move the robot arm end effector near the mug which inside the sliding cabinet.
- put_mug_in_microwave: generate waypoints for the robot arm end effector to put the mug into the microwave. Use this function to place the mug into the microwave regardless of whether the mug is in the slidecabinet or on the table. For example, if the mug is in the slidecabinet, you can directly use this function to put mug into microwave.
- move_to_slidecabinet_handle: generate waypoints to move the robot arm end effector near the sliding cabinet door handle.
- move_to_microwave_handle: generate waypoints to move the robot arm end effector near the microwave door handle.
- move_to_microwave_timer_knob: generate waypoints to move the robot arm end effector near the microwave timer knob.
- move_to_microwave_temperature_knob: generate waypoints to move the robot arm end effector near the microwave temperature knob.
- retrieve_the_mug_from_slidecabinet: generate waypoints to retrieve the mug from slidecabinet and put it on the table.

Here is a list of available robot arm end effector closure degrees.It is used to determine the closure degree of the end effector when the robotic arm executes above primitives.
- 0.0: End effector closure for robotic arm simple motions or when releasing an object.
- 0.38: End effector closure for robotic arm when grasping a big object (like a mug) or carrying big object (like a mug) while moving.
- 0.8: End effector closure for robotic arm when grasping a small object (like door handle or knob).

You should decide the primitive function name, waypoint_func_name and pinch_grasp, return the result in the following format:
```primitive
- primitive_func_name: some_primitive_function
- waypoint_func_name: some_waypoint_function or None
- gripper_closure: some_closure_degrees
```

The APIs you can use for writing the reward function:
- get_site_position(env_data, site_name): get the position of center of mass of object with site_name. Currently, site_name supports 'slidehandle_site', 'microhandle_site', 'micro_timer_knob_site' and 'micro_temperature_knob_site'. 'slidehandle_site' represents the center point of the sliding cabinet door handle. 'microhandle_site' represents the center point of the microwave door handle. 'micro_timer_knob_site' represents the center point of the microwave timer knob. 'micro_temperature_knob_site' represents the center point of the microwave temperature knob.Please choose the appropriate site_name based on the substep description.
- get_joint_state(env_data, joint_name): get the joint angle value of a joint in an object.
- get_joint_limit(env_model, joint_name): get the lower and upper joint angle limit of a joint in an object, returned as a 2-element tuple.
- get_eef_pos_and_quat(env_data): returns the position, orientation of the robot end-effector as a 2-element tuple.

The action space you can use for learning with the reward: delta-translation is better suited for small movements, and normalized-direct-translation is better suited for directly specifying the target location of the robot end-effector. 
You can assume that for objects, the joint value: 0 corresponds to their natural state, e.g., a box is closed with the lid joint being 0, and a lever is unpushed when the joint angle is 0.
Note that, If the decomposed sub-steps include taking an object out of Container A and placing it into Container B or on table, **it can be assumed that the doors of A and B are already open**, allowing the robotic arm to directly grasp the object and place it into B, and close the door of B. For example, if moving a mug from the sliding cabinet to the microwave, the robot arm can directly grasp the mug and place it into microwave, and close microwave door.
In reward function, for opening door, we think 75 percent is enough, for closing door, we think 95 percent is enough, for rotating knob, we think 50 percent is enough
**Please do not repeat the given configurations in you reply. Please do not generate any final summary for these substeps.**
"""
]

assistant_contents = [
"""
Yes, I understand the goal. Please proceed with the next example.
""",

"""
Yes, I understand the goal. Please proceed with the next example.
"""
]



reward_file_header = """
import numpy as np
from manipulation.gpt_reward_api import *

{}

{}
"""

def find_flag(src_str, target_str_list):
    flag = False
    for item in target_str_list:
        if src_str.startswith(item):
            flag = True
            break
    return flag


def decompose_and_generate_reward_or_primitive(task_name, task_description, initial_config, articulation_tree_list, semantics_list, 
                              involved_bodies, involved_joints, yaml_config_path, save_path, temperature=0.4, model='gpt-4'):
    query_task = """
Task name: {}
Description: {}
Initial config:
```yaml
{}
```
""".format(task_name, task_description, initial_config)

    articulation_str = "\n"
    for i in range(len(articulation_tree_list)):
        articulation_str += articulation_tree_list[i]
        articulation_str += semantics_list[i]

    involved_str = f"\nBodies:\n {involved_bodies} \nJoints: \n{involved_joints} \n"
    
    filled_user_contents = copy.deepcopy(user_contents)
    filled_user_contents[-1] = filled_user_contents[-1] + query_task + articulation_str + involved_str

    system = "You are a helpful assistant."
    reward_response = qwen_query(system, filled_user_contents, assistant_contents, save_path=save_path, debug=False, 
                            temperature=temperature, model=model)
    res = reward_response.split("\n")

    substeps = []
    substep_types = []
    reward_or_primitives = []
    action_spaces = []

    ignore_substep_list = ["### substeps",]
    substep_str_list = ["substep", "### substep", "#### substep", "**substep", "** substep",]
    reward_str_list = ["```reward", "**reward", 'reward']
    primitive_str_list = ["```primitive", "**primitive", 'primitive']
    action_space_str_list = ["```action space", "**action space", 'action space']

    num_lines = len(res)
    for l_idx, line in enumerate(res):
        line = line.lower()

        if find_flag(line, ignore_substep_list):
            continue

        if find_flag(line, substep_str_list):
            substep_name = line.split(":")[1].replace("*", "")
            substeps.append(substep_name)

            py_start_idx, py_end_idx = l_idx, l_idx
            for l_idx_2 in range(l_idx + 1, num_lines):
                ### this is a reward
                if find_flag(res[l_idx_2], reward_str_list):
                    substep_types.append("reward")
                    py_start_idx = l_idx_2 + 1
                    for l_idx_3 in range(l_idx_2 + 1, num_lines):
                        if "```" in res[l_idx_3]:
                            py_end_idx = l_idx_3
                            break
            
                if find_flag(res[l_idx_2], primitive_str_list):
                    substep_types.append("primitive")
                    action_spaces.append("None")
                    py_start_idx = l_idx_2 + 1
                    for l_idx_3 in range(l_idx_2 + 1, num_lines):
                        if "```" in res[l_idx_3]:
                            py_end_idx = l_idx_3
                            break
                    break
                
                if find_flag(res[l_idx_2], action_space_str_list):
                    action_space = res[l_idx_2 + 1]
                    action_spaces.append(action_space)
                    break

            reward_or_primitive_lines = res[py_start_idx:py_end_idx]
            reward_or_primitive_lines = [line.lstrip() for line in reward_or_primitive_lines]
            if substep_types[-1] == 'reward':
                # reward_or_primitive_lines[0] = "    " + reward_or_primitive_lines[0]
                for idx in range(1, len(reward_or_primitive_lines)):
                    reward_or_primitive_lines[idx] = "    " + reward_or_primitive_lines[idx]
                reward_or_primitive = "\n".join(reward_or_primitive_lines) + "\n"
            else:
                reward_or_primitive = "\n".join(reward_or_primitive_lines)
                reward_or_primitive = yaml.safe_load(reward_or_primitive)
            

            reward_or_primitives.append(reward_or_primitive)

    task_name = task_name.replace(" ", "_")
    parent_folder = os.path.dirname(os.path.dirname(save_path))
    task_save_path = os.path.join(parent_folder, "task_{}".format(task_name))    
    os.makedirs(task_save_path, exist_ok=True)

    print("\nsubstep: ", substeps)
    print("\nsubstep types: ", substep_types)
    print("\nreward or primitives: ", reward_or_primitives)
    print("\naction spaces: ", action_spaces)

    if len(substeps) == 0:
        raise ValueError("Parse substeps error, please check the LLM result")

    with open(os.path.join(task_save_path, "substeps.txt"), "w") as f:
        f.write("\n".join(substeps))
    with open(os.path.join(task_save_path, "substep_types.txt"), "w") as f:
        f.write("\n".join(substep_types))
    with open(os.path.join(task_save_path, "action_spaces.txt"), "w") as f:
        f.write("\n".join(action_spaces))
    with open(os.path.join(task_save_path, "config_path.txt"), "w") as f:
        f.write(yaml_config_path)

    for idx, (substep, type, reward_or_primitive) in enumerate(zip(substeps, substep_types, reward_or_primitives)):
        substep = substep.lstrip().replace(" ", "_")
        substep = substep.replace("'", "")

        if type == 'reward':
            file_name = os.path.join(task_save_path, f"{substep}.py")
            func_anno = f"## substep: {substep}"
            file_content = reward_file_header.format(func_anno, reward_or_primitive)
            with open(file_name, "w") as f:
                f.write(file_content)
        elif type == 'primitive':
            file_name = os.path.join(task_save_path, f"{substep}.yaml")
            with open(file_name, 'w') as f:
                yaml.dump(reward_or_primitive, f, indent=4)

    return task_save_path


