import copy
import os
from gpt.qwen_query import qwen_query

user_contents = [
"""
Your goal is to set the  joint angles of some articulated objects to the right value in the initial state, given a task. The task is for a robot arm to learn the corresponding skills to manipulate the articulated object. 

The input to you will include the task name, a short description of the task, the articulation tree of the articulated object, a semantic file of the articulated object, the bodies and joints of the articulated objects that will be involved in the task, and the substeps for doing the task. 

You should output for each joint involved in the task, what joint value it should be set to. You should output a number with 0 or 1, where 0 corresponds to the lower limit of that joint angle, and 1 corresponds to the upper limit of the joint angle. 

By default, the joints in an object are set to their lower joint limits. You can assume that the lower joint limit corresponds to the natural state of the articulated object. E.g., for a door's hinge joint, 0 means it is closed, and 1 means it is open.

Here are some examples:

Input:
Task Name: Close the microwave door
Description: The robotic arm will close the microwave door after put the mug in it.

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

substeps:
move to the microwave door	
grasp the microwave door
close the microwave door
release the microwave door


Output:
The goal is for the robot arm to learn to close the microwave door. Therefore, the door needs to be initially opened, thus, we are setting its value to 1, which corresponds to the upper joint limit. 
```joint values
microdoorroot_joint: 1
```

Another example:
Task Name: Adjust Microwave Temperature
Description: The robotic arm will turn the microwaves temperature knob to set the desired heating power.

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

substeps:
move to the microwave temperature knob
grasp the microwave temperature knob
rotate the microwave temperature knob
release the microwave temperature knob

Output:
For the robot to learn to rotate the temperature knob, it should be off. Therefore, micro_temperature_joint should be set to its value to 0, which corresponds to the lower joint limit.
```joint value
micro_temperature_joint: 0
```

One more example:
Task Name: Heat_Food_in_Microwave
Description: The robot arm places a mug inside the microwave, and sets the microwave temperature to be appropriate for heating the food

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
- micro_temperature_knob: the robot needs to approach micro_temperature_knob, which is the temperature knob, to rotate it to set the desired temperature.

Joints:
- micro_temperature_joint: from the articulation tree, this connects micro_temperature_knob. The robot needs to actuate it to rotate micro_temperature_knob to the desired temperature.

substeps:
move to the mug inside the sliding cabinet
grasp the mug
put the mug into microwave
release the mug
move to the microwave temperature knob
grasp the microwave temperature knob
rotate the microwave temperature knob
release the microwave temperature knob


Output:
As noted in the substeps, this task involves retrieve one item from slidecabinet, put it into microwave, and rotate temperature knob. The robot needs to first move into the sliding cabinet, the slide door should be initially opened, its value should be 1, which corresponds to the upper joint limit. Then the robot need to put the item into microwave, so the microwave door also need to be initially opened, its value should be 1, which corresponds to the upper joint limit. Finally, the robot need to rotate the temperature knob, it should be off. Therefore, micro_temperature_joint should be set to its value to 0, which corresponds to the lower joint limit.

```joint value
slidedoor_joint: 1
microdoorroot_joint: 1
micro_temperature_joint: 0
```


One more example:
Task Name: Open Slide Cabinet Door
Description:  The robotic arm will open the slide cabinet door

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
- slidedoor: from the semantics, this is the door of the microwave. The robot needs to approach this door in order to open it.

Joints:
- slidedoor_joint: from the articulation tree, this is the hinge joint that connects microdoorroot (the door). Therefore, the robot needs to actuate this joint for opening the door.

substeps:
move to the slide cabinet door
grasp the slide cabinet door
open the slide cabinet door
release the slide cabinet door

Output:
The goal is for the robot arm to learn to open the slide cabinet door. Therefore, the door needs to be initially closed, thus, we are setting its value to 0, which corresponds to the lower joint limit. 
```joint values
slidedoor_joint: 0
```

Note that, If the substeps include taking an object out of Container A and placing it into Container B or on table, **it can be assumed that the doors of A and B are already open**, allowing the robotic arm to directly grasp the object and place it into B. For example, if moving a mug from the sliding cabinet to the microwave, the robot arm can directly grasp the mug and place it into microwave, and close microwave door. Therefore, both the sliding cabinet door and the microwave door joint should be set to 1, which corresponds to the upper joint limit.
If the substeps include 'move to the mug inside the sliding cabinet', the door of sliding cabinet should be initially open, the joint value should be set to 1, which corresponds to the upper joint limit.

Can you do it for the following task:
"""
]

def find_flag(src_str, target_str_list):
    flag = False
    for item in target_str_list:
        if src_str.startswith(item):
            flag = True
            break
    return flag

assistant_contents = []

def query_joint_angle(task_name, task_description, articulation_tree_list, semantics_list, involved_bodies, involved_joints, substeps, save_path=None, 
                      temperature=0.1, model='gpt-4'):
    query_task = """
Task name: {}
Description: {}

""".format(task_name, task_description,)
    
    articulation_str = "\n"
    for i in range(len(articulation_tree_list)):
        articulation_str += articulation_tree_list[i]
        articulation_str += semantics_list[i]
    
    involved_str = f"\nBodies:\n {involved_bodies} \nJoints: \n{involved_joints} \n"
    
    substeps_str = f"\nsubsteps: \n{substeps} \n"
    
    new_user_contents = copy.deepcopy(user_contents)
    new_user_contents[0] = new_user_contents[0] + query_task + articulation_str + involved_str + substeps_str

    system = "You are a helpful assistant."
    response = qwen_query(system, new_user_contents, assistant_contents, save_path=save_path, temperature=temperature, model=model)

    # TODO: parse the response to get the joint angles
    response = response.split("\n")

    joint_value_str_list = ["```joint value", "**joint value", 'joint value']

    joint_values = {}
    for l_idx, line in enumerate(response):
        if find_flag(line.lower(), joint_value_str_list):
            for l_idx_2 in range(l_idx+1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                line_content = response[l_idx_2].split("#")[0].strip()  # 移除注释部分
                if not line_content or line_content.lower().strip() == "none":
                    continue
                joint_name, joint_value = line_content.split(":")
                joint_values[joint_name.strip().lstrip()] = joint_value.strip().lstrip()

    return joint_values
