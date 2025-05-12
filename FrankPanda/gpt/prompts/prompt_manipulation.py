import numpy as np
import copy
import time, datetime
import os
import json

from gpt.prompts.utils import build_task_given_text, parse_task_response
from gpt.qwen_query import qwen_query

task_user_contents = """
I will give you an articulated object, with its articulation tree and semantics. Your goal is to imagine some tasks that a robotic arm can perform with this articulated object in household scenarios. You can think of the robotic arm as a Franka Panda robot. The task will be built in a simulator for the robot to learn it. 

Focus on manipulation or interaction with the object itself. Sometimes the object will have functions, e.g., a microwave can be used to heat food, in these cases, feel free to include other objects that are needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality.
Please do not think of tasks that bodies or joints are not in its articulation tree. Do not think of tasks that may move the object itself.

For each task you imagined, please write in the following format: 
Task name: the name of the task.
Description: some basic descriptions of the tasks. 
Additional Objects: Additional objects other than the provided articulated object required for completing the task. 
Bodies: Bodies of the articulated objects that are required to perform the task. 
- Body 1: reasons why this body is needed for the task
- Body 2: reasons why this body is needed for the task
- …
Joints: Joints of the articulated objects that are required to perform the task. 
- Joint 1: reasons why this joint is needed for the task
- Joint 2: reasons why this joint is needed for the task
- …


Example Input: 

```Oven articulation tree
bodies: 
base
body_0
body_1
body_2
body_3
body_4
body_5
body_6
body_7

joints: 
joint_name: joint_0 joint_type: hinge parent_body: body_7 child_body: body_0
joint_name: joint_1 joint_type: hinge parent_body: body_7 child_body: body_1
joint_name: joint_2 joint_type: hinge parent_body: body_7 child_body: body_2
joint_name: joint_3 joint_type: hinge parent_body: body_7 child_body: body_3
joint_name: joint_4 joint_type: hinge parent_body: body_7 child_body: body_4
joint_name: joint_5 joint_type: hinge parent_body: body_7 child_body: body_5
joint_name: joint_6 joint_type: hinge parent_body: body_7 child_body: body_6
joint_name: joint_7 joint_type: none parent_body: base child_body: body_7
```

```Oven semantics
body_0 hinge door
body_1 hinge knob
body_2 hinge knob
body_3 hinge knob
body_4 hinge knob
body_5 hinge knob
body_6 hinge knob
body_7 heavy oven_body
```

Example output:

Task Name: Open Oven Door
Description: The robotic arm will open the oven door.
Additional Objects: None
Bodies:
- body_0: from the semantics, this is the door of the oven. The robot needs to approach this door in order to open it. 
Joints: 
- joint_0: from the articulation tree, this is the hinge joint that connects body_0. Therefore, the robot needs to actuate this joint for opening the door.


Task Name: Adjust Oven Temperature
Description: The robotic arm will turn one of the oven's hinge knobs to set a desired temperature.
Additional Objects: None
Bodies:
- body_1: the robot needs to approach body_1, which is assumed to be the temperature knob, to rotate it to set the temperature.
Joints:
- joint_1: joint_1 connects body_1 from the articulation tree. The robot needs to actuate it to rotate body_1 to the desired temperature.


Task Name: Heat a hamburger Inside Oven 
Description: The robot arm places a hamburger inside the oven, and sets the oven temperature to be appropriate for heating the hamburger.
Additional Objects: hamburger
Bodies:
- body_0: body_0 is the oven door from the semantics. The robot needs to open the door in order to put the hamburger inside the oven.
- body_1: the robot needs to approach body_1, which is the temperature knob, to rotate it to set the desired temperature.
Joints:
- joint_0: from the articulation tree, this is the revolute joint that connects body_0 (the door). Therefore, the robot needs to actuate this joint for opening the door.
- joint_1: from the articulation tree, joint_1 connects body_1, which is the temperature knob. The robot needs to actuate it to rotate body_1 to the desired temperature.

Task Name: Set Oven Timer
Description: The robot arm turns a timer knob to set cooking time for the food.
Additional Objects: None.
Bodies: 
- body_2: body_2 is assumed to be the knob for controlling the cooking time. The robot needs to approach body_2 to set the cooking time.
Joints:
- joint_2: from the articulation tree, joint_2 connects body_2. The robot needs to actuate joint_2 to rotate body_2 to the desired position, setting the oven timer.


Can you do the same for the following object:
"""

def generate_task(object_category=None, existing_response=None, temperature_dict=None, model_dict=None, meta_path="generated_tasks"):
    # send the object articulation tree, semantics file and get task descriptions, invovled objects and joints
    articulation_tree_path = f"data/assets/dataset/{object_category}/body_and_joint.txt"
    with open(articulation_tree_path, 'r') as f:
        articulation_tree = f.readlines()
    
    semantics = f"data/assets/dataset/{object_category}/semantics.txt"
    with open(semantics, 'r') as f:
        semantics = f.readlines()

    task_user_contents_filled = copy.deepcopy(task_user_contents)
    articulation_tree_filled = """
```{} articulation tree
{}
```""".format(object_category, "".join(articulation_tree))
    semantics_filled = """
```{} semantics
{}
```""".format(object_category, "".join(semantics))
    task_user_contents_filled = task_user_contents_filled + articulation_tree_filled + semantics_filled

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = "data/{}/{}_{}".format(meta_path, object_category, time_string)
        if not os.path.exists(save_folder + "/gpt_response"):
            os.makedirs(save_folder + "/gpt_response")

        save_path = "{}/gpt_response/task_generation.json".format(save_folder)

        print("=" * 50)
        print("=" * 20, "generating task", "=" * 20)
        print("=" * 50)

        task_response = qwen_query(system, [task_user_contents_filled], [], save_path=save_path, debug=False, 
                              temperature=temperature_dict['task_generation'],
                              model=model_dict['task_generation'])
   
    else:
        with open(existing_response, 'r') as f:
            data = json.load(f)
        task_response = data["res"]
        print(task_response)
            
    ### generate task yaml config
    task_names, task_descriptions, additional_objects, bodies, joints = parse_task_response(task_response)
    task_number = len(task_names)
    print("task number: ", task_number)

    all_config_paths = []
    for task_idx in range(task_number):
        if existing_response is None:
            time.sleep(20)
        task_name = task_names[task_idx]
        task_description = task_descriptions[task_idx]
        additional_object = additional_objects[task_idx]
        involved_bodies = bodies[task_idx]
        involved_joints = joints[task_idx]

        config_path = build_task_given_text(object_category, task_name, task_description, additional_object, involved_bodies, involved_joints, 
                          articulation_tree_filled, semantics_filled, save_folder, temperature_dict, model_dict=model_dict)
        all_config_paths.append(config_path)

    return all_config_paths
