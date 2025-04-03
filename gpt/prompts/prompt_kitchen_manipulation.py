import numpy as np
import copy
import time, datetime
import os
import json

from gpt.prompts.utils import build_task_given_text, parse_task_response
from gpt.qwen_query import qwen_query

task_user_contents = """
I will give you one or several articulated objectes, with their's articulation tree and semantics. Your goal is to imagine some tasks that a robotic arm can perform with these articulated objectes in kitchen scenarios. You can think of the robotic arm as a Franka Panda robot. The task will be built in a simulator for the robot to learn it. 

Focus on manipulation or interaction with the object itself. Sometimes the object will have functions, e.g., a microwave can be used to heat food, in these cases, only mug can bu used to contain food that is needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality.
Please do not think of tasks that bodies or joints are not in its articulation tree. Do not think of tasks that may move the object itself.

For each task you imagined, please write in the following format: 
Task name: the name of the task.
Description: some basic descriptions of the tasks. 
Additional Objects: Additional objects other than the provided articulated objectes required for completing the task. If mug in articulated objectes, this value should be None.
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


Task Name: Heat food Inside Oven 
Description: The robot arm places a mug inside the oven, and sets the oven temperature to be appropriate for heating the food.
Additional Objects: mug
Bodies:
- body_1: the robot needs to approach body_1, which is the temperature knob, to rotate it to set the desired temperature.
Joints:
- joint_1: from the articulation tree, joint_1 connects body_1, which is the temperature knob. The robot needs to actuate it to rotate body_1 to the desired temperature.

Task Name: Set Oven Timer
Description: The robot arm turns a timer knob to set cooking time for the food.
Additional Objects: None.
Bodies: 
- body_2: body_2 is assumed to be the knob for controlling the cooking time. The robot needs to approach body_2 to set the cooking time.
Joints:
- joint_2: from the articulation tree, joint_2 connects body_2. The robot needs to actuate joint_2 to rotate body_2 to the desired position, setting the oven timer.

Note that, If the task include taking an object out of Container A or placing an object into Container B **it can be assumed that the doors of A and B are already open**, allowing the robotic arm to directly grasp the object, you do not need to open it. For example, if moving a mug from the sliding cabinet, or put mug into the microwave, both the sliding cabinet door and the microwave door are already open.

Can you do the same for the following objectes:
"""

def generate_kitchen_task(existing_response=None, temperature_dict=None, model_dict=None, meta_path="generated_tasks"):
    # send the object articulation tree, semantics file and get task descriptions, invovled objects and joints
    assets_dir = 'data/assets/dataset'
    object_list = ["microwave", "slidecabinet", "mug"]
    object_scene = "kitchen"
    task_user_contents_filled = copy.deepcopy(task_user_contents)
    
    articulation_tree_filled_list = []
    semantics_filled_list = []
    for item in object_list:
        articulation_tree_path = f"{assets_dir}/{item}/body_and_joint.txt"
        with open(articulation_tree_path, 'r') as f:
            articulation_tree = f.readlines()
        
        semantics = f"{assets_dir}/{item}/semantics.txt"
        with open(semantics, 'r') as f:
            semantics = f.readlines()

        articulation_tree_filled = """\n```{} articulation tree \n{} \n```\n""".format(item, "".join(articulation_tree))
        
        semantics_filled = """\n```{} semantics \n{} \n```\n""".format(item, "".join(semantics))

        articulation_tree_filled_list.append(articulation_tree_filled)
        semantics_filled_list.append(semantics_filled)
        
        task_user_contents_filled = task_user_contents_filled + articulation_tree_filled + semantics_filled

    other_conditions = "\nThe mug is already placed in the slidecabinet. Please do not generate the task: 'Place the mug into the slide cabinet'.\n"
    task_user_contents_filled = task_user_contents_filled + other_conditions


    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = "data/{}/{}_{}".format(meta_path, object_scene, time_string)
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
    if task_number == 0:
        print("Failling to generate kitchen tasks, please run script again!")
        return

    object_scene = "microwave, slidecabinet, mug"
    additional_object = "none"

    all_config_paths = []
    for task_idx in range(task_number):
        task_name = task_names[task_idx]
        task_description = task_descriptions[task_idx]
        additional_object = additional_objects[task_idx]
        involved_bodies = bodies[task_idx]
        involved_joints = joints[task_idx]

        config_path = build_task_given_text(object_scene, task_name, task_description, additional_object, involved_bodies, involved_joints, 
                          articulation_tree_filled_list, semantics_filled_list, save_folder, temperature_dict, model_dict=model_dict)
        all_config_paths.append(config_path)

    return all_config_paths
