import copy
import os
import yaml

from gpt.prompts.prompt_manipulation_reward_primitive import decompose_and_generate_reward_or_primitive
from gpt.prompts.prompt_set_joint_angle import query_joint_angle
from gpt.qwen_query import qwen_query
from gpt.generate_mujoco_scene import generate_mujoco_scene_xml

task_yaml_config_prompt = """
I need you to describe the initial scene configuration for a given task in the following format, using a yaml file. This yaml file will help build the task in a simulator. The task is for a mobile Franka panda robotic arm to learn a manipulation skill in the simulator. The Franka panda arm is mounted on a table, at location (0, 0, 1.6). The z axis is the gravity axis. 

The format is as follows:
```yaml 
- use_table: whether the task requires using a table. This should be decided based on common sense. If a table is used, its location will be fixed at (0.5, -0.6, 0). The height of the table will be 1.6m. Usually, if the objects invovled in the task are usually placed on a table (not directly on the ground), then the task requires using a table.
# for each object involved in the task, we need to specify the following fields for it.
- type: mesh
  name: name of the object, so it can be referred to in the simulator
  lang: this should be a language description of the mesh. The language should be a concise description of the obejct, such that the language description can be used to search an existing database of objects to find the object.
  center: the location of the object center. If there is a table in the task and the object needs to be placed on the table, this center should be in table coordinate range, where (0.6, 0.4, 1.6) is the left bottom corner of the table, and (0.85, 0.2, 1.6) is the right top corner of the table. In either case, you should try to specify a location such that there is no collision between objects. For slidecabinet, the center in yaml should be in the range of (0.82, 0.2, 2.3) and (0.82, -0.4, 2.3). For mug, the center in yaml should be in the slidecabinet's range, and has a coordinate offset (-0.22, 0.24, -0.159). For example, if the center of slidecabinet is (0.82 0.16 2.3), the center should be (0.6 0.4 2.141).
  movable: if the object is movable or not in the simulator due to robot actions. This option should be falsed for most tasks; it should be true only if the task specifically requires the robot to move the object. 
```

For slidecabinet, the center in yaml should be in the range of (0.82, 0.2, 2.3) and (0.82, -0.6, 2.3). For mug, the center in yaml should be in the slidecabinet's range, and has a coordinate offset (-0.22, 0.24, -0.159). For example, if the center of slidecabinet is (0.82 0.16 2.3), the center should be (0.6 0.4 2.141).

An example input includes the task names, task descriptions, and objects involved in the task. I will also provide with you the articulation tree and semantics of the articulated object. 
This can be useful for knowing what parts are already in the articulated object, and thus you do not need to repeat those parts as separate objects in the yaml file.

Your task includes two parts:
1. Output the yaml configuration of the task.
2. Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 

Example input:
Task Name: Insert Bread Slice 
Description: The robotic arm will insert a bread slice into the toaster.
Objects involved: Toaster, bread slice. Only the objects specified here should be included in the yaml file.

```Toaster articulation tree
bodies: 
base
body_0
body_1
body_2
body_3
body_4
body_5

joints: 
joint_name: joint_0 joint_type: hinge parent_body: body_5 child_body: body_0
joint_name: joint_1 joint_type: slide parent_body: body_5 child_body: body_1
joint_name: joint_2 joint_type: slide parent_body: body_5 child_body: body_2
joint_name: joint_3 joint_type: slide parent_body: body_5 child_body: body_3
joint_name: joint_4 joint_type: slide parent_body: body_5 child_body: body_4
joint_name: joint_5 joint_type: none parent_body: base child_body: body_5
```

```Toaster semantics
body_0 hinge knob
body_1 slide slider
body_2 slide button
body_3 slide button
body_4 slide button
body_5 hinge toaster_body
```


An example output:
```yaml
- use_table: True ### Toaster and bread are usually put on a table. 
- type: mesh
  name: "Toaster"
  center: (0.8, 0.4, 1.6) # Remember that when an object is placed on the table, the center is expressed in the table coordinate range, where (0.6, 0.4, 1.6) is the left bottom corner and (0.85, 0.2, 1.6) is the right top corner of the table. Here we put the toaster near the right top corner of the table.  
  lang: "a common toaster"
  movable: False
- type: mesh
  name: "bread slice"
  center: (0.6, 0.2, 1.6) # Remember that when an object is placed on the table, the center is expressed in the table coordinate range, where (0.6, 0.4, 1.6) is the left bottom corner and (0.85, 0.2, 1.6) is the right top corner of the table. Here we put the bread slice near the left corner of the table.  
  lang: "a slice of bread"
  movable: True
```

Another example input:
Task Name: Removing Lid From Pot
Description: The robotic arm will remove the lid from the pot.
Objects involved: KitchenPot. Only the objects specified here should be included in the yaml file.

```KitchenPot articulation tree
bodies: 
base
body_0
body_1

joints: 
joint_name: joint_0 joint_type: slide parent_body: body_1 child_body: body_0
joint_name: joint_1 joint_type: none parent_body: base child_body: body_1
```

```KitchenPot semantics
body_0 slider lid
body_1 free pot_body
```
Output:
```yaml
- use_table: True # A kitchen pot is usually placed on the table.
- type: mesh
  name: "KitchenPot"
  center: (0.6, 0.2, 1.6) # Remember that when an object is placed on the table, the center is expressed in the table coordinate range, where (0.6, 0.4, 1.6) is the left bottom corner and (0.85, 0.2, 1.6) is the right top corner of the table. Here we put the kitchen pot just at a random location on the table.  
  lang: "a common kitchen pot"
  movable: True
```
Note in this example, the kitchen pot already has a lid from the semantics file. Therefore, you do not need to include a separate lid in the yaml file.

Another example:
Task Name: Put an item into the box drawer
Description: The robot will open the drawer of the box, and put an item into it.
Objects involved: A box with drawer, an item to be placed in the drawer. 

```Box articulation tree
bodies: 
base
body_0
body_1
body_2

joints: 
joint_name: joint_0 joint_type: hinge parent_body: body_2 child_body: body_0
joint_name: joint_1 joint_type: slide parent_body: body_2 child_body: body_1
joint_name: joint_2 joint_type: none parent_body: base child_body: body_2
```

```Box semantics
body_0 hinge rotation_lid
body_1 slide drawer
body_2 free box_body
```

Output:
```yaml
-   use_table: true
-   center: (0.8, 0.2, 1.6)
    lang: "a wooden box"
    name: "Box"
    type: mesh
    movable: False
-   center: (0.6, 0.4, 1.6)
    lang: "A toy" # Note here, we changed the generic/placeholder "item" object to be a more concrete object: a toy. 
    name: "Item"
    type: mesh
    movable: True
```

Rules: 
- You do not need to include the robot in the yaml file.
- The yaml file should only include the objects listed in "Objects involved".
- Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 
- Do not to create multiple configurations for a object in yaml. For example, for a mug object, you should not create mug1, mug2 in yaml.


Can you do this for the following task:
Task Name: {}
Description: {}
Objects involved: {}
"""

def find_flag(src_str, target_str_list):
    flag = False
    for item in target_str_list:
        if src_str.lower().startswith(item):
            flag = True
            break
    return flag

def parse_task_response(task_response):
    task_names = []
    task_descriptions = []
    additional_objects = []
    bodies = []
    joints = []

    task_str_list = ["task name:", "### task name:", "#### task name:", "** task name:", "**task name:"]
    joint_str_list = ["joints:", "**joints:", "** joints:"]

    task_response = task_response.split("\n")
    for l_idx, line in enumerate(task_response):
        if find_flag(line.lower(), task_str_list):
            task_name = line.split(":")[1].strip()
            task_name = task_name.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace("*", "")
            task_names.append(task_name)
            task_description = task_response[l_idx+1].split(":")[1].strip()
            task_description = task_description.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace(")", ".").replace("(", ".").replace("*", "")
            task_descriptions.append(task_description)
            additional_object = task_response[l_idx+2].split(":")[1].strip()
            additional_object = additional_object.replace(".", "").replace("'", "").replace('"', "").replace(")", ".").replace("(", ".").replace("*", "")
            additional_objects.append(additional_object)
            involved_bodies = ""
            for body_idx in range(l_idx+4, len(task_response)):
                if find_flag(task_response[body_idx].lower(), joint_str_list):
                    break
                else:
                    involved_bodies += (task_response[body_idx][2:]).replace("'", "").replace('"', "").replace(")", ".").replace("(", ".").replace("*", "")
            bodies.append(involved_bodies)
            involved_joints = ""
            for joint_idx in range(body_idx+1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    involved_joints += (task_response[joint_idx][2:]).replace("'", "").replace('"', "").replace(")", ".").replace("(", ".").replace("*", "")
            joints.append(involved_joints)

    return task_names, task_descriptions, additional_objects, bodies, joints

def parse_response_to_get_yaml(response, task_description):
    yaml_string = []
    for l_idx, line in enumerate(response):
        if "```yaml" in line:
            for l_idx_2 in range(l_idx + 1, len(response)):
                if response[l_idx_2].lstrip().startswith("```"):
                    break

                yaml_string.append(response[l_idx_2])

            yaml_string = '\n'.join(yaml_string)
            description = f"{task_description}".replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")
            save_name =  description + '.yaml'

            parsed_yaml = yaml.safe_load(yaml_string)

            return parsed_yaml, save_name

def yaml_post_process(parsed_yaml, object_category):
    # NOTE: post-process such that articulated object is xml.
    for obj in parsed_yaml:
        if "name" in obj and (obj['name'].lower() == object_category or obj['name'].lower() in object_category ):
            obj['type'] = 'xml'
        
        # 手动过滤重复创建的资产
        if "name" in obj:
            if obj['name'].lower() == "mug1":
                obj['name'] = 'mug'
            elif obj['name'].lower() in ["mug2", "mug3"]:
                parsed_yaml.remove(obj)

    slide_pos = None
    for obj in parsed_yaml:
        #  确保 slidecabinet 的坐标范围在 (0.82, 0.2, 2.3) and (0.82, -0.5, 2.3)
        if "name" in obj and obj['name'].lower() == "slidecabinet":
            center = obj['center'].replace("(", "").replace(")", "")
            pos = [float(x) for x in center.split(",")]
            pos[0] = 0.82
            pos[-1] = 2.3
            if pos[1] < -0.5:
                pos[1] = -0.5
            if pos[1] > 0.2:
                pos[1] = 0.2
            obj['center'] = f"({pos[0]}, {pos[1]}, {pos[2]})"
            slide_pos = pos
        #  确保 microwave 的坐标高度为1.6
        elif "name" in obj and obj['name'].lower() == "microwave":
            center = obj['center'].replace("(", "").replace(")", "")
            pos = [float(x) for x in center.split(",")]
            pos[-1] = 1.6
            obj['center'] = f"({pos[0]}, {pos[1]}, {pos[2]})"
    
    # 确保马克杯在slidecabinet内
    offset = [-0.22, 0.24, -0.159]
    for obj in parsed_yaml:
        if "name" in obj and obj['name'].lower() == "mug":
            pos = []    
            for i in range(3):
                pos.append(slide_pos[i]+offset[i])
            obj['center'] = f"({pos[0]}, {pos[1]}, {pos[2]})"
            break

    return parsed_yaml

def build_task_given_text(object_category, task_name, task_description, additional_object, involved_bodies, involved_joints, 
                          articulation_tree_filled_list, semantics_filled_list, save_folder, temperature_dict, model_dict=None):
    ### generate the task yaml config file in the simulator.
    task_yaml_config_prompt_filled = copy.deepcopy(task_yaml_config_prompt)
    if additional_object.lower() == "none":
        task_object = object_category
    else:
        task_object = "{}, {}".format(object_category, additional_object)

    task_yaml_config_prompt_filled = task_yaml_config_prompt_filled.format(task_name, task_description, task_object)

    for i in range(len(articulation_tree_filled_list)):
        task_yaml_config_prompt_filled += articulation_tree_filled_list[i]
        task_yaml_config_prompt_filled += semantics_filled_list[i]

    system = "You are a helpful assistant."
    save_path = os.path.join(save_folder, "gpt_response/task_yaml_config_{}.json".format(task_name))
    print("=" * 50)
    print("=" * 20, "generating task yaml config", "=" * 20)
    print("=" * 50)
    task_yaml_response = qwen_query(system, [task_yaml_config_prompt_filled], [], save_path=save_path, debug=False, 
                            temperature=temperature_dict["yaml"], model=model_dict["yaml"])

    description = f"{task_name}_{task_description}".replace(" ", "_").replace(".", "").replace(",", "")
    task_yaml_response = task_yaml_response.split("\n")
    parsed_yaml, save_name = parse_response_to_get_yaml(task_yaml_response, description)

    parsed_yaml = yaml_post_process(parsed_yaml, object_category)

    config_path = save_folder
    task_config_path = os.path.join(config_path, save_name)
    with open(task_config_path, 'w') as f:
        yaml.dump(parsed_yaml, f, indent=4)

    initial_config = yaml.safe_dump(parsed_yaml)

    ### decompose and generate reward
    yaml_file_path = task_config_path
    reward_save_path = os.path.join(save_folder, "gpt_response/reward_{}.json".format(task_name))
    print("=" * 50)
    print("=" * 20, "generating reward", "=" * 20)
    print("=" * 50)
    solution_path = decompose_and_generate_reward_or_primitive(task_name, task_description, initial_config, 
                                                                articulation_tree_filled_list, semantics_filled_list, 
                                                                involved_bodies, involved_joints, yaml_file_path, save_path=reward_save_path,
                                                                temperature=temperature_dict["reward"],
                                                                model=model_dict["reward"])
    

    ### generate joint angle
    save_path = os.path.join(save_folder, "gpt_response/joint_angle_{}.json".format(task_name))
    substep_file_path = os.path.join(solution_path, "substeps.txt")
    with open(substep_file_path, 'r') as f:
        substeps_list = f.readlines()
    substeps = ''.join(substeps_list)
    print("=" * 50)
    print("=" * 20, "generating initial joint angle", "=" * 20)
    print("=" * 50)
    joint_angle_values = query_joint_angle(task_name, task_description, articulation_tree_filled_list, semantics_filled_list, 
                                            involved_bodies, involved_joints, substeps, save_path=save_path, 
                                            temperature=temperature_dict['joint'], model=model_dict["joint"])

    config = yaml.safe_load(initial_config)

    config.append(dict(solution_path=solution_path))
    config.append(joint_angle_values)
    config.append(dict(task_name=task_name, task_description=task_description))

    # 根据场景配置创建mujoco场景xml文件
    scene_xml_path = generate_mujoco_scene_xml(config, task_config_path, task_name)
    config.append(dict(scene_xml_path=scene_xml_path))

    with open(task_config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
    with open(os.path.join(solution_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f, indent=4)

    return task_config_path