import argparse
import os
import random

from gpt.prompts.prompt_kitchen_manipulation import generate_kitchen_task
from execute import Executer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action="store_true", default=False)
    parser.add_argument('--gui', action="store_true", default=False, 
                        help="Default False, means using cpu to render image, because c500 not support rendering.")
    args = parser.parse_args()

    if not args.gui:
        os.environ["MUJOCO_GL"] = "osmesa"

    temperature_dict = {
        "task_generation": 0.6,
        "reward": 0.2,
        "yaml": 0.3,
        "joint": 0,
    }

    model_dict = {
        "task_generation": "Qwen2.5-Coder-14B-Instruct",
        "reward": "Qwen2.5-Coder-14B-Instruct",
        "yaml": "Qwen2.5-Coder-14B-Instruct",
        "joint": "Qwen2.5-Coder-14B-Instruct",
    }

    ### generate task, return config path
    meta_path = "generated_tasks_release"
    if not os.path.exists("data/{}".format(meta_path)):
        os.makedirs("data/{}".format(meta_path))

    # generate task
    all_task_config_paths = generate_kitchen_task(temperature_dict=temperature_dict, model_dict=model_dict, meta_path=meta_path)
    
    print("-------------- all_task_config_paths --------------")
    for item in all_task_config_paths:
        print(item)
    print("---------------------------------------------------")


    if args.execute:
        task_num = len(all_task_config_paths)
        # random choose one task to execute
        idx = random.randint(0, task_num-1)
        task_config_path = all_task_config_paths[idx]
        print("trying to learn skill: ", task_config_path)
        try:
            ## run RL for each substep
            executer = Executer(task_config_path=task_config_path, gui=args.gui)
            executer.run()
            del executer

        except Exception as e:
            print("=" * 20, "an error occurred", "=" * 20)
            print("an error occurred: ", e)
            print("=" * 20, "an error occurred", "=" * 20)
            print("failed to execute task: ", task_config_path)