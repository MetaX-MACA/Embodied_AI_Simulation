# Installation

Recommended runtime environment:

OS: Ubuntu 22.04

Python: 3.10

PyTorch >= 2.0

GPU: MetaX C500

Driver/SDK Version: 2.31.0.4 or higger

RAM: >= 128GB

## Install MetaX C500 Driver and SDK
1. Go to the [MetaX Developer Center](https://sw-developer.metax-tech.com/member.php?mod=register) to register an account.

2. Download MetaX C500 [Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85) and [SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85), version: 2.31.0.4 or higger. Please download the local install version.

3. Follow the instructions on the webpage to complete the installation.

4. Update `.bashrc` file.
    
    ```
    vi ~/.bashrc
    ```
    Add the following environment variables
    ```shell
    export MACA_PATH=/opt/maca
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
    export PATH=${MACA_PATH}/bin:${MACA_PATH}/tools/cu-bridge/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_CLANG_PATH}:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
    export MXLOG_LEVEL=err
    ```
    Update environment variables
    ```
    source ~/.bashrc
    ```
5. Add current user to video group.

    ```shell
    sudo usermod -aG video $USER
    newgrp video
    ```

6. Reboot your server.

7. You can use the `mx-smi` command to check GPU information.

## Using Conda(Suggested)
### Create python environment
``` shell
# create conda environment
conda create -n sim_mujoco python=3.10
conda activate sim_mujoco
```

### Download PyTorch and vLLM from MetaX Developer Center
**Note** Please download the version that matches the Driver, such as `2.31.x.x`.

PyTorch: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)


You will receive two tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.4-py310-2.31.0.4-x86_64.tar.xz
cd 2.31.0.4/wheel/
pip install ./*.whl
```

### Install python packages
``` shell
cd ./embodied_ai_simulation_for_mujoco
pip install -r requirements.txt
```

## Additional required documents
Since the current project leverages the multi-objective optimization-based reinforcement learning algorithm from Stage-Wise-CMORL, please follow these steps:
``` shell
# 1. Clone the Stage-Wise-CMORL repository
git clone https://github.com/rllab-snu/Stage-Wise-CMORL

# 2. Copy the required files
cp ./main_teacher_mujoco.py ./Stage-Wise-CMORL/
cp ./utils/wrappermujoco.py ./Stage-Wise-CMORL/utils/
cp ./tasks/__init__.py ./Stage-Wise-CMORL/tasks/
cp ./tasks/go1_backflip_mujoco.py ./Stage-Wise-CMORL/tasks/
cp ./tasks/go1_backflip_mujoco.yaml ./Stage-Wise-CMORL/tasks/
cp ./assets/go1/xml/go1_correct.xml ./Stage-Wise-CMORL/assets/go1/xml/
cp ./algos/comoppo/go1_backflip_mujoco.yaml ./Stage-Wise-CMORL/algos/comoppo/

```

## Rendering
We use the `Mujoco` to render the simulation results.Since most server configurations lack display outputs, use the following command for off-screen rendering.
``` shell
apt-get install libosmesa6-dev
export MUJOCO_GL=osmesa
```