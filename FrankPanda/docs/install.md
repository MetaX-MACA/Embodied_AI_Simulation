# Installation

Recommended runtime environment:

OS: Ubuntu 22.04

Python: 3.10

PyTorch: 2.4

GPU: MetaX C500

Driver/SDK Version: 2.31.0.4 or higger

RAM: >=64GB

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
### Install dependencies
``` shell
sudo apt-get update
sudo apt-get install libosmesa6-dev -y
```

### Create python environment
``` shell
# create conda environment
conda create -n sim_panda_mujoco python=3.10
conda activate sim_panda_mujoco
```

### Download PyTorch and vLLM from MetaX Developer Center
**Note** Please download the version that matches the Driver, such as `2.31.x.x`.

PyTorch: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&ai_label=Pytorch)

vLLM: [link](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm&ai_label=vLLM)

You will receive two tar archives. After extracting them, navigate to the `wheel` directory and install using `pip`.
``` shell
# install PyTorch
tar -xvf maca-pytorch2.4-py310-2.31.0.4-x86_64.tar.xz
cd 2.31.0.4/wheel/
pip install ./*.whl

# install vLLM
tar -xvf mxc500-vllm-py310-2.31.0.4-linux-x86_64.tar.xz
cd mxc500-vllm-2.31.0.4/wheel/
pip install ./*.whl
```

### Install python packages
``` shell
cd ./FrankPanda
pip install -r requirements.txt
```

### Install Open Motion Planning Library

We use [Open Motion Planning Library (OMPL)](https://ompl.kavrakilab.org/) for motion planning as part of the pipeline to solve the generated task.

Note: if you are having trouble building OMPL from source, the maintainer of OMPL has suggested to use the prebuilt python wheels at [here](https://github.com/ompl/ompl/releases/tag/prerelease). Use the wheel that matches your python version, e.g., if you are using python3.10, download [this wheel](https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl).Then run the pip command in your python environment to install ompl.

```shell
pip install ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

To install OMPL from source, the reference document at: https://ompl.kavrakilab.org/installation.html , installation steps are here:

``` shell
# download the install script
wget https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh
chmod +x install-ompl-ubuntu.sh

# install dependencies
sudo apt-get install doxygen libflann-dev -y

# modify line 88 of install-ompl-ubuntu.sh and replace it with your Python path from the Conda environment
cmake ../.. -DPYTHON_EXEC=~/miniconda3/envs/sim_panda_mujoco/bin/python${PYTHONV}

# install OMPL with Python bindings
./install-ompl-ubuntu.sh --python

# copy ompl folder to your python environment
cp -r ./ompl-1.6.0/py-bindings/ompl ~/miniconda3/envs/sim_panda_mujoco/lib/python3.10/site-packages/
```
