# Overview

This section introduces how to transfer complex skills from Isaac Gym to the MuJoCo simulator, providing best practices for MuJoCo-based embodied AI researchers. The process includes four main steps.

Asset alignment → API replacement  → Action refinement → Motion decomposition

Taking the backflip of a quadruped robot as an example, we detail the key considerations when migrating from Isaac Gym to MuJoCo, following the four steps outlined above

## 1. Asset alignment

The primary goal is to ensure that the robot model (e.g., MJCF/XML) is reconstructed in MuJoCo, matching the dimensions, mass, joint limits, and other physical properties used in Isaac Gym.

```shell
# The project provides robot models for Isaac Gym and MuJoCo, with details as follows:
Issac Gym robot model: ./Stage-Wise-CMORL/assets/go1/urdf/go1.urdf
MuJoCo robot model: ./Stage-Wise-CMORL/assets/go1/xml/go1.xml
```

However, when using the provided XML for training the quadruped robot's skills in MuJoCo, it was found that the corresponding skills could not be learned. The main reason is that the provided URDF and XML are not equivalent at the model level, with the XML lacking corresponding components.
The main differences are as follows:

### Absence of key components

The primary difference is the absence of key components of the quadruped robot. In the URDF, the robot has four legs, each consisting of hip, thigh, calf, and foot. However, in the XML, only hip, thigh, and calf are included, with the foot missing. This leads to anomalies when obtaining foot contact forces. Therefore, it is necessary to add the following to the corresponding XML:

```shell
# Add one foot to the quadruped robot
<body name="RL_foot" pos="0 0 -0.213">
    <inertial pos="0 0 0" mass="0.06" diaginertia="9.6e-06 9.6e-06 9.6e-06" />
</body>
```

### Incorrect attribute values

Set the correct attribute values in the XML, note that in Isaac Gym, model attribute values are dynamically set during the creation of the simulation (e.g., create_sim). In contrast, MuJoCo requires these values to be predefined in the XML. For example, the foot attributes in Isaac Gym, corresponding to self.friction_coeffs=1 and self.restitution_coeffs=0, need to be configured in MuJoCo's XML as follows:

```shell
# Set the correct attribute values in the XML
<body name="RL_foot" pos="0 0 -0.213">
    <inertial pos="0 0 0" mass="0.06" diaginertia="9.6e-06 9.6e-06 9.6e-06" />
    <geom name="RL_foot_geom" size="0.02" type="sphere" solref="0.01 20" solimp="0.6 0.95 0.001" contype="1" conaffinity="1" rgba="0 0 0 1" />
</body>
```

### Better visual demonstration

Modify floor physical properties to achieve superior visual demonstration:

```shell
 <asset>
    <texture name="plane" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.4 0.4 0.4" width="512" height="512"/>
    <material name="plane" reflectance="0.5" texture="plane" texrepeat="10 10" texuniform="true"/>
</asset>
<geom name='floor' type='plane' conaffinity='1' condim='3' contype='1' material='plane' pos='0 0 0' size='0 0 1'/>
```

We provide the corrected XML configuration file at: Go1/assets/go1/xml/go1_correct.xml

## 2. API replacement

Replace Isaac Gym’s simulation and control APIs with their MuJoCo counterparts, including functions for state reading, action application, and simulation stepping. The following lists the main API correspondence replacements, with details provided in **tasks/go1_backflip_mujoco.py**


| **Isaac Gym API**                                     | **Mujoco API**                                                                                                                                                          | **Function Description**                                              |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `self.gym.refresh_dof_state_tensor(self.sim)`         | `self.dof_positions=self.data.qpos[7:]` `self.dof_velocities=self.data.qvel[6:]`                                                                                       | `Obtain joint positions and velocities`                               |
| `self.gym.refresh_dof_force_tensor(self.sim)`         | `self.dof_torques=self.data.ctrl`                                                                                                                                       | `Obtain joint torques`                                                |
| `self.gym.refresh_actor_root_state_tensor(self.sim)`  | `self.base_positions=self.data.qpos[0:3]` `self.base_quaternions=self.data.qpos[3:7]` `self.base_lin_vel=self.data.qvel[0:3]` `self.base_ang_vel=self.data.qvel[3:6]` | `Obtain base position, quaternion, linear/angular velocity`           |
| `self.gym.refresh_net_contact_force_tensor(self.sim)` | `self.contact_forces= self._extract_contact_force(i)`                                                                                                                   | `Obtain rigid body contact forces`                                    |
| `self.gym.refresh_rigid_body_state_tensor(self.sim)`  | `torso_pos=self.data.xpos[self.trunk_id]` `torso_quat=self.data.xquat[self.trunk_id]` `torso_vel=self.data.qvel[0:3]` `torso_ang_vel=self.data.qvel[3:6]`              | `Obtain rigid body positions, quaternions, linear/angular velocities` |

## 3. Action refinement

Integrating strong prior knowledge into specific tasks can significantly accelerate skill acquisition. For instance, enforcing bilateral symmetry constraints on the hind legs during a robotic dog's backflip not only reduces learning complexity but also enhances stability.

```python
 actions[:, 3] = -actions[:, 0]  # FR_hip = -FL_hip
actions[:, 9] = -actions[:, 6]  # RR_hip = -RL_hip
actions[:, [1, 4]] = actions[:, [1, 4]].mean(dim=1, keepdim=True)  # FL_thigh = FR_thigh
actions[:, [2, 5]] = actions[:, [2, 5]].mean(dim=1, keepdim=True)  # FL_calf = FR_calf
actions[:, [7, 10]] = actions[:, [7, 10]].mean(dim=1, keepdim=True)  # RL_thigh = RR_thigh
actions[:, [8, 11]] = actions[:, [8, 11]].mean(dim=1, keepdim=True)  # RL_calf = RR_calf
```

## 4. Motion decomposition

Decomposing complex motor skills into multiple execution stages simplifies the task into a combination of elementary skills at each stage. Subsequent policy training via multi-objective optimization algorithms ultimately enables the acquisition of complex skills.More details see [Stage-Wise-CMORL](https://github.com/rllab-snu/Stage-Wise-CMORL)
