import cv2
import gymnasium as gym
import mujoco
import numpy as np
import os

from typing import Literal

VIS_WIDTH = 1920
VIS_HEIGHT = 1080

class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: str,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        vis_width: int = VIS_WIDTH,
        vis_height: int = VIS_HEIGHT,
        time_limit: float = float("inf"),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",    # 这里渲染器会将仿真的图像作为一个RGB数组返回
        image_obs: bool = False,
        save_video: bool =False, 
        video_path: str="./mujoco_env.mp4"
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._model.vis.global_.offwidth = vis_width
        self._model.vis.global_.offheight = vis_height
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._seed = seed
        
        self.init_qpos = self._data.qpos.copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_names = ['front_cam']
        self.image_obs = image_obs
        if self.image_obs or save_video:
            self.renderer = mujoco.Renderer(self._model, vis_height, vis_width)

        self.save_video = save_video
        if save_video:
            video_path = os.path.abspath(video_path)
            video_dir = os.path.dirname(video_path)
            os.makedirs(video_dir, exist_ok=True)
            self.video_writer = cv2.VideoWriter(video_path, 
                                                cv2.VideoWriter_fourcc(*'mp4v'), 20, (vis_width, vis_height))

        return

    def close(self):
        if self.save_video:
            self.video_writer.release()
        return

    def time_limit_exceeded(self):
        flag = self._data.time >= self._time_limit
        return flag

    def render_frame(self, camera_name):
        frame = self.cam_rgb_render(camera_name)
        if self.save_video:
            video_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(video_frame)
        return frame

    def cam_rgb_render(self, camera_name):
        self.renderer.update_scene(self._data, camera=camera_name)
        frame = self.renderer.render()
        return frame
    
    def cam_depth_render(self, camera_name):
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self._data, camera=camera_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        depth = np.uint8(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX))
        return depth
    
    def cam_seg_render(self, camera_name):
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self._data, camera=camera_name)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return seg

    def write_frame(self, camera_name='front_cam'):
        # 捕获当前帧
        self.renderer.update_scene(self._data, camera=camera_name)
        frame = self.renderer.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        return

    # Accessors.
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
