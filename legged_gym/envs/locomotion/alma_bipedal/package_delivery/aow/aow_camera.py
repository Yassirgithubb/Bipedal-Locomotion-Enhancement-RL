# isaac-gym
from isaacgym.torch_utils import quat_rotate, quat_mul

# python
import torch
from torch import Tensor
import re

# legged-gym
from legged_gym.common.gym_interface import GymInterface
from legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.aow.aow_camera_cfg import AowCameraCfg


class AowCamera(LeggedRobot):

    camera_pos_w: Tensor = None
    """ Position of the camera (Tensor), shape=(num_envs, 3), view of rigid_body_states"""

    camera_quat_w: Tensor = None
    """ Orientation of the camera (Tensor), shape=(num_envs, 4), view of rigid_body_states"""

    package_in_fov: Tensor = None
    """ Boolean tensor indicating if package is in fov of camera (Tensor), shape=(num_envs)"""

    def __init__(self, cfg: AowCameraCfg, num_envs: int, gym_iface: GymInterface) -> None:
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg

    def init_buffers(self):
        super().init_buffers()

        # create tensors from cfg
        self.camera_pos = torch.tensor(self.cfg.camera_pos, device=self.device).repeat((self.num_envs, 1))
        self.camera_rot = torch.tensor(self.cfg.camera_rot, device=self.device).repeat((self.num_envs, 1))

        # compute absolute camera pose
        self.camera_pos_w = self.root_pos_w + quat_rotate(self.root_quat_w, self.camera_pos)
        self.camera_quat_w = quat_mul(self.root_quat_w, self.camera_rot)

        # init tensor for camera checking
        self.package_in_fov = torch.empty(self.num_envs, device=self.device)

    def update_buffers(self, dt: float, env_ids=None):
        super().update_buffers(dt, env_ids)

        # compute absolute camera pose
        self.camera_pos_w = self.root_pos_w + quat_rotate(self.root_quat_w, self.camera_pos)
        self.camera_quat_w = quat_mul(self.root_quat_w, self.camera_rot)
