# isaac-gym
from isaacgym.torch_utils import torch_rand_float, to_torch, get_axis_params, quat_rotate_inverse
from isaacgym import gymapi

# python
import torch
from torch import Tensor

# legged-gym
from legged_gym.common.gym_interface import GymInterface
from legged_gym.common.assets.robots.articulation import Articulation
from legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from legged_gym.common.assets.robots.manipulators.manipulator import Manipulator
from .legged_mobile_manipulators_cfg import LeggedMobileManipulatorCfg


class LeggedMobileManipulator(LeggedRobot, Manipulator):
    def __init__(self, cfg: LeggedMobileManipulatorCfg, num_envs: int, gym_iface: GymInterface) -> None:
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg
