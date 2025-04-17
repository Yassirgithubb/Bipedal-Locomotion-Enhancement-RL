# python
from typing import Tuple

# legged-gym
from legged_gym.utils.config_utils import configclass
from legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg


@configclass
class AowCameraCfg(LeggedRobotCfg):
    camera_pos: Tuple[float, float, float] = (0.415, 0.0, -0.17)  # relative position of camera to robot
    camera_rot: Tuple[float, float, float, float] = (
        0.7032,
        -0.7032,
        -0.0739,
        0.0739,
    )  # relative orientation of camera to robot
    camera_fov: Tuple[float, float] = (
        90.0,
        82.5,
    )  # (110.0, 82.5) camera fov in horizontal and vertical direction in deg
