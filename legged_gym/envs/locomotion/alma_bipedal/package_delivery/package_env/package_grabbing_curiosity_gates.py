# python
import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

from legged_gym.envs.curiosity_gates import *

""" Package specific curiosity gates"""


def package_state(env: "PackageGrabbingEnv", params):
    # returns package velocity in world frame
    return env.package.root_lin_vel_w
