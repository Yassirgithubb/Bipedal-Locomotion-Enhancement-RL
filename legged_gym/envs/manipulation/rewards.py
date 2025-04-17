import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv

from legged_gym.envs.rewards import *

"""
Manipulation specific reward Functions
"""


def ee_tracking(env: "ArmEnv", params):
    # penalize tracking error of end effector
    return torch.exp(
        -torch.norm(env.robot.ee_pos_b - env.commands[:, :3], dim=1) / params["sigma"]
    )  # track the first EE
