import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv

from legged_gym.envs.observations import *

""" Manipulation specific observation functions"""


def ee_pos_commands(env: "ArmEnv", params):
    return env.commands[:, :3]
