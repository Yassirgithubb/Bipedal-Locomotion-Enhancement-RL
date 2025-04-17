import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv

from legged_gym.envs.observations import *


""" Locomotion specific observation functions"""


def projected_gravity(env: "LeggedEnv", params):
    return env.robot.projected_gravity_b


def base_lin_vel(env: "LeggedEnv", params):
    return env.robot.root_lin_vel_b


def base_ang_vel(env: "LeggedEnv", params):
    return env.robot.root_ang_vel_b


def velocity_commands(env: "LeggedEnv", params):
    return env.command_generator.get_command()
