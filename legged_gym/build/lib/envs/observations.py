# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv
    from legged_gym.envs.locomotion import LeggedEnv

    ANY_ENV = Union[ArmEnv, LeggedEnv]

""" Common observation functions
"""


def dof_pos(env: "ANY_ENV", params):
    return env.robot.dof_pos - env.robot.default_dof_pos


def dof_pos_selected(env: "ANY_ENV", params):
    indices = params["dof_indices"]
    return env.robot.dof_pos[:, indices] - env.robot.default_dof_pos[:, indices]


def dof_vel(env: "ANY_ENV", params):
    return env.robot.dof_vel


def actions(env: "ANY_ENV", params):
    return env.actions


def ray_cast(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    heights = env.robot.root_pos_w[:, 2].unsqueeze(1) - 0.5 - sensor.get_data()[..., 2]
    return heights
