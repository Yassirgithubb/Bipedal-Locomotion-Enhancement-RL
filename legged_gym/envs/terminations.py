import torch
import numpy
# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv
    from legged_gym.envs.locomotion import LeggedEnv

    ANY_ENV = Union[ArmEnv, LeggedEnv]

"""
Common termitation checking functions
"""

def illegal_contact(env: "ANY_ENV", params):
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1) > 1.0,
        dim=1,
    )



def bad_orientation(env: "ANY_ENV", params):
    print(torch.acos(env.robot.projected_gravity_b[:, 2]) )
    return torch.acos(env.robot.projected_gravity_b[:, 2]) > params["limit_angle"]




def torque_limit(env: "ANY_ENV", params):
    return torch.any(torch.isclose(env.robot.des_dof_torques, env.robot.dof_torques), dim=1)


def dof_pos_limit(env: "ANY_ENV", params):
    out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0) + (
        env.dof_pos - env.dof_pos_limits[:, 1]
    ).clip(min=0.0)
    return torch.any(out_of_limits > 1.0e-6, dim=1)
