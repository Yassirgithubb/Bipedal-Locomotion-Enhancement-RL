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
def illegal_contact_standing(env: "ANY_ENV", params):
    packaged_dropped = numpy.array(torch.any(env.package.rigid_body_states[:, :, 2] < 0.33,dim=1,).numpy())
    knees_height_violation= numpy.array(torch.any(env.robot.rigid_body_states[:, [0, 6, 9, 17, 20, 2 ,13], 2] < 0.2,dim=1,).numpy())
    contact_forces_violation= numpy.array( torch.any(torch.norm(env.robot.net_contact_forces[:,  params["body_indices"], :], dim=-1) > 0.1,dim = 1).numpy())
    violation = torch.tensor(contact_forces_violation | knees_height_violation)
    return violation

def bad_orientation_bipedal(env: "ANY_ENV", params):
    upper_orientation_violation = torch.acos(env.robot.projected_gravity_b[:, 2]) > params["limit_angle_upper"]
    lower_orientation_violation = torch.acos(env.robot.projected_gravity_b[:, 2]) < params["limit_angle_lower"]
    return ( upper_orientation_violation| lower_orientation_violation)