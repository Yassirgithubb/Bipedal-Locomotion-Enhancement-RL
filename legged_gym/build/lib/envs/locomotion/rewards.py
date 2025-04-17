import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv

from legged_gym.envs.rewards import *

"""
Locomotion specific reward Functions
"""


def lin_vel_z(env: "LeggedEnv", params):
    # Penalize z axis base linear velocity
    return torch.square(env.robot.root_lin_vel_b[:, 2])


def ang_vel_xy(env: "LeggedEnv", params):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(env.robot.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, :2]), dim=1)


def vertical_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, 1:]), dim=1)


def base_height(env: "LeggedEnv", params):
    # Penalize base height away from target
    base_height = torch.mean(env.robot.root_pos_w[:, 2].unsqueeze(1) , dim=1)
    return torch.square(base_height - params["height_target"])

def link_linear_velocity(env: "ANY_ENV", params):
     # Penalize link velocities
    
     return torch.sum(torch.square( env.robot.rigid_body_states[:, params["body_indices"], [7,8,9]]), dim=1)

def tracking_lin_vel(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.sum(
        torch.square(env.command_generator.get_command()[:, :2] - env.robot.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / params["std"])

def tracking_lin_vel_standing(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.square(env.command_generator.get_command()[:, 0]- env.robot.root_lin_vel_b[:, 0])
    
    return torch.exp(-lin_vel_error / params["std"])

def tracking_ang_vel(env: "LeggedEnv", params):
    # Tracking of angular velocity commands (yaw)
    # "std" defines the width of the bel curve
    ang_vel_error = torch.square(env.command_generator.get_command()[:, 2] - env.robot.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / params["std"])

def tracking_ang_vel_standing(env: "LeggedEnv", params):
    # Tracking of angular velocity commands (yaw) is the roll angle of the base 
    # "std" defines the width of the bel curve
    #ang_vel_error = torch.square(env.command_generator.get_command()[:, 0] - env.robot.root_ang_vel_b[:, 0])
    z_ang_tensor = torch.reshape( env.robot.rigid_body_states[:, [26], 12],(-1,))
    ang_vel_error = torch.square(env.command_generator.get_command()[:, 0] -  z_ang_tensor)
    return torch.exp(-ang_vel_error / params["std"])


def feet_air_time(env: "LeggedEnv", params):
    # Reward long steps
    first_contact = env.robot.feet_last_air_time > 0.0
    reward = torch.sum((env.robot.feet_last_air_time - params["time_threshold"]) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_generator.get_command()[:, :2], dim=1) > 0.1
    return reward


def no_fly(env: "LeggedEnv", params):
    # Penalize jumps
    contacts = env.robot.net_contact_forces[:, env.robot.feet_indices, 2] > 0.1
    single_contact = torch.sum(1.0 * contacts, dim=1) == 1
    return 1.0 * single_contact


def stumble(env: "LeggedEnv", params):
    # Penalize feet hitting vertical surfaces
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, env.robot.feet_indices, :2], dim=2)
        > params["hv_ratio"] * torch.abs(env.robot.net_contact_forces[:, env.robot.feet_indices, 2]),
        dim=1,
    )


def stand_still(env: "LeggedEnv", params):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * (
        torch.norm(env.command_generator.get_command()[:, :2], dim=1) < 0.1
    )


def foot_slippage(env: "LeggedEnv", params):
    # penalize foot velocity while in contact with the ground
    return torch.sum((torch.norm(env.robot.feet_velocities, dim=2) * env.robot.contact), dim=1)