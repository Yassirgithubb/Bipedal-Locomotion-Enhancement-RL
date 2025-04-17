import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv
    from legged_gym.envs.locomotion import LeggedEnv

    ANY_ENV = Union[ArmEnv, LeggedEnv]

"""
Common reward Functions
"""


def torques(env: "ANY_ENV", params):
    # Penalize torques
    return torch.sum(torch.square(env.robot.dof_torques), dim=1)


def torques_selected(env: "ANY_ENV", params):
    # Penalize torques on selected dofs
    return torch.sum(torch.square(env.robot.dof_torques[:, params["dof_indices"]]), dim=1)


def motor_torques(env: "ANY_ENV", params):
    # Penalize motor torques
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques), dim=1)


def motor_torques_selected(env: "ANY_ENV", params):
    # Penalize motor torques on selected dofs
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques[:, params["dof_indices"]]), dim=1)


def power(env: "ANY_ENV", params):
    # Penalize mechanical power (joints torques*velocities)
    # "std" defines the width of the bel curve
    powers = env.robot.dof_torques*env.robot.dof_vel
    clip_powers = torch.clamp(powers, min=0)
    power = torch.sum(clip_powers, dim=1)
    return torch.exp(-power / params["std"])


def dof_vel(env: "ANY_ENV", params):
    # Penalize dof velocities
    return torch.sum(torch.square(env.robot.dof_vel), dim=1)


def dof_acc(env: "ANY_ENV", params):
    # Penalize dof accelerations
    return torch.sum(torch.square(env.robot.dof_acc), dim=1)


def action_rate(env: "ANY_ENV", params):
    # Penalize changes in actions
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)


def collision(env: "ANY_ENV", params):
    # Penalize collisions on selected bodies
    return torch.sum(
        torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1) > 1.0,
        dim=1,
    )


def termination(env: "ANY_ENV", params):
    # Terminal reward / penalty
    return env.reset_buf * ~env.time_out_buf


def dof_pos_limits(env: "ANY_ENV", params):
    # Penalize dof positions too close to the limit. The soft limit is computed by the env
    out_of_limits = -(env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
    out_of_limits += (env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def dof_vel_limits(env: "ANY_ENV", params):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.dof_vel) - env.robot.soft_dof_vel_limits * params["soft_ratio"]).clip(min=0.0, max=1.0),
        dim=1,
    )


def torque_limits(env: "ANY_ENV", params):
    # penalize torques too close to the limit
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.des_dof_torques) - env.robot.soft_dof_torque_limits * params["soft_ratio"]).clip(min=0.0),
        dim=1,
    )


def contact_forces(env: "ANY_ENV", params):
    # penalize high contact forces
    return torch.sum(
        (torch.norm(env.robot.net_contact_forces, dim=-1) - params["max_contact_force"]).clip(min=0.0),
        dim=1,
    )
