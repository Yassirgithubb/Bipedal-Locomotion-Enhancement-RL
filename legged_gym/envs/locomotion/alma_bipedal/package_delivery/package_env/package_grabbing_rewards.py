from torch import pi
from isaacgym import torch_utils
import math

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

from legged_gym.envs.locomotion.rewards import *

###############################
# related to package grabbing #
###############################


def package_lifted(env: "PackageGrabbingEnv", params):
    # reward packages that are lifted high enough and do not interact with the table
    return env.package_lifted


def tracking_lin_vel_standing(env: "PackageGrabbingEnv", params):
    # reward tracking of x velocity command
    #lin_vel_error = torch.square(abs(env.command_generator.get_command()[:, 0]) - env.robot.root_lin_vel_b[:, 0])
    lin_vel_error = torch.square(env.robot.root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_error / params["std"]) * env.package_lifted


def tracking_ang_vel_standing(env: "PackageGrabbingEnv", params):
    # reward tracking of yaw velocity command
    ang_vel_error = torch.square(env.command_generator.get_command()[:, 2] - env.robot.root_ang_vel_b[:, 0])
    return torch.exp(-ang_vel_error / params["std"]) * env.package_lifted


#######################
# related to standing #
#######################


def stand(env: "PackageGrabbingEnv", params):

    lf_wheel_hight = env.robot.rigid_body_states[:, 4, 2]
    rf_wheel_hight = env.robot.rigid_body_states[:, 12, 2]

    l_min_one_hand_on_ground = torch.logical_or(lf_wheel_hight < 0.2, rf_wheel_hight < 0.2).int()

    angle_x_world_z = torch.acos(
        torch.clamp(
            torch_utils.quat_rotate(env.robot.root_quat_w, env.robot._forward_vec_b)[:, 2], min=-1 + 1e-9, max=1 - 1e-9
        )
    )

    # Reward upright base
    r_angle = ((pi / 2 - angle_x_world_z) / (pi / 2)) * 0.2
    # Reward high base
    r_height_z = env.robot.root_states[:, 2] * 3
    # Penalize ground contact
    r_ground_contact = l_min_one_hand_on_ground * 2
    # Penalize wheel speed
    r_wheel_speed = (
        (torch.square(env.robot.dof_vel[:, 3]) + torch.square(env.robot.dof_vel[:, 11]))
        * 0.003
        * (1 - l_min_one_hand_on_ground)
    )

    reward = torch.clamp(r_angle + r_height_z - r_ground_contact - r_wheel_speed, min=0)
    return reward


def height(env: "PackageGrabbingEnv", params):
    # Reward a high robot center
    base_height = torch.mean(env.robot.root_pos_w[:, 2].unsqueeze(1) , dim=1)
    return torch.square(base_height - params["height_target"])


def stand_straight(env: "PackageGrabbingEnv", params):
    # Reward an upright base
    angle_x_world_z = torch.acos(
        torch.clamp(
            torch_utils.quat_rotate(env.robot.root_quat_w, env.robot._forward_vec_b)[:, 2], min=-1 + 1e-9, max=1 - 1e-9
        )
    )
    return (pi / 2 - angle_x_world_z) / (pi / 2)


def shoulder_symmetry(env: "PackageGrabbingEnv", params):
    # Penalize if shoulder not in default configuration
    return torch.sum(torch.square(env.robot.dof_pos[:, [0, 4, 8, 12]]), dim=1)


def stand_no_move(env: "PackageGrabbingEnv", params):
    # Reward no movement of the base
    reward = torch.exp(
        -(
            torch.sum(torch.square(env.robot.root_lin_vel_w), dim=1)
            + torch.sum(torch.square(env.robot.root_ang_vel_w[:, 2:3]), dim=1)
        )
        * 25
    )
    return reward


def hind_shoulder_symmetry(env: "PackageGrabbingEnv", params):
    # Penalize if hind shoulders not in default configuration
    return torch.sum(torch.square(env.robot.dof_pos[:, [4, 12]]), dim=1)


def straight_knees(env: "PackageGrabbingEnv", params):
    # Reward straight knees
    return torch.exp(-1 * torch.sum(torch.square(env.robot.dof_pos[:, [6, 14]]), dim=1))
