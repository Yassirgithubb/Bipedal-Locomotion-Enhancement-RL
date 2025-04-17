from isaacgym import torch_utils

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

from legged_gym.envs.terminations import *


def not_standing_angle(env: "PackageGrabbingEnv", params):
    return (
        torch.acos(torch_utils.quat_rotate(env.robot.root_quat_w, env.robot._forward_vec_b)[:, 2])
        > params["limit_angle"]
    )


def not_standing_height(env: "PackageGrabbingEnv", params):
    return env.robot.root_pos_w[:, 2] < params["limit_height"]


def torque_limit_timed(env: "PackageGrabbingEnv", params):
    # improved torque limit termination starting only after init_time seconds
    init_passed = env.episode_length_buf > params["init_time"] / env.dt
    joint_torque_vel_limit = torch.any(
        ~torch.isclose(
            env.robot.des_dof_torques[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]],
            env.robot.dof_torques[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]],
        ),
        dim=1,
    )
    # wheel joints are handled differently due to actuator model
    if params["include_wheels"]:
        wheel_torque_limit = torch.any(torch.abs(env.robot.dof_torques[:, [3, 7, 11, 15]]) > 35, dim=1)
        wheel_vel_limit = torch.any(torch.abs(env.robot.dof_vel[:, [3, 7, 11, 15]]) > 50, dim=1)
        limit_exceeded = (joint_torque_vel_limit + wheel_torque_limit + wheel_vel_limit) > 0
        return init_passed * limit_exceeded
    else:
        return init_passed * joint_torque_vel_limit


def moved_away_from_package(env: "PackageGrabbingEnv", params):
    return env.package_to_goal_dist > params["limit_dist"]
