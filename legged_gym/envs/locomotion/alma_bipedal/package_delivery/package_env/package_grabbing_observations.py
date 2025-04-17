# python
import torch

# isaac-gym
from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

from legged_gym.envs.locomotion.observations import *


""" Package specific observation functions"""


def velocity_commands_x_yaw(env: "PackageGrabbingEnv", params):
    # only return x and yaw and ignore y command
    commands = env.command_generator.get_command()
    return commands[:, [0, 2]]


def camera_to_package_c(env: "PackageGrabbingEnv", params):
    vec_w = env.package.root_pos_w - env.robot.camera_pos_w
    vec_c = quat_rotate_inverse(env.robot.camera_quat_w, vec_w)
    if params["fov_check"]:
        return vec_c * env.robot.package_in_fov[:, None]
    else:
        return vec_c
    
def camera_to_package_quat(env: "PackageGrabbingEnv", params):
    quat = quat_mul(quat_conjugate(env.robot.camera_quat_w),env.package.root_quat_w)
    if params["fov_check"]:
        return quat * env.robot.package_in_fov[:, None]
    else:
        return quat

def camera_to_table_c(env: "PackageGrabbingEnv", params):
    #vec_w = env.table.root_pos_w - env.robot.camera_pos_w
    vec_w =  env.robot.camera_pos_w
    vec_c = quat_rotate_inverse(env.robot.camera_quat_w, vec_w)
    return vec_c

def package_velocity_c(env: "PackageGrabbingEnv", params):
    vec_w = env.package.root_lin_vel_w
    vec_c = quat_rotate_inverse(env.robot.camera_quat_w, vec_w)
    if params["fov_check"]:
        return vec_c * env.robot.package_in_fov[:, None]
    else:
        return vec_c
