# isaac-gym
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, quat_mul, quat_conjugate

# python
import torch
import numpy as np

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv


def draw_camera(env: "PackageGrabbingEnv", i):
    num_lines = 12
    line_vertices = torch.empty((num_lines * 2, 3))
    line_colors = torch.empty((num_lines, 3))

    # plot camera to package vector
    line_vertices[0, :] = env.package.root_pos_w[i, ...]
    line_vertices[1, :] = env.robot.camera_pos_w[i, ...]

    # draw fov
    lz = 0.5  # length of camera cone in visualization
    endx = lz * torch.tan(torch.deg2rad(torch.tensor(env.robot.cfg.camera_fov[0] / 2)))
    endy = lz * torch.tan(torch.deg2rad(torch.tensor(env.robot.cfg.camera_fov[1] / 2)))
    endPoints = torch.tensor(
        [
            [endx, endy, lz],
            [-endx, endy, lz],
            [-endx, -endy, lz],
            [endx, -endy, lz],
        ]
    )
    line_vertices[2:10:2, :] = env.robot.camera_pos_w[i, ...]
    line_vertices[3, :] = line_vertices[2, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), endPoints[0].unsqueeze(0)
    )
    line_vertices[5, :] = line_vertices[2, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), endPoints[1].unsqueeze(0)
    )
    line_vertices[7, :] = line_vertices[2, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), endPoints[2].unsqueeze(0)
    )
    line_vertices[9, :] = line_vertices[2, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), endPoints[3].unsqueeze(0)
    )

    line_vertices[10, :] = line_vertices[3, :]
    line_vertices[11, :] = line_vertices[5, :]
    line_vertices[12, :] = line_vertices[5, :]
    line_vertices[13, :] = line_vertices[7, :]
    line_vertices[14, :] = line_vertices[7, :]
    line_vertices[15, :] = line_vertices[9, :]
    line_vertices[16, :] = line_vertices[9, :]
    line_vertices[17, :] = line_vertices[3, :]

    # draw camera frame
    frame = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )
    line_vertices[18:23:2, :] = env.robot.camera_pos_w[i, ...]

    line_vertices[19, :] = line_vertices[20, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), frame[0].unsqueeze(0)
    )
    line_vertices[21, :] = line_vertices[20, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), frame[1].unsqueeze(0)
    )
    line_vertices[23, :] = line_vertices[20, :] + quat_rotate(
        env.robot.camera_quat_w[i, ...].unsqueeze(0).to(device="cpu"), frame[2].unsqueeze(0)
    )

    # colors
    line_colors[0, :] = torch.tensor([0, 0, 255])

    if env.robot.package_in_fov[i]:
        line_colors[1:9, :] = torch.tensor([0, 255, 0])
    else:
        line_colors[1:9, :] = torch.tensor([255, 0, 0])

    line_colors[9, :] = torch.tensor([255, 0, 0])
    line_colors[10, :] = torch.tensor([0, 255, 0])
    line_colors[11, :] = torch.tensor([0, 0, 255])

    env.gym.add_lines(env.viewer, env.envs[i], num_lines, line_vertices, line_colors)


def draw_target(env: "PackageGrabbingEnv", i):
    num_lines = 1
    line_vertices = torch.empty((num_lines * 2, 3))
    line_colors = torch.empty((num_lines, 3))

    # plot packege to goal vector
    line_vertices[0, :] = env.package.root_pos_w[i, ...]
    line_vertices[1, :] = env.package_target_pos_w[i, ...]

    if env.package_to_goal_dist[i] < 0.2:
        line_colors[0, :] = torch.tensor([0, 255, 0])
    else:
        line_colors[0, :] = torch.tensor([255, 0, 0])

    env.gym.add_lines(env.viewer, env.envs[i], num_lines, line_vertices, line_colors)
    # plot target position
    sphere(env, env.package_target_pos_w[i, ...])


def sphere(env: "PackageGrabbingEnv", rb_state_or_pos, radius=0.2, color=(1, 0, 0)):
    """Draws debug sphere at specified position.

    Args:
        rb_state_or_pos (_type_): either rigid body state (13) or position (3)
        radius (float, optional): radius of sphere to draw. Defaults to 0.2.
        color (tuple, optional): RGB color of sphere. Defaults to (1, 0, 0).
    """
    if rb_state_or_pos.shape == torch.Size([3]):
        rb_state_or_pos = torch.cat((rb_state_or_pos, torch.zeros(10, device=env.device)))
        rb_state_or_pos[6] = 1

    transform = state2transform(rb_state_or_pos)

    axes_geom = gymutil.AxesGeometry(0.1)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * 3.14, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(radius, 12, 12, sphere_pose, color=color)

    gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[0], transform)
    gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], transform)


def state2transform(state):
    r = gymapi.Vec3(state[0], state[1], state[2])
    q = gymapi.Quat(state[3], state[4], state[5], state[6])
    return gymapi.Transform(r, q)
