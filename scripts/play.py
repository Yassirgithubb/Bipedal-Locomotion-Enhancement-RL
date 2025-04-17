"""
Plays a trained policy and logs statistics.
"""

# legged-gym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, export_policy_as_jit, export_policy_as_onnx, Logger
from legged_gym.utils.joystick import ViewerCamera, Joystick
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg
from legged_gym.envs.rl_config import PPOCfg

# python
import argparse
import os
import numpy as np
import torch


from legged_gym.utils.config_utils import class_to_dict, update_class_from_dict
from legged_gym.utils.helpers import print_dict

# global settings
EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = False


def play(args: argparse.Namespace):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # type hinting for easiness
    env_cfg: LeggedEnvCfg
    train_cfg: PPOCfg
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.curriculum.terrain_levels = None
    env_cfg.observations.policy.add_noise = False
    env_cfg.robot.randomization.randomize_friction = False
    env_cfg.randomization.push_robots = False
    # env_cfg.env.enable_debug_vis = True

    #  prepare joystick and view_cam
    joystick = Joystick()
    view_cam = ViewerCamera()
    robotReference = 0
    maxXVel = 1.0
    maxYVel = 1.0
    maxYawVel = 1.0
    # env_cfg.env.episode_length_s *= 10

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module and as onnx model (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        name = "policy"
        export_policy_as_jit(ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path, filename=f"{name}.pt")
        export_policy_as_onnx(ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path, filename=f"{name}.onnx")
        print("Exported policy to: ", path)

    logger = Logger(env.dt)
    robot_index = 1  # which robot is used for logging
    joint_index = 3  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.gym.viewer.eye, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.gym.viewer.target) - np.array(env_cfg.gym.viewer.eye)
    img_idx = 0

    # env.keyboard_controller.print_options()

    for i in range(10 * int(env.max_episode_length)):
        # update the observation through teleoperation device
        if joystick.has_joystick():
            joystickInput = joystick.update()
            # Changing the observation command based on the joystick input
            obs[robotReference, 9] = joystickInput.xVel * maxXVel
            obs[robotReference, 10] = joystickInput.yVel * maxYVel
            obs[robotReference, 11] = joystickInput.yawVel * maxYawVel
            # Changing the robot reference
            if joystickInput.switchRobot:
                robotReference = int(np.floor(np.random.rand(1) * env_cfg.env.num_envs))
            # Changing the viewer camera  position
            robot_pos = env.robot.root_pos_w[robotReference].detach().cpu().numpy()
            camera_position = view_cam.get_camera_position(robot_pos, joystickInput)
            # Set the camera position
            env.gym_iface.set_camera_view(camera_position, robot_pos)

        actions = policy(obs.detach())
        obs, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.gym_iface.set_camera_view(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    "dof_pos": env.robot.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.robot.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.robot.dof_torques[robot_index, joint_index].item(),
                    "command_x": env.command_generator.get_command()[robot_index, 0].item(),
                    "command_y": env.command_generator.get_command()[robot_index, 1].item(),
                    "command_yaw": env.command_generator.get_command()[robot_index, 2].item(),
                    "base_vel_x": env.robot.root_lin_vel_b[robot_index, 0].item(),
                    "base_vel_y": env.robot.root_lin_vel_b[robot_index, 1].item(),
                    "base_vel_z": env.robot.root_lin_vel_b[robot_index, 2].item(),
                    "base_vel_yaw": env.robot.root_ang_vel_b[robot_index, 2].item(),
                    "contact_forces_z": env.robot.net_contact_forces[robot_index, env.robot.feet_indices, 2]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    args = get_args()
    play(args)
