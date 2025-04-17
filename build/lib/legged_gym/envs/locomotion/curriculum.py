import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv

from legged_gym.envs.curriculum import *


"""
Curriculum scorer functions for the locomotion environments.
"""


def terrain_levels(env: "LeggedEnv", env_ids, params):
    """If the robot walked more than half the terrain length, it moves to a harder level.
    Else if it walked less than half of the distance required by the commanded velocity, it goes to a simpler level"""
    distance = torch.norm(env.robot.root_pos_w[env_ids, :2] - env.terrain.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > env.terrain.cfg.terrain_length / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (
        distance < torch.norm(env.command_generator.get_command()[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down *= ~move_up
    # update terrain levels
    env.terrain.update_terrain_levels(env_ids, move_up, move_down)


def terrain_tracking(env: "LeggedEnv", env_ids, params):
    """If the robot tracks the velocity commands, it moves to a harder level else it goes to a simpler level"""
    avg_tracking_error = (
        torch.sum(env.command_generator.tracking_error_sum[env_ids, :], dim=1)
        / env.command_generator.log_step_counter[env_ids]
        / 3
    )
    move_up = avg_tracking_error < 0.3
    move_down = avg_tracking_error > 0.7
    move_down *= ~move_up
    env.terrain.update_terrain_levels(env_ids, move_up, move_down)


def max_lin_vel_command(env: "LeggedEnv", env_ids, params):
    # TODO test
    """If the tracking reward is above <params[reward_threshold]% of the maximum, increase the range of commands"""
    mean_tracking_reward = (
        torch.mean(env.reward_manager.episode_sums["tracking_lin_vel"][env_ids]) / env.max_episode_length
    )
    if (
        mean_tracking_reward
        > params["reward_threshold"] * env.reward_manager.reward_params["tracking_lin_vel"]["scale"]
    ):
        if "x" in params["axis"]:
            # min range: (-max_curriculum, 0)
            env._command_ranges.lin_vel_x[0] = np.clip(
                env._command_ranges.lin_vel_x[0] - params["increment_x"], params["max_range_x"][0], 0.0
            )
            # max range: (0, -max_curriculum)
            env._command_ranges.lin_vel_x[1] = np.clip(
                env._command_ranges.lin_vel_x[1] + params["increment_x"], 0.0, params["max_range_x"][1]
            )
        if "y" in params["axis"]:
            # min range: (-max_curriculum, 0)
            env._command_ranges.lin_vel_y[0] = np.clip(
                env._command_ranges.lin_vel_y[0] - params["increment_y"], params["max_range_y"][0], 0.0
            )
            # max range: (0, -max_curriculum)
            env._command_ranges.lin_vel_y[1] = np.clip(
                env._command_ranges.lin_vel_y[1] + params["increment_y"], 0.0, params["max_range_y"][1]
            )

    # some more possible examples
    # def torque_reward_scale()
    # def ray_caster_noise()
    # def feet_friction() # if it can be set at runtime
    # def base_mass() # if it can be set at runtime
