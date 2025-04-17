# python
import torch
import numpy as np

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

from legged_gym.envs.locomotion.curriculum import *


def reward_curriculum(env: "PackageGrabbingEnv", env_ids, params):
    # scale shaping rewards linearly from min_mult at start_it to max_mult at end_it (iterations)
    factor = np.clip(
        params["min_mult"]
        + (params["max_mult"] - params["min_mult"])
        * np.ceil((env.common_step_counter / 24) - params["start_it"])
        / (params["end_it"] - params["start_it"]),
        params["min_mult"],
        params["max_mult"],
    )
    # scale factor with env_dt since scale_cur does not get scaled in reward_manager init
    factor *= env.dt
    # rewards that are scaled
    env.cfg.rewards.height["scale"] = env.cfg.rewards.height["scale_cur"] * factor
    env.cfg.rewards.stand_straight["scale"] = env.cfg.rewards.stand_straight["scale_cur"] * factor
    env.cfg.rewards.straight_knees["scale"] = env.cfg.rewards.straight_knees["scale_cur"] * factor
    env.cfg.rewards.shoulder_symmetry["scale"] = env.cfg.rewards.shoulder_symmetry["scale_cur"] * factor

    env.cfg.rewards.torques["scale"] = env.cfg.rewards.torques["scale_cur"] * factor
    env.cfg.rewards.dof_acc["scale"] = env.cfg.rewards.dof_acc["scale_cur"] * factor
    env.cfg.rewards.dof_vel["scale"] = env.cfg.rewards.dof_vel["scale_cur"] * factor
    env.cfg.rewards.action_rate["scale"] = env.cfg.rewards.action_rate["scale_cur"] * factor
