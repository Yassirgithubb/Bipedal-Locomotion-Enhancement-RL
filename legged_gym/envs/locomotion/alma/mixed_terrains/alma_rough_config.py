from legged_gym.utils.config_utils import configclass
from legged_gym.envs.base_env_config import EnvCfg
from legged_gym.envs.locomotion.anymal_c.mixed_terrains.anymal_c_rough_config import (
    AnymalCRoughEnvCfg,
    AnymalCRoughPPOCfg,
)
from legged_gym.common.assets.robots import alma_robot_cfg
from legged_gym.envs.locomotion.legged_env_config import RewardsCfg
from legged_gym.envs.rl_config import RunnerCfg
import legged_gym.envs.locomotion.rewards as R


@configclass
class AlmaRoughEnvCfg(AnymalCRoughEnvCfg):
    robot = alma_robot_cfg
    env = EnvCfg(num_envs=4096, num_actions=18, send_timeouts=True, episode_length_s=12, enable_debug_vis=False)
    rewards = RewardsCfg(
        tracking_lin_vel={"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25},
        tracking_ang_vel={"func": R.tracking_ang_vel, "scale": 0.5, "std": 0.25},
        lin_vel_z={"func": R.lin_vel_z, "scale": -0.2},
        ang_vel_xy={"func": R.ang_vel_xy, "scale": -0.01},
        torques={"func": R.torques, "scale": -0.0000002},
        dof_acc={"func": R.dof_acc, "scale": -2.5e-8},
        feet_air_time={"func": R.feet_air_time, "scale": 0.5, "time_threshold": 0.5},
        collision={"func": R.collision, "scale": -0.1, "bodies": ".*(THIGH|SHANK)"},
        action_rate={"func": R.action_rate, "scale": -0.01},
    )


@configclass
class AlmaRoughPPOCfg(AnymalCRoughPPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="rough_alma",
        load_run=-1,
        max_iterations=1500,
    )


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("alma_rough", LeggedEnv, AlmaRoughEnvCfg, AlmaRoughPPOCfg)
