from legged_gym.utils.config_utils import configclass
from legged_gym.common.terrains.terrain_cfg import TerrainCfg
from legged_gym.envs.manipulation.arm_env_config import (
    CommandsCfg,
    ArmEnvCfg,
    RewardsCfg,
    ObservationsCfg,
)
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
from legged_gym.common.assets.robots import dynaarm_robot_cfg
import legged_gym.common.rewards.reward_manager as R


@configclass
class DynaarmEnvCfg(ArmEnvCfg):

    robot = dynaarm_robot_cfg

    # Overwrite some parameters for flat terrain
    terrain = TerrainCfg(mesh_type="plane")
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    commands = CommandsCfg(resampling_time=3.0)


@configclass
class DynaarmPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="", experiment_name="dynaarm", load_run=-1, max_iterations=3000, empirical_normalization=False
    )
    policy: PolicyCfg = PolicyCfg(actor_hidden_dims=[128, 128, 128], critic_hidden_dims=[128, 128, 128])


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.manipulation.arm_env import ArmEnv

task_registry.register("dynaarm", ArmEnv, DynaarmEnvCfg, DynaarmPPOCfg)
