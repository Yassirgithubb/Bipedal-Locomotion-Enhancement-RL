from legged_gym.utils.config_utils import configclass
from legged_gym.common.terrains.terrain_cfg import TerrainCfg
from legged_gym.envs.locomotion.legged_env_config import (
    LeggedEnvCfg,
    RandomizationCfg,
    RewardsCfg,
    ObservationsCfg,
    CurriculumCfg,
)
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
from legged_gym.common.assets.robots import anymal_c_robot_cfg
from legged_gym.common.sensors.sensors_cfg import AnymalCSensors
from legged_gym.common.commands.command_cfg import UnifromVelocityCommandCfg
import legged_gym.envs.locomotion.rewards as R


@configclass
class AnymalCFlatEnvCfg(LeggedEnvCfg):

    robot = anymal_c_robot_cfg
    sensors = AnymalCSensors()

    # Overwrite some parameters for flat terrain
    terrain = TerrainCfg(mesh_type="plane")
    observations = ObservationsCfg(policy=ObservationsCfg.Policy(height_scan=None))
    rewards = RewardsCfg(
        flat_orientation={"func": R.flat_orientation, "scale": -5.0},
        torques={"func": R.torques, "scale": -0.000025},
        feet_air_time={"func": R.feet_air_time, "scale": 2.0, "time_threshold": 0.5},
    )
    commands = UnifromVelocityCommandCfg(heading_command=False, resampling_time=4.0)
    curriculum = CurriculumCfg(terrain_levels=None)


@configclass
class AnymalCFlatPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="flat_anymal_c",
        load_run=-1,
        max_iterations=300,
        # empirical_normalization=False
    )
    policy: PolicyCfg = PolicyCfg(actor_hidden_dims=[128, 128, 128], critic_hidden_dims=[128, 128, 128])
    # algorithm: Algorithm = Algorithm(
    #     entropy_coef= 0.
    # )


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("anymal_c_flat", LeggedEnv, AnymalCFlatEnvCfg, AnymalCFlatPPOCfg)
