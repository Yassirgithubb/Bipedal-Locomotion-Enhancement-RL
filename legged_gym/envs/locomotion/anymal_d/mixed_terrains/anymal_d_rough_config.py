from legged_gym.utils.config_utils import configclass
from legged_gym.common.sensors.sensors_cfg import RaycasterCfg, BpearlPatternCfg, RealSensePatternCfg
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg, SensorsCfg
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
from legged_gym.common.assets.robots import anymal_d_robot_cfg
from legged_gym.common.sensors.sensors_cfg import AnymalDSensors


@configclass
class AnymalDRoughEnvCfg(LeggedEnvCfg):

    robot = anymal_d_robot_cfg
    sensors = AnymalDSensors()


@configclass
class AnymalDRoughPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="rough_anymal_d",
        load_run=-1,
        # empirical_normalization=False
    )


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("anymal_d_rough", LeggedEnv, AnymalDRoughEnvCfg, AnymalDRoughPPOCfg)
