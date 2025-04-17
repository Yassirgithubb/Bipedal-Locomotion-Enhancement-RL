from legged_gym.utils.config_utils import configclass
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
from legged_gym.common.assets.robots import anymal_b_robot_cfg


@configclass
class AnymalBRoughEnvCfg(LeggedEnvCfg):

    robot = anymal_b_robot_cfg


@configclass
class AnymalBRoughPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="rough_anymal_b",
        load_run=-1,
        # empirical_normalization=False
    )


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("anymal_b_rough", LeggedEnv, AnymalBRoughEnvCfg, AnymalBRoughPPOCfg)
