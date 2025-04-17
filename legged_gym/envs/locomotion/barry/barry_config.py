from legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, SimParamsCfg
from legged_gym.common.terrains.terrain_cfg import SubTerrainsCfg, TerrainCfg
from legged_gym.envs.base_env_config import ControlCfg
from legged_gym.envs.locomotion.legged_env_config import (
    CurriculumCfg,
    CustomSubTerrainsCfg,
    LeggedEnvCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from legged_gym.common.assets.robots import barry
from legged_gym.utils.config_utils import configclass
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
import legged_gym.common.observations.observation_manager as O
import legged_gym.envs.locomotion.rewards as R
import legged_gym.envs.locomotion.terminations as T
from legged_gym.common.commands.command_cfg import UnifromVelocityCommandCfg
import legged_gym.envs.locomotion.curriculum as C


@configclass
class BarryRewards:
    only_positive_rewards = False
    tracking_lin_vel = {"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25}
    tracking_ang_vel = {"func": R.tracking_ang_vel, "scale": 0.5, "std": 0.25}
    termination = {"func": R.termination, "scale": -100.0}
    flat_orientation = {"func": R.flat_orientation, "scale": -1.0}  # or 0. or -10.?
    motor_torques_hip = {
        "func": R.motor_torques_selected,
        "scale": -0.0003 / 0.67,
        "dofs": ".*(HAA|HFE)",
    }  # scaled by motor constant to get winding losses
    motor_torques_knee = {
        "func": R.motor_torques_selected,
        "scale": -0.0003 / 1.1,
        "dofs": ".*KFE",
    }  # scaled by motor constant to get winding losses
    dof_vel = {"func": R.dof_vel, "scale": -0.001}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-7}
    lin_vel_z = {"func": R.lin_vel_z, "scale": -1.0}
    ang_vel_xy = {"func": R.ang_vel_xy, "scale": -0.025}
    dof_pos_limits = {"func": R.dof_pos_limits, "scale": -100.0}
    # contact_forces = {"func_name": "contact_forces", "scale": -0.01, "max_contact_force": 1000.}
    dof_vel_limits = {
        "func": R.dof_vel_limits,
        "scale": -0.1,
        "soft_ratio": 0.9,
    }
    feet_air_time = {"func": R.feet_air_time, "scale": 2.0, "time_threshold": 0.5}
    action_rate = {"func": R.action_rate, "scale": -0.01}
    collision = {"func": R.collision, "scale": -1.0, "bodies": ".*(THIGH|SHANK).*"}
    # stand_still = {"func_name": "stand_still", "scale": -0.5}
    # base_height = {"func_name": "base_height", "scale": -5.0, "height_target": 0.7}
    # torque_limits: {"func_name": "torque_limits", "scale": -0.0001, "soft_ratio": 0.9}


@configclass
class BarryRoughCfg(LeggedEnvCfg):
    terrain = TerrainCfg(
        sub_terrains=CustomSubTerrainsCfg(
            hf_pyramid_slope=SubTerrainsCfg.HfPyramidSlopeCfg(proportion=0.1, max_slope=0.6, platform_size=2.0),
            hf_pyramid_slope_inv=SubTerrainsCfg.HfPyramidSlopeInvCfg(proportion=0.1, max_slope=0.6, platform_size=2.0),
        )
    )
    robot = barry

    rewards = BarryRewards()

    control = ControlCfg(action_scale=0.35, decimation=8)

    terminations = TerminationsCfg(dof_pos_limit=None)  # enable for retraining ?

    gym = GymInterfaceCfg(sim_params=SimParamsCfg(dt=0.0025))


@configclass
class BarryFlatCfg(BarryRoughCfg):
    terrain = TerrainCfg(mesh_type="plane")
    observations = ObservationsCfg(policy=ObservationsCfg.Policy(height_scan=None))
    curriculum = CurriculumCfg(terrain_levels=None)
    commands = UnifromVelocityCommandCfg(resampling_time=4.0, heading_command=False)


@configclass
class BarryRoughPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="rough_barry",
        load_run=-1,
        max_iterations=3000,
        # policy_class_name='ActorCriticRecurrent'
    )

    # algorithm = AlgorithmCfg(
    #     entropy_coef=0.005
    # )

    policy = PolicyCfg(
        init_noise_std=0.75,
        # rnn_type = "lstm",
        # rnn_hidden_size = 512,
        # rnn_num_layers = 1
    )


@configclass
class BarryFlatPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="flat_barry",
        load_run=-1,
        max_iterations=800,
        # policy_class_name='ActorCriticRecurrent'
    )

    algorithm = AlgorithmCfg(entropy_coef=0.005)

    policy = PolicyCfg(
        init_noise_std=0.75,
        # rnn_type = "lstm",
        # rnn_hidden_size = 512,
        # rnn_num_layers = 1
    )


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("barry_rough", LeggedEnv, BarryRoughCfg, BarryRoughPPOCfg)
task_registry.register("barry_flat", LeggedEnv, BarryFlatCfg, BarryFlatPPOCfg)
