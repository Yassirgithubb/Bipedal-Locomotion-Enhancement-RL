from legged_gym.utils import configclass
from legged_gym.envs.locomotion.legged_env_config import *
from legged_gym.envs.base_env_config import EnvCfg, ControlCfg
from legged_gym.common.commands.command_cfg import NormalVelocityCommandCfg
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg
import legged_gym.envs.locomotion.observations as O
import legged_gym.envs.locomotion.rewards as R
import legged_gym.envs.locomotion.terminations as T
import legged_gym.envs.locomotion.curriculum as C
from legged_gym.common.assets.robots import anymal_wheels_robot_cfg


@configclass
class WheeledLeggedControlCfg(ControlCfg):
    """Control configuration for stepping the environment."""

    wheel_action_scale: float = 1.0
    """Scaling of wheel input actions provided to the environment."""


@configclass
class AnymalWheelsBaseRoughRewardsCfg:
    # general params
    only_positive_rewards: bool = True
    # reward functions
    termination = {"func": R.termination, "scale": -0.0}
    tracking_lin_vel = {"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25}
    tracking_ang_vel = {"func": R.tracking_ang_vel, "scale": 0.5, "std": 0.25}
    lin_vel_z = {"func": R.lin_vel_z, "scale": -2.0}
    ang_vel_xy = {"func": R.ang_vel_xy, "scale": -0.05}
    torques = {"func": R.torques, "scale": -0.000008}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-7}
    collision = {"func": R.collision, "scale": -1.0, "bodies": ".*(THIGH|SHANK)"}
    action_rate = {"func": R.action_rate, "scale": -0.01}
    torque_limits = {"func": R.torque_limits, "scale": -0.00001, "soft_ratio": 0.95}


@configclass
class CommandCfg(NormalVelocityCommandCfg):
    @configclass
    class Ranges(NormalVelocityCommandCfg.Ranges):
        mean_vel: Tuple = (1.0, 0.8, 0.8)  # linear x, linear y, angular yaw [m/s]
        std_vel: Tuple = (0.5, 0.4, 0.4)  # linear x, linear y, angular yaw [m/s]

    ranges = Ranges()


@configclass
class WheeledLeggedObservationsCfg(ObservationsCfg):
    @configclass
    class Policy:
        # optinal parameters: scale, clip([min, max]), noise
        add_noise: bool = True  # turns off the noise in all observations
        base_lin_vel: dict = {"func": O.base_lin_vel, "noise": 0.1}
        base_ang_vel: dict = {"func": O.base_ang_vel, "noise": 0.2}
        projected_gravity: dict = {"func": O.projected_gravity, "noise": 0.05}
        velocity_commands: dict = {"func": O.velocity_commands}
        dof_pos: dict = {"func": O.dof_pos_selected, "noise": 0.01, "dofs": ".*(HAA|HFE|KFE)"}
        dof_vel: dict = {"func": O.dof_vel, "noise": 1.5}
        actions: dict = {"func": O.actions}
        height_scan: dict = {"func": O.ray_cast, "noise": 0.1, "sensor": "height_scanner", "clip": (-1, 1.0)}

    policy = Policy()


# @configclass
# class TerminationsCfg(TerminationsCfg):
#     # general params
#     illegal_contact = {
#         "func": T.illegal_contact,
#         "bodies": ["base", ".*THIGH", ".*SHANK"],
#     }


@configclass
class CurriculumCfg:
    # general params
    terrain_levels = {"func": C.terrain_tracking}
    max_lin_vel_command = None


@configclass
class WheeledLeggedEnvCfg(LeggedEnvCfg):

    # common configuration (from base env)
    env = EnvCfg(num_envs=4096, num_actions=16, send_timeouts=True, episode_length_s=20, enable_debug_vis=False)
    control = WheeledLeggedControlCfg(decimation=4, action_scale=0.3, action_clipping=100.0)

    # legged-env specific configurations
    # -- scene designing
    robot = anymal_wheels_robot_cfg
    # -- command processing
    commands = CommandCfg()
    # -- mdp signals
    observations = WheeledLeggedObservationsCfg()
    rewards = AnymalWheelsBaseRoughRewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()


@configclass
class AnymalWheelsRoughPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="rough_anymal_wheels_base",
        load_run=-1,
        empirical_normalization=True,
        max_iterations=5000,
    )
    algorithm: AlgorithmCfg = AlgorithmCfg(entropy_coef=0.002)


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.anymal_wheels.wheeled_legged_env import WheeledLeggedEnv

task_registry.register("anymal_wheels_rough", WheeledLeggedEnv, WheeledLeggedEnvCfg, AnymalWheelsRoughPPOCfg)
