# legged-gym
from typing import Dict, List, Tuple
from legged_gym.common.sensors.sensors_cfg import RaycasterCfg, GridPatternCfg, BpearlPatternCfg
from legged_gym.utils import configclass
from legged_gym.envs.base_env_config import BaseEnvCfg, EnvCfg, ControlCfg
from legged_gym.common.terrains.terrain_cfg import TerrainCfg
from legged_gym.common.assets.robots import ManipulatorCfg
from legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg
import legged_gym.envs.manipulation.observations as O
import legged_gym.envs.manipulation.rewards as R
import legged_gym.envs.manipulation.terminations as T
import legged_gym.envs.manipulation.curriculum as C


@configclass
class CommandsCfg:
    curriculum = False
    max_curriculum = 1.0
    num_commands = 3  # default: pos_x, pos_y, pos_z in env frame
    resampling_time = 10.0  # time before commands are changed [s]

    @configclass
    class Ranges:
        pos_x = (0.0, 0.6)  # min max [m/s]
        pos_y = (-0.6, 0.6)  # min max [m/s]
        pos_z = (0.0, 0.6)  # min max [rad/s]

    ranges = Ranges()


@configclass
class RandomizationCfg:
    push_robots = False


@configclass
class ObservationsCfg:
    @configclass
    class Policy:
        # optinal parameters: scale, clip([min, max]), noise
        add_noise: bool = True  # turns off the noise in all observations
        ee_pos_commands: dict = {"func": O.ee_pos_commands}
        dof_pos: dict = {"func": O.dof_pos, "noise": 0.01}
        dof_vel: dict = {"func": O.dof_vel, "noise": 1.5}
        actions: dict = {"func": O.actions}

    policy = Policy()


@configclass
class RewardsCfg:
    # general params
    only_positive_rewards: bool = False
    # reward functions
    termination = {"func": R.termination, "scale": -0.0}
    ee_tracking = {"func": R.ee_tracking, "scale": 1.0, "sigma": 0.2}
    torques = {"func": R.torques, "scale": -0.00003}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-8}
    collision = {"func": R.collision, "scale": -1.0, "bodies": ".*(THIGH|SHANK)"}
    action_rate = {"func": R.action_rate, "scale": -0.05}
    dof_vel = {"func": R.dof_vel, "scale": -0.0}


@configclass
class TerminationsCfg:
    # general params
    illegal_contact = {"func": T.illegal_contact, "bodies": "base"}
    bad_orientation = None
    dof_torque_limit = None
    dof_pos_limit = None


@configclass
class CurriculumCfg:
    # general params
    curriculum = None


@configclass
class ArmEnvCfg(BaseEnvCfg):

    # common configuration (from base env)
    env = EnvCfg(num_envs=4096, num_actions=6, send_timeouts=True, episode_length_s=6, enable_debug_vis=False)
    gym = GymInterfaceCfg(viewer=ViewerCfg(eye=(10, 0, 6), target=(11, 5, 3)))
    control = ControlCfg(decimation=4, action_scale=0.5, action_clipping=100.0)

    # legged-env specific configurations
    # -- scene designing
    terrain = TerrainCfg()
    robot = ManipulatorCfg()
    # sensors = SensorsCfg()
    # -- command processing
    commands = CommandsCfg()
    # -- mdp signals
    randomization = RandomizationCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
