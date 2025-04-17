# legged-gym
from typing import Dict, List, Tuple
from legged_gym.utils import configclass
from legged_gym.envs.base_env_config import BaseEnvCfg, EnvCfg, ControlCfg
from legged_gym.common.terrains.terrain_cfg import TerrainCfg, SubTerrainsCfg
from legged_gym.common.assets.robots import LeggedRobotCfg
from legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg
from legged_gym.common.sensors.sensors_cfg import SensorsCfg
from legged_gym.common.commands.command_cfg import UnifromVelocityCommandCfg
import legged_gym.envs.locomotion.observations as O
import legged_gym.envs.locomotion.rewards as R
import legged_gym.envs.locomotion.terminations as T
import legged_gym.envs.locomotion.curriculum as C


@configclass
class CustomSubTerrainsCfg(SubTerrainsCfg):
    pyramid_stairs = SubTerrainsCfg.PyramidStairsCfg(proportion=0.25)
    pyramid_stairs_inv = SubTerrainsCfg.PyramidStairsInvCfg(proportion=0.35)
    table = SubTerrainsCfg.TableCfg(proportion=0.0)
    gap = SubTerrainsCfg.GapCfg(proportion=0.0)
    pit = SubTerrainsCfg.PitCfg(proportion=0.0)
    boxes = SubTerrainsCfg.BoxesCfg(proportion=0.2)
    # hf_discrete_obstacles = SubTerrainsCfg.HfDiscreteObstaclesCfg(proportion=0.2)
    hf_pyramid_slope = SubTerrainsCfg.HfPyramidSlopeCfg(proportion=0.1)
    hf_pyramid_slope_inv = SubTerrainsCfg.HfPyramidSlopeInvCfg(proportion=0.1)


@configclass
class RandomizationCfg:
    # randomize_friction: bool = True
    # friction_range: Tuple = (0.5, 1.25)
    # randomize_base_mass: bool = False
    # added_mass_range: Tuple = (-1.0, 1.0)
    max_init_pos = 1.0  # max xy position added to default position [m]
    max_init_yaw = 3.14  # max yaw angle added to default orientation [rad]
    max_init_roll_pitch = 0.000001  # max roll and pitch angles added to default orientation [rad]
    push_robots = True
    push_interval_s = 15  # push applied each time interval [s]
    max_push_vel = 1.0  # velocity offset added by push [m/s]
    max_external_force = 0.0  # wind force applied at base, constant over episode [N]
    max_external_torque = 0.0  # wind torque applied at base, constant over episode [Nm]
    max_external_foot_force = 0.0  # wind force applied at feet, constant over episode [N]


@configclass
class ObservationsCfg:
    @configclass
    class Policy:
        # optinal parameters: scale, clip([min, max]), noise
        add_noise: bool = True  # turns off the noise in all observations
        base_lin_vel: dict = {"func": O.base_lin_vel, "noise": 0.1}
        base_ang_vel: dict = {"func": O.base_ang_vel, "noise": 0.2}
        projected_gravity: dict = {"func": O.projected_gravity, "noise": 0.05}
        velocity_commands: dict = {"func": O.velocity_commands}
        dof_pos: dict = {"func": O.dof_pos, "noise": 0.01}
        dof_vel: dict = {"func": O.dof_vel, "noise": 1.5}
        actions: dict = {"func": O.actions}
        height_scan: dict = {"func": O.ray_cast, "noise": 0.1, "sensor": "height_scanner", "clip": (-1, 1.0)}
        # bpearl: dict = {"func_name": O.ray_cast, "noise": 0.1, "sensor": "bpearl_front"}
        # bpearl2: dict = {"func_name": O.ray_cast, "noise": 0.1, "sensor": "bpearl_rear"}

    policy = Policy()


@configclass
class RewardsCfg:
    # general params
    only_positive_rewards: bool = False
    # reward functions
    termination = {"func": R.termination, "scale": -0.0}
    tracking_lin_vel = {"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25}
    tracking_ang_vel = {"func": R.tracking_ang_vel, "scale": 0.5, "std": 0.25}
    lin_vel_z = {"func": R.lin_vel_z, "scale": -2.0}
    ang_vel_xy = {"func": R.ang_vel_xy, "scale": -0.05}
    torques = {"func": R.torques, "scale": -0.00002}
    power = {"func": R.power, "scale": -0.0, "std": 1.0}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-7}
    feet_air_time = {"func": R.feet_air_time, "scale": 0.5, "time_threshold": 0.5}
    no_fly = {"func": R.no_fly, "scale": 0.0}
    collision = {"func": R.collision, "scale": -1.0, "bodies": ".*(THIGH|SHANK)"}
    action_rate = {"func": R.action_rate, "scale": -0.01}
    dof_vel = {"func": R.dof_vel, "scale": -0.0}
    stand_still = {"func": R.stand_still, "scale": -0.0}
    base_height = {"func": R.base_height, "scale": -0.0, "height_target": 0.5, "sensor": "ray_caster"}
    flat_orientation = {"func": R.flat_orientation, "scale": -0.0}
    foot_slippage = {"func": R.foot_slippage, "scale": -0.0}
    dof_pos_limits = {"func": R.dof_pos_limits, "scale": -0.0}
    link_velocity = {"func": R.link_linear_velocity, "scale": -0.0,  "bodies": ".*(THIGH|SHANK)"}
    # stumble = {"func": "stumble", "scale": -1.0, "hv_ratio": 2.0}
    # contact_forces = {"func": "contact_forces", "scale": -0.01, "max_contact_force": 450}


@configclass
class TerminationsCfg:
    # general params
    illegal_contact = {"func": T.illegal_contact, "bodies": ".*(RF_FOOT|LF_FOOT)"}
    bad_orientation = None
    dof_torque_limit = None
    dof_pos_limit = None


@configclass
class CurriculumCfg:
    # general params
    terrain_levels = {"func": C.terrain_levels}
    max_lin_vel_command = None


@configclass
class LeggedEnvBipedalCfg(BaseEnvCfg):

    # common configuration (from base env)
    env = EnvCfg(num_envs=4096, num_actions=12, send_timeouts=True, episode_length_s=20, enable_debug_vis=False)
    gym = GymInterfaceCfg(viewer=ViewerCfg(eye=(10, 0, 6), target=(11, 5, 3)))
    control = ControlCfg(decimation=4, action_scale=0.5, action_clipping=100.0)

    # legged-env specific configurations
    # -- scene designing
    terrain = TerrainCfg(sub_terrains=CustomSubTerrainsCfg())
    robot = LeggedRobotCfg()
    sensors = SensorsCfg()
    # -- command processing
    commands = UnifromVelocityCommandCfg()
    # -- mdp signals
    randomization = RandomizationCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()
