from legged_gym.utils.config_utils import configclass
from legged_gym.envs.base_env_config import EnvCfg
from legged_gym.envs.locomotion.anymal_c.flat.anymal_c_flat_config import AnymalCFlatEnvCfg, AnymalCFlatPPOCfg
from legged_gym.common.assets.robots import alma_bipedal_robot_cfg
from legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_legged_env_config import RewardsCfg,CurriculumCfg,TerrainCfg,ObservationsCfg, TerminationsCfg,RandomizationCfg
from legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_legged_env_config import LeggedEnvBipedalCfg
from legged_gym.envs.rl_config import RunnerCfg
import legged_gym.envs.locomotion.rewards as R
import legged_gym.envs.locomotion.terminations as T
import legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_terminations as Tb
import legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_rewards as Rb
from legged_gym.common.sensors.sensors_cfg import AnymalCSensors
from legged_gym.common.commands.command_cfg import UnifromVelocityCommandCfg

@configclass
class AlmaBipedalEnvCfg(AnymalCFlatEnvCfg):
    randomization = RandomizationCfg(
    max_init_pos = 0.0001 , # max xy position added to default position [m]
    max_init_yaw = 0.0001 , # max yaw angle added to default orientation [rad]
    max_init_roll_pitch = 0.00001 , # max roll and pitch angles added to default orientation [rad]
    push_robots = False,
    push_interval_s = 0.00001 , # push applied each time interval [s]
    max_push_vel = 0.0001, # velocity offset added by push [m/s]
    max_external_force = 0.00001, # wind force applied at base, constant over episode [N]
    max_external_torque = 0.00001,  # wind torque applied at base, constant over episode [Nm]
    max_external_foot_force = 0.00001  # wind force applied at feet, constant over episode [N]
    )
    robot = alma_bipedal_robot_cfg
    sensors = AnymalCSensors()

    terrain = TerrainCfg(mesh_type="plane")
    observations = ObservationsCfg(policy=ObservationsCfg.Policy(height_scan=None))

    env = EnvCfg(num_envs=4096, num_actions=18, send_timeouts=True, episode_length_s=8, enable_debug_vis=False)
    rewards = RewardsCfg(
        tracking_ang_vel={"func": Rb.tracking_ang_vel_standing, "scale": -150, "std": 0.25},
        base_height = {"func": R.base_height, "scale": -2000.0, "height_target": 0.8},
        flat_orientation = {"func": R.flat_orientation, "scale":0.0},
        termination = {"func": R.termination, "scale": -5000},
        goal_dist = {"func" : Rb.dist_from_goal, "scale": -0},
        collision={"func": R.collision, "scale": 0, "bodies": ".*(base|RF_FOOT|LF_FOOT|LH_THIGH|RH_THIGH)"},
        feet_air_time={"func": R.feet_air_time, "scale":3000, "time_threshold": 0.5},
        torques={"func": R.torques, "scale": -0.000002},
        dof_acc={"func": R.dof_acc, "scale": -2.5e-6},
        link_velocity = {"func": Rb.link_linear_velocity, "scale": 0,  "body_indices": [0]},
        tracking_lin_vel={"func": Rb.tracking_lin_vel_standing, "scale":0, "std": 0.25},
        stand_still = {"func": R.stand_still, "scale": 0.0},
        
        dof_vel = {"func": R.dof_vel, "scale":-0},
        lin_vel_z={"func": R.lin_vel_z, "scale": -0},
        ang_vel_xy={"func": R.ang_vel_xy, "scale": 0},
        action_rate={"func": R.action_rate, "scale": -0},
    )

    terminations = TerminationsCfg(    

    illegal_contact = {"func": Tb.illegal_contact_standing, "bodies": ".*(LF_THIGH|RF_THIGH|LF_KFE|RF_KFE)"},
    bad_orientation = {"func": Tb.bad_orientation_bipedal, "limit_angle_lower":1, "limit_angle_upper": 2.6},
    dof_torque_limit = None,
    dof_pos_limit = None
    )

    commands = UnifromVelocityCommandCfg(heading_command=False, resampling_time=4.0)
    curriculum = CurriculumCfg(terrain_levels=None)


@configclass
class AlmaFlatPPOCfg(AnymalCFlatPPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="flat_alma_bipedal",
        load_run=-1,
        max_iterations=300,
    )



# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_legged_env import LeggedEnvBipedal

task_registry.register("alma_flat_bipedal", LeggedEnvBipedal, AlmaBipedalEnvCfg, AlmaFlatPPOCfg)
