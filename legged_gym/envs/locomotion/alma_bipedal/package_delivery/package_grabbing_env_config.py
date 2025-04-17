# utils
from legged_gym.utils import configclass

# common
from legged_gym.common.terrains.terrain_cfg import TerrainCfg
from legged_gym.common.gym_interface.gym_interface_cfg import GymInterfaceCfg, ViewerCfg, SimParamsCfg

# envs
from legged_gym.envs.base_env_config import EnvCfg
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg
from legged_gym.envs.locomotion.anymal_wheels.wheeled_legged_env_config import *
from legged_gym.envs.locomotion.legged_env_config import *

# package env
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_curiosity_gates as G
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_observations as O
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_rewards as R
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_terminations as T
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_curriculum as C
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_assets import (
    standing_robot_cfg,
    package_cfg,
    table_cfg,
)
import legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_terminations as Tb
import legged_gym.envs.locomotion.alma_bipedal.flat.bipedal_env.bipedal_rewards as Rb

@configclass
class PackageGrabbingRewardsCfg:
    only_positive_rewards: bool = False

    # related to package grabbing
    package_lifted = {"func": R.package_lifted, "scale":40, "scale_cur": 0.0}

    # related to navigating
    tracking_lin_vel_standing = {"func": R.tracking_lin_vel_standing, "scale": -2000.0, "scale_cur": 0.0, "std": 0.25}
    tracking_ang_vel_standing = {"func": R.tracking_ang_vel_standing, "scale": 0.0, "scale_cur": 0.0, "std": 0.25}

    # related to standing
    height = {"func": R.height, "scale": -2000.0, "height_target": 0.7}
    stand_straight = {"func": R.stand_straight, "scale": 0.0, "scale_cur": 0.0}
    straight_knees = {"func": R.straight_knees, "scale": 0.0, "scale_cur": 0.0}
    shoulder_symmetry = {"func": R.shoulder_symmetry, "scale": 0.0, "scale_cur": 0.0}
    base_height = {"func": R.base_height, "scale": 3.0, "height_target": 0.8},

    # miscellaneous
    torques = {"func": R.torques, "scale": -1.5e-5, "scale_cur": 0.0}
    dof_acc = {"func": R.dof_acc, "scale": -2.5e-7, "scale_cur": 0.0}
    dof_vel = {"func": R.dof_vel, "scale":-2.5e-4, "scale_cur": 0.0}
    action_rate = {"func": R.action_rate, "scale": 0.0, "scale_cur": 0.0}
    termination = {"func": R.termination, "scale": -1000.0}
    collision={"func": R.collision, "scale": -500, "bodies": ".*(base|RF_FOOT|LF_FOOT|LH_THIGH|RH_THIGH)"},


@configclass
class PackageGrabbingObservationsCfg:
    @configclass
    class Policy:
        # optional parameters: scale, clip([min, max]), noise
        add_noise: bool = True
        base_lin_vel: dict = {"func": O.base_lin_vel, "noise": 0}
        base_ang_vel: dict = {"func": O.base_ang_vel, "noise": 0}
        projected_gravity: dict = {"func": O.projected_gravity, "noise": 0}
        dof_pos: dict = {"func": O.dof_pos_selected, "noise": 0, "dofs": ".*(HAA|HFE|KFE)"}
        dof_vel: dict = {"func": O.dof_vel, "noise": 0}
        actions: dict = {"func": O.actions}
        velocity_commands: dict = {"func": O.velocity_commands_x_yaw}

        # package related observations
        camera_to_package_c: dict = {"func": O.camera_to_package_c, "fov_check": False, "noise": 0}
        camera_to_package_quat: dict = {"func": O.camera_to_package_quat, "fov_check": False, "noise": 0}
        package_velocity_c: dict = {"func": O.package_velocity_c, "fov_check": False, "noise": 0}
        camera_to_table_c: dict = {"func": O.camera_to_table_c, "noise": 0}

    policy = Policy()


@configclass
class PackageGrabbingCommandsCfg(UnifromVelocityCommandCfg):
    heading_command = False
    resampling_time = 2.5


@configclass
class PackageGrabbingTerminationsCfg(TerminationsCfg):
    illegal_contact = {"func": Tb.illegal_contact_standing, "bodies": ["base","LH_THIGH","RH_THIGH","LH_KFE","RH_KFE"]}
    #bad_orientation = {"func": Tb.bad_orientation_bipedal, "limit_angle_lower":0.9,"limit_angle_upper": 2.6}
    bad_orientation= None 
    """ not_standing_height = {
        "func": T.not_standing_height,
        "limit_height": 0.2,
    } """
    not_standing_height = None
    moved_away_from_package = {
        "func": T.moved_away_from_package,
        "limit_dist": 2.0,
    }


@configclass
class PackageGrabbingEnvCfg(WheeledLeggedEnvCfg):
    # common configuration
    robot = standing_robot_cfg
    package = package_cfg
    #table = table_cfg
    env = EnvCfg(
        num_envs=4096,
        num_actions=16,
        send_timeouts=True,
        episode_length_s=5,
        enable_debug_vis=True,
    )
    gym = GymInterfaceCfg(
        viewer=ViewerCfg(eye=(-1, -1, 1.3), target=(1.5, -0.3, 1.1)), sim_params=SimParamsCfg(dt=0.005)
    )
    terrain = TerrainCfg(mesh_type="plane")

    # legged-env configuration
    control = WheeledLeggedControlCfg(decimation=4, action_scale=0.5)
    randomization = RandomizationCfg(
       
        
        max_init_pos=0,
        max_init_yaw=0,
    )  # yaw is applied in + and - direction, 3.14/2 is full rotation

    # package grabbing configuration
    commands = PackageGrabbingCommandsCfg()
    observations = PackageGrabbingObservationsCfg()
    rewards = PackageGrabbingRewardsCfg()
    terminations = PackageGrabbingTerminationsCfg()
    curriculum = CurriculumCfg(terrain_levels=None)


""" @configclass
class PackageGrabbingRndConfig(RandomNetworkDistillationCfg):
    # Workaround to solve issue that callable gets converted in config_utils lines 80,81
    # when converting train_cfg from class to dict in task_registry line 195
    # Solution for now: pass function handle inside list so that it's not recognized as callable
    gate_config = {"func": [G.package_state], "output_size": 3}
    embedding_size = 1
    layers_target = 2
    layers_predictor = 4
    weight = 200
    reward_normalization = False
    gate_normalization = True """


@configclass
class PackageGrabbingEnvPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="package_grabbing",
        load_run=-1,
        empirical_normalization=True,
        max_iterations=1000,
        num_steps_per_env=24,  # hardcoded in reward_curriculum
        save_interval=500,
    )
    #algorithm = AlgorithmCfg(use_rnd=False, rnd_cfg=PackageGrabbingRndConfig(), entropy_coef=0.004)


# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env import PackageGrabbingEnv

task_registry.register("package_grabbing", PackageGrabbingEnv, PackageGrabbingEnvCfg, PackageGrabbingEnvPPOCfg)
