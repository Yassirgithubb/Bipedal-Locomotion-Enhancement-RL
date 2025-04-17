from legged_gym.utils.config_utils import configclass
from legged_gym.common.terrains.terrain_cfg import TerrainCfg
from legged_gym.common.sensors.sensors_cfg import CassieSensor
from legged_gym.envs.locomotion.legged_env_config import (
    CurriculumCfg,
    LeggedEnvCfg,
    RandomizationCfg,
    RewardsCfg,
    ObservationsCfg,
    ControlCfg,
    EnvCfg,
    GymInterfaceCfg,
    ViewerCfg,
    TerminationsCfg,
)
from legged_gym.common.gym_interface.gym_interface_cfg import SimParamsCfg
from legged_gym.envs.rl_config import PPOCfg, RunnerCfg, AlgorithmCfg, PolicyCfg
from legged_gym.common.assets.robots import cassie_robot_cfg
import legged_gym.envs.locomotion.observations as O
import legged_gym.envs.locomotion.rewards as R
import legged_gym.envs.locomotion.terminations as T
import legged_gym.envs.locomotion.curriculum as C
from legged_gym.common.commands.command_cfg import UnifromVelocityCommandCfg
from legged_gym.common.terrains.terrain_cfg import TerrainCfg, SubTerrainsCfg

@configclass
class HardSubTerrainsCfg(SubTerrainsCfg):
    hf_discrete_obstacles = SubTerrainsCfg.HfDiscreteObstaclesCfg(proportion=0.1,
        num_rectangles = 40,
        rectangle_min_size = 1.0,
        rectangle_max_size = 2.0,
        rectangle_min_height = 0.01,
        rectangle_max_height = 0.06)

@configclass
class CassieRoughCfg(LeggedEnvCfg):
    robot = cassie_robot_cfg

    observations = ObservationsCfg(
        policy=ObservationsCfg.Policy(
            add_noise = False,
            base_lin_vel = {"func": O.base_lin_vel, "noise": 0.1},
            base_ang_vel = {"func": O.base_ang_vel, "noise": 0.2},
            projected_gravity = {"func": O.projected_gravity, "noise": 0.05},
            velocity_commands = {"func": O.velocity_commands},
            dof_pos = {"func": O.dof_pos, "noise": 0.01},
            dof_vel = {"func": O.dof_vel, "noise": 1.5},
            actions = {"func": O.actions},
            height_scan = {"func": O.ray_cast, "noise": 0.0, "sensor": "height_scanner", "clip": (-1, 1.0)},
            )
        )

    sensors = CassieSensor()

    rewards = RewardsCfg(
        only_positive_rewards = False,
        tracking_lin_vel={"func": R.tracking_lin_vel, "scale": 1.0, "std": 0.25},
        tracking_ang_vel={"func": R.tracking_ang_vel, "scale": 1.0, "std": 0.25},
        torques={"func": R.torques, "scale": -5e-6},
        power={"func": R.power, "scale": 0.25, "std": 10.0}, 
        dof_acc={"func": R.dof_acc, "scale": -2e-7},
        action_rate={"func": R.action_rate, "scale": -0.02},
        foot_slippage = {"func": R.foot_slippage, "scale": -0.1},
        dof_pos_limits={"func": R.dof_pos_limits, "scale": -1.0},
        collision = {"func": R.collision, "scale": -1.0, "bodies": "pelvis"},
    
        lin_vel_z = {"func": R.lin_vel_z, "scale": -0.5},
        termination = {"func": R.termination, "scale": -200.0},
        ang_vel_xy = {"func": R.ang_vel_xy, "scale": -0.0},
        feet_air_time = {"func": R.feet_air_time, "scale": 5.0, "time_threshold": 1.0},
        no_fly = {"func": R.no_fly, "scale": 0.25},
        dof_vel = {"func": R.dof_vel, "scale": 0.0},
        stand_still = {"func": R.stand_still, "scale": 0.0},
        base_height = {"func": R.base_height, "scale": 0.0, "height_target": 0.3, "sensor": "height_scanner"},
        flat_orientation = {"func": R.flat_orientation, "scale": -0.0},
    )

    commands = UnifromVelocityCommandCfg(
        heading_command=True,
        resampling_time=4.0,
        rel_standing_envs=0.05,
        rel_heading_envs = 0.9,
        ranges = UnifromVelocityCommandCfg.Ranges(
            lin_vel_x = (-1.0, 1.0),
            lin_vel_y = (-0.5, 0.5),
            ang_vel_yaw = (-2.0, 2.0),
        )
    )

    randomization = RandomizationCfg(
        push_robots = False,
        max_push_vel = 0.0,
        max_external_force = 0.0,
        max_external_torque = 0.0
    )

    control = ControlCfg(
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1,
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8,
        action_clipping=100.0,
    )

    env = EnvCfg(
        num_envs=4096,
        num_actions=12,
        send_timeouts=True,
        episode_length_s=20,
        enable_debug_vis=False
    )

    gym = GymInterfaceCfg(
        viewer=ViewerCfg(eye=[10, -1, 0.7], target=[-1, 1, 0.0]),
        sim_params=SimParamsCfg(dt=0.0025)
    )

    terminations = TerminationsCfg(illegal_contact = {"func": T.illegal_contact, 
                                                         "bodies": ["pelvis"]})

    terrain = TerrainCfg(sub_terrains=HardSubTerrainsCfg())

@configclass
class CassieRoughPPOCfg(PPOCfg):
    runner = RunnerCfg(
        run_name="",
        experiment_name="cassie",
        load_run=-1,
        resume=False,
        max_iterations=10000,
        save_interval=100,
        empirical_normalization=False
    )

    policy: PolicyCfg = PolicyCfg(
        actor_hidden_dims = [128, 128, 128], 
        critic_hidden_dims = [128, 128, 128],
        init_noise_std = 1.2
    )

    algorithm: AlgorithmCfg = AlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,
        num_learning_epochs = 5,
        num_mini_batches = 10,  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 1.0e-3,  # 5.e-4
        schedule = "adaptive",  # adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )

# register
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.locomotion.legged_env import LeggedEnv

task_registry.register("cassie", LeggedEnv, CassieRoughCfg, CassieRoughPPOCfg)
