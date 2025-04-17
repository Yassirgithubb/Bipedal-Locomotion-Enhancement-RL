# isaac-gym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply, quat_mul

# python
from copy import deepcopy
import torch
import numpy as np

# legged-gym
from legged_gym.envs import BaseEnv
from legged_gym.envs.locomotion.alma_bipedal.flat.alma_flat_bipedal_config import LeggedEnvBipedalCfg
from legged_gym.common.assets.robots import LeggedRobot, LeggedMobileManipulator
from legged_gym.common.sensors.sensors import *
from legged_gym.common.terrains import Terrain, TerrainGenerator
from legged_gym.utils.math import wrap_to_pi
from legged_gym.common.commands.command import *


class LeggedEnvBipedal(BaseEnv):
    robot: LeggedRobot
    cfg: LeggedEnvBipedalCfg
    """Environment for locomotion tasks using a legged robot."""

    def __init__(self, cfg: LeggedEnvBipedalCfg):
        """Initializes the environment instance.

        Parses the provided config file, calls create_sim() (which creates, simulation,
        terrain and environments), initializes pytorch buffers used during training.

        Args:
            cfg (LeggedEnvCfg): Configuration for the environment.
        """
        # Save some helpful quantities from config to make life easier.
        self.dt = cfg.control.decimation * cfg.gym.sim_params.dt
        self._push_interval = np.int(np.ceil(cfg.randomization.push_interval_s / self.dt))

        # initialize the parent
        # note: calls the `create_env` function to create environments.
        self.g_cuda = torch.Generator(device="cuda")
        super().__init__(cfg)

    """
    Implementation Specifics - Public.
    """

    def _init_external_forces(self):
        self.external_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.external_torques = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)

    def reset_idx(self, env_ids):
        """Reset environments based on specified indices.

        Calls the following functions on reset:
        - :func:`_reset_robot`: Reset the root state and DOF state of the robot.

        Addition to above, the function fills up episode information into extras and resets buffers.

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        # -- reset robot state
        self._reset_robot(env_ids)
        # -- write to simulator
        self.gym_iface.write_states_to_sim()

        # -- reset robot buffers
        self.robot.reset_buffers(env_ids)
        # -- resample commands
        self.command_generator.resample(env_ids)
        self.command_generator.update()
        # -- reset env buffers
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.push_robots_buf[env_ids] = torch.randint(
            0, self._push_interval, (len(env_ids), 1), device=self.device
        ).squeeze()

        self.extras["episode"] = dict()
        self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        self.curriculum_manager.log_info(self, env_ids, self.extras["episode"])
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # Resample disturbances
        if self.cfg.randomization.max_external_force > 0.0:
            self.external_forces[env_ids, 0, :] = (
                2.0
                * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                * self.cfg.randomization.max_external_force
            )
        if self.cfg.randomization.max_external_torque > 0.0:
            self.external_torques[env_ids, 0, :] = (
                2.0
                * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                * self.cfg.randomization.max_external_torque
            )
        if self.cfg.randomization.max_external_foot_force > 0.0:
            for foot_id in self.robot.feet_indices:
                self.external_forces[env_ids, foot_id, :] = (
                    2.0
                    * (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                    * self.cfg.randomization.max_external_foot_force
                )

    """
    Implementation Specifics - Private.
    """

    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        terrain_curriculum = self.cfg.curriculum.__dict__.get("terrain_levels", None) is not None
        terrain_generator = TerrainGenerator(self.cfg.terrain, curriculum=terrain_curriculum)
        self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.gym_iface)
        self.terrain.set_terrain_origins(terrain_generator.terrain_origins)
        self.terrain.add_mesh(terrain_generator.terrain_mesh, name="terrain")
        self.terrain.add_to_sim()
        # add robot class
        robot_cls = eval(self.cfg.robot.cls_name)
        self.robot: LeggedRobot = robot_cls(self.cfg.robot, self.num_envs, self.gym_iface)

        # create environments
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = list()
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            # spawn robot
            pos = self.terrain.env_origins[i].clone()
            self.robot.spawn(i, pos)

    def _apply_actions(self, actions):
        """Apply actions to simulation buffers in the environment."""
        # set actions to interface buffers
        self.robot.apply_actions(actions)
        # set actions to sim
        self.gym_iface.write_dof_commands_to_sim()

    def _apply_external_disturbance(self):
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.external_forces),
            gymtorch.unwrap_tensor(self.external_torques),
            gymapi.ENV_SPACE,
        )

    def _post_physics_step(self):
        """Check terminations, checks erminations and computes rewards, and cache common quantities."""
        # refresh all tensor buffers
        self.gym_iface.refresh_tensors(
            root_state=True,
            net_contact_force=True,
            rigid_body_state=True,
            dof_state=True,
            dof_torque=self.robot.has_dof_torque_sensors,
        )
        # update env counters (used for curriculum generation)
        self.common_step_counter += 1
        self.push_robots_buf += 1

        # update robot
        self.robot.update_buffers(dt=self.dt)
        # update sensors TODO for optimatility we could avoid this, and only update once, if rewards/terminations don't rely on any sensor
        for _, s in self.sensors.items():
            s.update(dt=self.dt)
        # rewards, resets, ...
        # -- terminations
        self.reset_buf = self.termination_manager.check_termination(self)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # -- rewards
        self.rew_buf = self.reward_manager.compute_reward(self)
        if len(env_ids) != 0:
            # -- update curriculum
            if self._init_done:
                self.curriculum_manager.update_curriculum(self, env_ids)
            # -- reset terminated environments
            self.reset_idx(env_ids)
            # re-update robots for envs that were reset
            self.robot.update_buffers(dt=self.dt, env_ids=env_ids)
            # re-update sensors for envs that were reset
            for _, s in self.sensors.items():
                s.update(dt=self.dt, env_ids=env_ids)

        # Resample commands
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten()
        self.command_generator.resample(env_ids)
        self.command_generator.update()

        # -- obs
        self.obs_dict = self.obs_manager.compute_obs(self)

        # Push robots
        self._push_robots()

    def _draw_debug_vis(self):
        """Draws height measurement points for visualization."""
        # draw height lines
        if "height_scanner" not in self.sensors:
            return
        self.sensors["height_scanner"].debug_vis(self)

    """
    Helper functions (order of calling).
    """

    def _init_buffers(self):
        super()._init_buffers()
        """Initialize torch tensors which will contain simulation states and processed quantities."""
        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        self.push_robots_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        # -- command
        self.command_generator: CommandBase = eval(self.cfg.commands.class_name)(self.cfg.commands, self)
        # assets buffers
        # -- robot
        self.robot.init_buffers()

    def _reset_robot(self, env_ids):
        """Resets root and dof states of robots in selected environments."""
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids)
        self.robot.set_dof_state(env_ids, dof_pos, dof_vel)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        # shift initial pose
        if self.cfg.randomization.max_init_pos > 0.0:
            root_state[:, :2] += torch.empty_like(root_state[:, :2]).uniform_(
                -self.cfg.randomization.max_init_pos, self.cfg.randomization.max_init_pos
            )
        if self.cfg.randomization.max_init_roll_pitch > 0.0 or self.cfg.randomization.max_init_yaw > 0.0:
            roll = torch.empty(len(env_ids), device=self.device).uniform_(
               -self.cfg.randomization.max_init_roll_pitch, self.cfg.randomization.max_init_roll_pitch
            )
            pitch = torch.empty(len(env_ids), device=self.device).uniform_(
                -np.pi/2-self.cfg.randomization.max_init_roll_pitch, -np.pi/2 +self.cfg.randomization.max_init_roll_pitch
            )
            yaw = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.randomization.max_init_yaw, self.cfg.randomization.max_init_yaw
            )
            root_state[:, 3:7] = quat_mul(quat_from_euler_xyz(roll, pitch, yaw), root_state[:, 3:7])
        # base velocities: [7:10]: lin vel, [10:13]: ang vel
        root_state[:, 7:13].uniform_(-0.5, 0.5)
        # set into robot
        self.robot.set_root_state(env_ids, root_state)

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.cfg.randomization.push_robots:
            env_ids = self.push_robots_buf % self._push_interval == 0
            env_ids = env_ids.nonzero(as_tuple=False).flatten()

            if len(env_ids) == 0:
                return

            self.robot.root_states[env_ids, 7:13] += torch.empty(len(env_ids), 6, device=self.device).uniform_(
                -self.cfg.randomization.max_push_vel, self.cfg.randomization.max_push_vel
            )
            self.gym_iface.write_states_to_sim()

    def update_history(self):
        super().update_history()
        self.robot.update_history()
        self.last_actions[:] = self.actions[:]
