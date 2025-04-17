# isaac-gym
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import (
    quat_rotate_inverse,
    to_torch,
    get_axis_params,
    quat_apply,
)

# python
from typing import Dict
from copy import deepcopy
import torch
import numpy as np

# legged-gym
from legged_gym.envs import BaseEnv
from legged_gym.envs.manipulation.arm_env_config import ArmEnvCfg
from legged_gym.common.assets.robots import Manipulator
from legged_gym.common.sensors.sensors import *
from legged_gym.common.terrains import Terrain, TerrainGenerator


class ArmEnv(BaseEnv):
    robot: Manipulator
    """Environment for locomotion tasks using a legged robot."""

    def __init__(self, cfg: ArmEnvCfg):
        """Initializes the environment instance.

        Parses the provided config file, calls create_sim() (which creates, simulation,
        terrain and environments), initializes pytorch buffers used during training.

        Args:
            cfg (ArmEnvCfg): Configuration for the environment.
        """
        # Save some helpful quantities from config to make life easier.
        self.dt = cfg.control.decimation * cfg.gym.sim_params.dt
        self._command_ranges = deepcopy(cfg.commands.ranges)

        # initialize the parent
        # note: calls the `create_env` function to create environments.
        super().__init__(cfg)

    """
    Implementation Specifics - Public.
    """

    def reset_idx(self, env_ids):
        """Reset environments based on specified indices.

        Calls the following functions on reset:
        - :func:`_reset_robot`: Reset the root state and DOF state of the robot.
        - :func:`_resample_commands`: Resample the goal/command for the task. E.x.: desired velocity command.

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
        self._resample_commands(env_ids)
        # -- reset env buffers
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = dict()
        self.reward_manager.log_info(self, env_ids, self.extras["episode"])
        self.curriculum_manager.log_info(self, env_ids, self.extras["episode"])
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    """
    Implementation Specifics - Private.
    """

    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        terrain_generator = TerrainGenerator(self.cfg.terrain)
        self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.gym_iface)
        self.terrain.set_terrain_origins(terrain_generator.terrain_origins)
        self.terrain.add_mesh(terrain_generator.terrain_mesh, name="terrain")
        self.terrain.add_to_sim()
        # add robot class
        robot_cls = eval(self.cfg.robot.cls_name)
        self.robot: Manipulator = robot_cls(self.cfg.robot, self.num_envs, self.gym_iface)

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
        # computed obs only when we are done messing with the env
        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.obs_dict = self.obs_manager.compute_obs(self)

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            x = self.commands[i][0] + self.robot.root_states[i, 0]
            y = self.commands[i][1] + self.robot.root_states[i, 1]
            z = self.commands[i][2] + self.robot.root_states[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    """
    Helper functions (order of calling).
    """

    def _init_buffers(self):
        super()._init_buffers()
        """Initialize torch tensors which will contain simulation states and processed quantities."""
        # initialize some data used later on
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- action buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        # assets buffers
        # -- robot
        self.robot.init_buffers()

    def _reset_robot(self, env_ids):
        """Resets root and dof states of robots in selected environments."""
        # -- dof state (handled by the robot)
        dof_pos, dof_vel = self.robot.get_random_dof_state(env_ids)
        self.robot.set_dof_state(env_ids, dof_pos, dof_vel)
        # -- root state (custom)
        root_state = self.robot.get_default_root_state(env_ids)
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        # set into robot
        self.robot.set_root_state(env_ids, root_state)

    def _resample_commands(self, env_ids):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return
        r = torch.empty(
            len(env_ids), device=self.device
        )  # we can't do self.commands[env_ids, 0].uniform_ because it's a clone of self.commands
        self.commands[env_ids, 0] = r.uniform_(self._command_ranges.pos_x[0], self._command_ranges.pos_x[1])
        self.commands[env_ids, 1] = r.uniform_(self._command_ranges.pos_y[0], self._command_ranges.pos_y[1])
        self.commands[env_ids, 2] = r.uniform_(self._command_ranges.pos_z[0], self._command_ranges.pos_z[1])

    def _apply_external_disturbance(self):
        return  # skip for now

    def update_history(self):
        super().update_history()
        self.robot.update_history()
        self.last_actions[:] = self.actions[:]
