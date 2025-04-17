# python
from copy import deepcopy
import torch
import numpy as np

# isaacgym
from isaacgym.torch_utils import quat_apply

# legged_gym
from legged_gym.utils.math import wrap_to_pi
from .command_cfg import UnifromVelocityCommandCfg, NormalVelocityCommandCfg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs import BaseEnv


class CommandBase:
    def __init__(self, cfg, env):
        # prepare some values
        raise NotImplementedError()

    def resample(self, env_ids=None):
        # resample commands
        raise NotImplementedError()

    def update(self, env_ids=None):
        # compute stuff
        raise NotImplementedError()

    def get_command(self):
        # returns command data
        raise NotImplementedError()

    def reset(self):
        pass


class UnifromVelocityCommand(CommandBase):
    def __init__(self, cfg: UnifromVelocityCommandCfg, env: "BaseEnv"):
        self.cfg = cfg
        self.robot = getattr(env, cfg.robot_name)
        self.num_envs = self.robot.num_envs
        self.device = self.robot.device
        self.command_ranges = deepcopy(cfg.ranges)

        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, self.cfg.num_commands, device=self.device)
        self.tracking_error_sum = torch.zeros(self.num_envs, self.cfg.num_commands, device=self.device)
        self.log_step_counter = torch.zeros(self.num_envs, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def resample(self, env_ids=None):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return

        # set tracking error to zero
        self.tracking_error_sum[env_ids] = 0.0
        self.log_step_counter[env_ids] = 0.0

        # resample velocities
        self.resample_velocities(env_ids)

    def resample_velocities(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        # print(self.commands[env_ids], env_ids)
        self.commands[env_ids, 0] = r.uniform_(self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1])
        # linear velocity - y direction
        self.commands[env_ids, 1] = r.uniform_(self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1])
        # # ang vel yaw - rotation around z
        self.commands[env_ids, 2] = r.uniform_(self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1])
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(self.command_ranges.heading[0], self.command_ranges.heading[1])
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def update(self, env_ids=None):
        """Sets velocity commands to zero for standing envs, computes angular velocity from heading direction."""

        if self.cfg.heading_command:
            # Compute angular velocity from heading direction for heading envs
            heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            forward = quat_apply(self.robot.root_quat_w[heading_env_ids, :], self.robot._forward_vec_b[heading_env_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[heading_env_ids, 2] = torch.clip(
                0.5 * wrap_to_pi(self.heading_target[heading_env_ids] - heading),
                self.cfg.ranges.ang_vel_yaw[0],
                self.cfg.ranges.ang_vel_yaw[1],
            )

        # Enforce standing (i.e., zero velocity commands) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :] = 0.0

        self.log_data()

    def log_data(self):
        # logs data
        self.tracking_error_sum[:, :2] += torch.abs(self.commands[:, :2] - self.robot.root_lin_vel_b[:, :2])
        self.tracking_error_sum[:, 2] += torch.abs(self.commands[:, 2] - self.robot.root_ang_vel_b[:, 2])
        self.log_step_counter += 1

    def get_command(self):
        return self.commands


class NormalVelocityCommand(UnifromVelocityCommand):
    def __init__(self, cfg: NormalVelocityCommandCfg, env: "BaseEnv"):
        UnifromVelocityCommand.__init__(self, cfg, env)
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_zero_vel_y_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_zero_vel_yaw_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def resample_velocities(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.commands[env_ids, 0] = r.normal_(mean=self.command_ranges.mean_vel[0], std=self.command_ranges.std_vel[0])
        self.commands[env_ids, 1] = r.normal_(mean=self.command_ranges.mean_vel[1], std=self.command_ranges.std_vel[1])
        self.commands[env_ids, 2] = r.normal_(mean=self.command_ranges.mean_vel[2], std=self.command_ranges.std_vel[2])
        self.commands[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        self.commands[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        self.commands[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update zero vel envs
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.command_ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.command_ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.command_ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def update(self, env_ids=None):
        """Sets velocity commands to zero for standing envs."""
        # Enforce standing (i.e., zero velocity commands) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :] = 0.0

        # Enforce zero velocity
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        self.commands[zero_vel_x_env_ids, 0] = 0.0
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        self.commands[zero_vel_y_env_ids, 1] = 0.0
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.commands[zero_vel_yaw_env_ids, 2] = 0.0

        self.log_data()
