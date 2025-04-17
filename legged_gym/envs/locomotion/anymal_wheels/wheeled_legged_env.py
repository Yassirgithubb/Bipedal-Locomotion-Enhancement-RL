from legged_gym.envs.locomotion.legged_env import LeggedEnv
from legged_gym.common.observations.observation_manager import ObsManager
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg
from legged_gym.common.rewards.reward_manager import RewardManager
from legged_gym.common.assets.robots import LeggedRobot
from legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from legged_gym.common.terminations.termination_manager import TerminationManager
import torch


class WheeledLeggedEnv(LeggedEnv):
    robot: LeggedRobot

    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pre-process actions from the environment into actor's commands.
        The step call (by default) performs the following operations:
            - clipping of actions to a range (based on configuration)
            - scaling of actions (based on configuration)
        """
        # clip actions and move to env device
        actions = torch.clip(actions, -self.cfg.control.action_clipping, self.cfg.control.action_clipping)
        actions = actions.to(self.device)
        self.actions = actions
        # step physics
        # pre-process actions
        # -- default scaling of actions
        # TODO: Remove magic number of actuators and add namespace.
        wheel_indices = self.robot._actuators[1].dof_ids
        scaled_actions = self.cfg.control.action_scale * self.actions
        scaled_actions[:, wheel_indices] = self.cfg.control.wheel_action_scale * self.actions[:, wheel_indices]
        # scaled_actions[:, wheel_indices] += self.commands[:, 0].unsqueeze(1) / self.robot.cfg.feet_position_offset[2]
        return scaled_actions
