# python
from typing import Tuple, Union, Dict, Any
import math
import torch
import abc
from legged_gym.common.sensors import *

# legged-gym
from legged_gym.envs.base_env_config import BaseEnvCfg
from legged_gym.common.gym_interface import GymInterface
from legged_gym.common.observations.observation_manager import ObsManager
from legged_gym.common.rewards.reward_manager import RewardManager
from legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from legged_gym.common.terminations.termination_manager import TerminationManager


class BaseEnv:
    """Base class for RL tasks."""

    def __init__(
        self,
        cfg: BaseEnvCfg,
    ):
        """Initialize the base class for RL environment.

        The class initializes the simulation application. It also allocates buffers for observations,
        actions, rewards, reset, episode length, and extras (episode time-out, and optional observation groups).

        Args:
            cfg (BaseEnvCfg): Configuration for the environment.
        """
        self._init_done = False
        # copy input arguments into class members
        self.cfg = cfg
        # store the environment information from config
        self.num_envs = self.cfg.env.num_envs
        """Number of environment instances."""
        self.num_actions = self.cfg.env.num_actions
        """Number of actions in the environment."""
        self.dt = self.cfg.control.decimation * self.cfg.gym.sim_params.dt
        """Discretized time-step for episode horizon."""
        self.max_episode_length_s = self.cfg.env.episode_length_s
        """Maximum duration of episode (in seconds)."""
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        """Maximum number of steps per episode."""

        # create isaac-interface
        self.gym_iface = GymInterface(cfg.gym)
        # create envs, sim
        self.device = self.gym_iface.device
        self.gym = self.gym_iface.gym
        self.sim = self.gym_iface.sim
        self._create_envs()
        # prepare sim buffers
        self.gym_iface.prepare_sim()
        # store commonly used members from gym-interface for easy access.
        # TODO: Make this as unnecessary as possible.
        self.viewer = self.gym_iface.viewer

        # initialize buffers for environment
        self._init_buffers()
        self.sensors = dict()
        self._init_external_forces()

        # prepare mdp helper managers
        self.reward_manager = RewardManager(self)
        self.obs_manager = ObsManager(self)
        self.termination_manager = TerminationManager(self)
        self.curriculum_manager = CurriculumManager(self)

        # perform initial reset of all environments (to fill up buffers)
        self.reset()
        self.obs_dict = self.obs_manager.compute_obs(self)
        # we are ready now! :)
        self._init_done = True

    def _init_external_forces(self):
        pass

    """
    Properties.
    """

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.obs_buf, self.extras

    """
    Operations.
    """

    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset all environment instances."""
        # reset environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # perform single-step to get observations
        zero_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        obs, _, _, extras = self.step(zero_actions)
        # return obs
        return obs, extras

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply actions, simulate, call self.post_physics_step()

        After above, the scaled actions are passed to :func:`_pre_process_actions()` for environment
        specific processing of input actions. Following this, the commands are applied on the actors
        through the :func:`_apply_actions()`, which is called at specified decimation rate.
        After performing simulation stepping, the :func:`_post_physics_step()` is called for computing
        the MDP signals and handling resets.

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            VecEnvStepReturn: A tuple containing:
                - (VecEnvObs) observations from the environment
                - (torch.Tensor) reward from the environment
                - (torch.Tensor) whether the current episode is completed or not
                - (dict) misc information
        """
        # -- environment specific pre-processing
        processed_actions = self._preprocess_actions(actions)
        # apply actions into simulator
        for _ in range(self.cfg.control.decimation):
            # may include recomputing torques (based on actuator models)
            self._apply_actions(processed_actions)
            # apply external disturbance to base and feet
            self._apply_external_disturbance()
            # simulation step
            self.gym_iface.simulate()
            # refresh tensors
            self.gym_iface.refresh_tensors(dof_state=True)
        # render viewer
        self.render()
        # update sim counters
        self.episode_length_buf += 1
        # post-physics computation
        self._post_physics_step()
        # return clipped obs, rewards, dones and infos
        # return policy obs as the main and rest of observations into extras.
        self.obs_buf = self.obs_dict["policy"]
        self.extras["observations"] = self.obs_dict
        # Story memory
        self.update_history()
        # return mdp tuples
        return (self.obs_buf, self.rew_buf, self.reset_buf, self.extras)

    def enable_sensor(self, sensor_name):
        sensor_cfg = self.cfg.sensors.__getattribute__(sensor_name)
        if sensor_name not in self.sensors.keys():
            sensor: SensorBase = eval(sensor_cfg.class_name)(sensor_cfg, self)
            sensor.update(self.dt)
            self.sensors[sensor_name] = sensor

    def render(self, sync_frame_time=True):
        """Render the viewer."""
        # render the GUI
        # perform debug visualization
        self.gym.clear_lines(self.gym_iface.viewer)
        if (
            (self.cfg.env.enable_debug_vis or self.gym_iface.enable_debug_viz)
            and self.gym_iface.viewer
            and self.gym_iface._enable_viewer_sync
        ):
            self._draw_debug_vis()
        self.gym_iface.render(sync_frame_time)

    """
    Implementation Specifics - Public.
    """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Resets the MDP for given environment instances.

        Args:
            env_ids (torch.Tensor): A tensor containing indices of environment instances to reset.
        """
        raise NotImplementedError

    """
    Implementation Specifics - Private.
    """

    def _init_buffers(self):
        # allocate common buffers
        self.obs_dict = dict()
        self.rew_buf = None
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # allocate dictionary to store metrics
        self.extras = dict()

    @abc.abstractmethod
    def _create_envs(self):
        """Design the environment instances."""
        raise NotImplementedError

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
        scaled_actions = self.cfg.control.action_scale * self.actions
        return scaled_actions

    @abc.abstractmethod
    def _apply_actions(self, actions: torch.Tensor):
        """Apply actions to simulation buffers in the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_external_disturbance(self):
        """Apply external disturbance to simulation buffers in the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def _post_physics_step(self):
        """Post-physics computation: such as computing MDP signals, resetting the environment."""
        raise NotImplementedError

    def _draw_debug_vis(self):
        """Additional visualizations for debugging purposes (can make rendering slow)."""
        pass

    def update_history(self):
        pass
