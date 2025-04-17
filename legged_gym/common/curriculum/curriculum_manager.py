import torch

# solves circular imports of LeggedRobot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv


class CurriculumManager:
    def __init__(self, env: "LeggedEnv"):
        """Prepares a list of fucntions"""
        self.functions = {}
        self.params = {}
        for name, params in env.cfg.curriculum.__dict__.items():
            if not isinstance(params, dict):
                continue
            # function = getattr(self, name)
            function = params["func"]
            self.functions[name] = function
            self.params[name] = params
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])

    def update_curriculum(self, env: "LeggedEnv", env_ids):
        """Update curriculum
        Calls each update function which was defined in the config (processed in self.__init__). Each function modifies the env directly.
        """
        for name, function in self.functions.items():
            params = self.params[name]
            function(env, env_ids, params)

    def log_info(self, env: "LeggedEnv", env_ids, extras_dict):
        if "terrain_levels" in self.functions.keys():
            extras_dict["terrain_level"] = torch.mean(env.terrain.terrain_levels.float())
        if "max_lin_vel_command" in self.functions.keys():
            extras_dict["max_command_x"] = env.command_ranges["lin_vel_x"][1]
            extras_dict["max_command_y"] = env.command_ranges["lin_vel_y"][1]
