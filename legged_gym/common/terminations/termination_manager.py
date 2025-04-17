import torch

# solves circular imports of LeggedRobot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv


class TerminationManager:
    def __init__(self, env: "LeggedEnv"):
        """Prepares a list of functions"""
        self.functions = {}
        self.params = {}
        for name, params in env.cfg.terminations.__dict__.items():
            if not isinstance(params, dict):
                continue
            if "dofs" in params.keys():
                params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
            if "bodies" in params.keys():
                params["body_indices"], _ = env.robot.find_bodies(params["bodies"])

            # function = getattr(self, name)
            function = params["func"]
            self.functions[name] = function
            self.params[name] = params
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])

    def check_termination(self, env: "LeggedEnv"):
        """Check terminations
        Calls each termiantion function which was defined in the config (processed in self.__init__). Returns the logical OR of all functions.
        """
        terminated = torch.zeros_like(env.reset_buf)
        for name, function in self.functions.items():
            params = self.params[name]
            terminated |= function(env, params)
        return terminated
