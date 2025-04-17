# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from legged_gym.envs.manipulation import ArmEnv
    from legged_gym.envs.locomotion import LeggedEnv

    ANY_ENV = Union[ArmEnv, LeggedEnv]
"""
Common gate functions
A gate function is defined by g(s)->s', where s is the state and s' is the state after the gate function has been applied.
"""


def obs_selected(env: "ANY_ENV", params):
    indices = params["obs_indices"]
    return env.obs_dict["policy"][:, indices]
