"""Test cases for various situations with base configuration."""

# python
import pytest
from dataclasses import field, asdict

# legged-gym
from legged_gym.utils.config_utils import configclass, update_class_from_dict, class_to_dict

"""
Dummy configuration.
"""


def double(x):
    """Dummy function."""
    return 2 * x


@configclass
class Viewer:
    eye: list = [7.5, 7.5, 7.5]  # field missing on purpose
    lookat: list = field(default_factory=[0.0, 0.0, 0.0])


@configclass
class Env:
    num_envs: int = double(28)  # uses function for assignment
    episode_length: int = 2000
    viewer: Viewer = Viewer()


@configclass
class RobotDefaultState:
    pos = (0.0, 0.0, 0.0)  # type annotation missing on purpose (immutable)
    rot: tuple = (1.0, 0.0, 0.0, 0.0)
    dof_pos: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dof_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # type annotation missing on purpose (mutable)


@configclass
class DefaultConfig:
    device_id: int = 0
    env: Env = Env()
    robot_default_state: RobotDefaultState = RobotDefaultState()


"""
Test solutions.
"""

config_correct = {
    "env": {"num_envs": 56, "episode_length": 2000, "viewer": {"eye": [7.5, 7.5, 7.5], "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (1.0, 0.0, 0.0, 0.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
}

config_change_correct = {
    "env": {"num_envs": 22, "episode_length": 2000, "viewer": {"eye": (2.0, 2.0, 2.0), "lookat": [0.0, 0.0, 0.0]}},
    "robot_default_state": {
        "pos": (0.0, 0.0, 0.0),
        "rot": (1.0, 0.0, 0.0, 0.0),
        "dof_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "dof_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    "device_id": 0,
}

"""
Test Fixture
"""


def test_str():
    cfg = DefaultConfig()
    print()
    print(cfg)


def test_str_dict():
    cfg = DefaultConfig()
    print()
    print(asdict(cfg))


def test_dict_conversion():
    cfg = DefaultConfig()
    # dataclass function
    assert asdict(cfg) == config_correct
    assert asdict(cfg.env) == config_correct["env"]
    # legged-gym utility function
    assert class_to_dict(cfg) == config_correct
    assert class_to_dict(cfg.env) == config_correct["env"]


def test_config_update_constructor():
    cfg = DefaultConfig(env=Env(num_envs=22, viewer=Viewer(eye=(2.0, 2.0, 2.0))))
    assert asdict(cfg) == config_change_correct


def test_config_update_after_init():
    cfg = DefaultConfig()
    cfg.env.num_envs = 22
    cfg.env.viewer.eye = (2.0, 2.0, 2.0)  # note: changes from list to tuple
    assert asdict(cfg) == config_change_correct


def test_config_update_dict():
    cfg = DefaultConfig()
    cfg_dict = {"env": {"num_envs": 22, "viewer": {"eye": (2.0, 2.0, 2.0)}}}
    update_class_from_dict(cfg, cfg_dict)
    assert asdict(cfg) == config_change_correct


def test_invalid_update_key():
    cfg = DefaultConfig()
    cfg_dict = {"env": {"num_envs": 22, "viewer": {"pos": (2.0, 2.0, 2.0)}}}
    with pytest.raises(KeyError):
        update_class_from_dict(cfg, cfg_dict)


def test_multiple_instances():
    # create two config instances
    cfg1 = DefaultConfig()
    cfg2 = DefaultConfig()

    # check variables
    # mutable -- variables should be different
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    assert id(cfg1.robot_default_state) != id(cfg2.robot_default_state)
    # immutable -- variables are the same
    assert id(cfg1.robot_default_state.dof_pos) == id(cfg2.robot_default_state.dof_pos)
    assert id(cfg1.env.num_envs) == id(cfg2.env.num_envs)


def test_alter_values_multiple_instances():
    # create two config instances
    cfg1 = DefaultConfig()
    cfg2 = DefaultConfig()

    # alter configurations
    cfg1.env.num_envs = 22  # immutable data: int
    cfg1.env.viewer.eye[0] = 1.0  # mutable data: list
    cfg1.env.viewer.lookat[2] = 12.0  # mutable data: list

    # check variables
    # values should be different
    assert cfg1.env.num_envs != cfg2.env.num_envs
    assert cfg1.env.viewer.eye != cfg2.env.viewer.eye
    assert cfg1.env.viewer.lookat != cfg2.env.viewer.lookat
    # mutable -- variables are different ids
    assert id(cfg1.env.viewer.eye) != id(cfg2.env.viewer.eye)
    assert id(cfg1.env.viewer.lookat) != id(cfg2.env.viewer.lookat)
    # immutable -- altered variables are different ids
    assert id(cfg1.env.num_envs) != id(cfg2.env.num_envs)


"""
Test config with functions
"""

from functools import wraps


def dummy_function1():
    return 1


def dummy_function2():
    return 2


def dummy_wrapper(func):
    @wraps(func)
    def wrapper():
        return func() + 1

    return wrapper


@dummy_wrapper
def wrapped_dummy_function3():
    return 3


@dummy_wrapper
def wrapped_dummy_function4():
    return 4


@configclass
class ConfigWithFunctions:
    func = dummy_function1
    wrapped_func = wrapped_dummy_function3
    func_in_dict = {"func": dummy_function1}


config_with_obs_correct = {
    "func": "__main__:dummy_function1",
    "wrapped_func": "__main__:wrapped_dummy_function3",
    "func_in_dict": {"func": "__main__:dummy_function1"},
}

config_with_obs_changed = {
    "func": "__main__:dummy_function2",
    "wrapped_func": "__main__:wrapped_dummy_function4",
    "func_in_dict": {"func": "__main__:dummy_function2"},
}


def test_config_with_funcitons():
    cfg = ConfigWithFunctions()
    assert cfg.func() == 1
    assert cfg.wrapped_func() == 4
    assert cfg.func_in_dict["func"]() == 1

    cfg_dict = class_to_dict(cfg)
    assert cfg_dict["func"] == config_with_obs_correct["func"]
    assert cfg_dict["wrapped_func"] == config_with_obs_correct["wrapped_func"]
    assert cfg_dict["func_in_dict"]["func"] == config_with_obs_correct["func_in_dict"]["func"]

    update_class_from_dict(cfg, config_with_obs_changed)
    assert cfg.func() == 2
    assert cfg.wrapped_func() == 5
    assert cfg.func_in_dict["func"]() == 2


test_config_with_funcitons()
