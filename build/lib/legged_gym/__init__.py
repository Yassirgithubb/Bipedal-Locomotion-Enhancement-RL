import os

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""Absolute path to the legged-gym repository."""

LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs")
"""Absolute path to the module `legged_gym.envs` in legged-gym repository."""
