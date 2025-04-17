"""Installation script for the 'legged_gym' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "isaacgym",
    "rsl-rl",
    "matplotlib",
    "tensorboard",
    "trimesh",
    "warp-lang",
    "pytest",
    "pygame",
]

# Installation operation
setup(
    name="legged_gym",
    version="1.1.0",
    author="Nikita Rudin",
    packages=find_packages(),
    author_email="nikitar@leggedrobotics.com",
    description="Isaac Gym environments for Legged Robots",
    install_requires=INSTALL_REQUIRES,
)
