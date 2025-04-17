"""
Test whether environment runs with random actions.
"""
# legged-gym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg

# python
import argparse
import torch


def test_env(args: argparse.Namespace):
    env_cfg: LeggedEnvCfg
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_cols = 5
    env_cfg.env.enable_debug_vis = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # simulate steps
    for _ in range(int(10 * env.max_episode_length)):
        actions = 0.0 * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, info = env.step(actions)

    print("Done!")


if __name__ == "__main__":
    args = get_args()
    test_env(args)
