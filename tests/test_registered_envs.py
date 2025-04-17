"""
Test all environments in the task registry.
"""
# legged-gym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args

# python
import argparse
import torch
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError as e:
    print(e)

"""
Tests
"""


def test_zero_actions(args: argparse.Namespace):
    """Creates environment and applies no actions."""
    # get environment config
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # reset environment
    obs, _ = env.reset()
    assert not bool(torch.any(torch.isnan(obs))), "[FAILURE]: Invalid `obs` tensor found!"
    # simulate steps
    for _ in range(int(env.max_episode_length)):
        actions = 0.0 * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, _ = env.step(actions)
        # check no tensors are nan
        assert not bool(torch.any(torch.isnan(obs))), "[FAILURE]: Invalid `obs` tensor found!"
        assert not bool(torch.any(torch.isnan(rew))), "[FAILURE]: Invalid `rew` tensor found!"
        assert not bool(torch.any(torch.isnan(done))), "[FAILURE]: Invalid `done` tensor found!"


if __name__ == "__main__":
    # parse cli args
    args = get_args()
    args.headless = True
    # iterate over all tasks registered
    for task_name in task_registry.get_task_names():
        # set current task
        args.task = task_name
        # launch process
        p = Process(target=test_zero_actions, args=(args,))
        p.start()
        p.join()
        p.kill()
        print(f">>> Testing complete for task: {task_name}!")
