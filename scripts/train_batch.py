"""
Script for launching a training session over parameter space.
"""
# legged-gym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, class_to_dict
from legged_gym.utils.config_utils import BaseConfig

# python
import argparse
import yaml
import os
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError as e:
    print(e)


def train(args: argparse.Namespace, env_cfg: BaseConfig, train_cfg: BaseConfig):
    # initialize the runner and env
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # dump the configurations
    if not os.path.exists(ppo_runner.log_dir):
        os.makedirs(ppo_runner.log_dir, exist_ok=True)
    with open(os.path.join(ppo_runner.log_dir, "env.yaml"), "w+") as file:
        env_dict_cfg = class_to_dict(env_cfg)
        yaml.dump(env_dict_cfg, file)
    with open(os.path.join(ppo_runner.log_dir, "ppo.yaml"), "w+") as file:
        train_dict_cfg = class_to_dict(train_cfg)
        yaml.dump(train_dict_cfg, file)
    # run learning algorithm
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


def train_batch(args: argparse.Namespace):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    for i in range(5):
        # hyperparams to run over
        seed = 23 * i + 17
        train_cfg.seed = seed
        env_cfg.seed = seed
        # run name
        train_cfg.runner.run_name = f"_no_timeouts_{i}"
        # launch process
        p = Process(target=train, args=(args, env_cfg, train_cfg))
        p.start()
        p.join()
        p.kill()
        print(f">>> Run {i} done!")


if __name__ == "__main__":
    # set_np_formatting()
    args = get_args()
    train_batch(args)
