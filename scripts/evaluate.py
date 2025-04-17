"""
Evaluate a trained policy over different terrains.
"""

# legged-gym
from legged_gym.utils import get_args
from legged_gym.envs import task_registry
from legged_gym.envs.locomotion.legged_env_config import LeggedRobotCfg, LeggedRobotCfgPPO

# python
import argparse
import torch
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError as e:
    print(e)


def run(args: argparse.Namespace, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
    """Run evaluation of policy as single-process.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        env_cfg (LeggedRobotCfg): Environment configuration.
        train_cfg (LeggedRobotCfgPPO): Training agent configuration.
    """
    # create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy()
    # buffer to store intermediate stats
    device = env.device
    masks = torch.ones(env.num_envs, device=device, requires_grad=False)
    solved = torch.zeros(env.num_envs, device=device, requires_grad=False).to(torch.bool)
    num_crashes = 0

    env.keyboard_controller.print_options()

    # reset environment
    obs, _ = env.reset()
    # run episode
    for i in range(int(env.max_episode_length) - 1):
        env.update_keyboard_events()
        # acquire actions
        actions = policy(obs.detach())
        # stop moving after a done
        actions *= masks.to(env.device).unsqueeze(1)
        # step through environment
        obs, _, _, dones, infos = env.step(actions.detach())
        # compute statistics
        if "time_outs" in infos:
            time_outs = infos["time_outs"]
        else:
            time_outs = torch.Tensor([False], device=device)
        crashes = ~time_outs.squeeze() * dones
        masks *= ~crashes
        num_crashes += torch.sum(crashes * (~solved))
        distance = torch.norm(env.root_states[:, :2] - env.env_origins[:, :2], dim=1)
        solved |= distance > (env_cfg.terrain.border_size / 2.0)
    # evaluate stats
    num_solves = torch.sum(solved).item() / env.num_envs * 100
    num_crashes = num_crashes / env.num_envs * 100
    height = env_cfg.terrain.terrain_kwargs["step_height"]
    # print results
    print("=" * 20)
    print(f"Height: {height:0.2f}, solved: {num_solves:0.2f}%, crashes: {num_crashes:0.4f}%")


def evaluate(args: argparse.Namespace):
    """Evaluate a trained policy over various terrains.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    # get configuration for task
    env_cfg: LeggedRobotCfg
    train_cfg: LeggedRobotCfgPPO
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for evaluation
    env_cfg.env.num_envs = 1000
    env_cfg.terrain.num_levels = 1
    env_cfg.terrain.num_terrains = 20
    env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.terrain_kwargs = {
        "type": "pyramid_stairs_terrain",
        "step_width": 0.31,
        "step_height": -0.18,
        "platform_size": 3.0,
    }
    env_cfg.commands.ranges.lin_vel_x = [0.75, 0.75]
    env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    env_cfg.seed = 2
    # change default settings from parsed argument (hack)
    # args.headless = True
    # run processes with various test heights for stairs
    # step_heights = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    step_heights = [-0.2]
    for height in step_heights:
        env_cfg.terrain.terrain_kwargs["step_height"] = height
        p = Process(target=run, args=(args, env_cfg, train_cfg))
        p.start()
        p.join()
        p.kill()
        print("Done running with stairs of height: ", height)


if __name__ == "__main__":
    # set_np_formatting()
    args = get_args()
    evaluate(args)
