"""
Test whether environment runs with random actions.
"""
from isaacgym import gymtorch

# legged-gym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.locomotion import LeggedEnv

# python
import argparse
import torch

import contextlib
from time import time


@contextlib.contextmanager
def timer(str):

    start = time()
    yield
    print("---")
    print(str, f" took {time()-start}")


def test(args: argparse.Namespace):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env: LeggedEnv
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    gym = env.gym
    sim = env.sim
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(10):
        env.step(actions)

    # reset indexed
    env_ids = torch.arange(env.num_envs, device=env.device)
    root_states = env.robot.state.root_states.clone()
    root_states[:, 7:] += +0.001
    with timer("indexed reset"):
        gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids),
        )
        # gym.simulate(sim)

    with timer("reset all"):
        gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_states))
        # gym.simulate(sim)

    with timer("refresh dof state"):
        gym.refresh_dof_state_tensor(sim)
    with timer("refresh root state"):
        gym.refresh_actor_root_state_tensor(sim)
    with timer("refresh net contact forces"):
        gym.refresh_net_contact_force_tensor(sim)
    with timer("refresh rigid_body state"):
        gym.refresh_rigid_body_state_tensor(sim)

    with timer("simulation step"):
        gym.simulate(sim)

    with timer("actuator net"):
        env.robot._actuators[0].compute_torque()


if __name__ == "__main__":
    args = get_args()
    test(args)
