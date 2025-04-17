"""
Times various segments of simulation.
"""
# isaac-gym
from isaacgym import gymtorch

# python
import os
from time import time
import torch
from torch.multiprocessing import Process, set_start_method
import contextlib
from collections import defaultdict

# legged-gym
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs.locomotion.legged_env import LeggedEnv
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg


try:
    set_start_method("spawn")
except RuntimeError as e:
    print(e)

# global flags
LOG_TIMING_DATA = False


@contextlib.contextmanager
def timer(name, times_dict):

    torch.cuda.synchronize()
    start = time()
    yield
    times_dict[name] += time() - start


# def timed_reset(env: LeggedEnv, actions: )


def timed_step(env: LeggedEnv, actions: torch.Tensor, times: dict) -> tuple:
    actions = torch.clip(actions, -env.cfg.control.action_clipping, env.cfg.control.action_clipping)
    actions = actions.to(env.device)
    env.actions = actions
    # step physics and render each frame
    env.render()
    # pre-process actions
    # -- default scaling of actions
    scaled_actions = env.cfg.control.action_scale * env.actions
    # -- environment specific pre-processing
    processed_actions = env._preprocess_actions(scaled_actions)
    # apply actions into simulator
    for _ in range(env.cfg.control.decimation):
        # may include recomputing torques (based on actuator models)
        with timer(times_dict=times, name="apply actions"):
            env._apply_actions(processed_actions)
        # simulation step
        with timer(times_dict=times, name="simulate"):
            env.gym_iface.simulate()
        # refresh tensors
        with timer(times_dict=times, name="refresh tensors"):
            env.gym_iface.refresh_tensors(dof_state=True)
    # update sim counters
    env.episode_length_buf += 1

    # post-physics computation
    with timer(times_dict=times, name="refresh tensors"):
        env.gym_iface.refresh_tensors(
            root_state=True,
            net_contact_force=True,
            rigid_body_state=True,
            dof_state=True,
            dof_torque=env.robot.has_dof_torque_sensors,
        )
    # update env counters (used for curriculum generation)
    env.common_step_counter += 1
    # update robot
    with timer(times_dict=times, name="update robot buffers"):
        env.robot.update_buffers(env.dt)
    # update sensors
    with timer(times_dict=times, name="refresh sensors"):
        for _, s in env.sensors.items():
            s.update(env.dt)
    # rewards, resets, ...
    # -- terminations
    with timer(times_dict=times, name="check termination"):
        env.reset_buf = env.termination_manager.check_termination(env)
        env.time_out_buf = env.episode_length_buf >= env.max_episode_length  # no terminal reward for time-outs
        env.reset_buf |= env.time_out_buf
        env_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
    # -- rewards
    with timer(times_dict=times, name="reward"):
        env.rew_buf = env.reward_manager.compute_reward(env)
    # -- reset terminated environments
    with timer(times_dict=times, name="reset"):
        env_ids = env.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        if len(env_ids) != 0:
            # -- update curriculum
            if env._init_done:
                env.curriculum_manager.update_curriculum(env, env_ids)
            # reset terminated environments
            env.reset_idx(env_ids)
            # # re-update robots for envs that were reset
            env.robot.update_buffers(env.dt, env_ids)
            # # re-update sensors for envs that were reset
            for _, s in env.sensors.items():
                s.update(env_ids)
            print(env_ids)
    # set randomization
    with timer(times_dict=times, name="push robots"):
        if env.cfg.randomization.push_robots and env.common_step_counter % env._push_interval == 0:
            env._push_robots()
            env.gym_iface.write_states_to_sim()
    # compute observations
    # resample commands if time has come
    with timer(times_dict=times, name="commands"):
        env_ids = env.episode_length_buf % int(env.cfg.commands.resampling_time / env.dt) == 0
        env_ids = env_ids.nonzero(as_tuple=False).flatten().tolist()
        env._resample_commands(env_ids)
        env._update_commands()
    # in some cases a simulation step might be required to refresh some obs (for example body positions)
    with timer(times_dict=times, name="obs"):
        env.obs_dict = env.obs_manager.compute_obs(env)
        env.obs_buf = env.obs_dict["policy"]
        env.extras["observations"] = env.obs_dict
    # update history
    with timer(times_dict=times, name="update history"):
        env.last_actions[:] = env.actions[:]

    # return mdp tuples
    return (env.obs_buf, env.rew_buf, env.reset_buf, env.extras)


def test_timings(args, env_cfg: LeggedEnvCfg, train_cfg):
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # specifications
    num_runs = 10
    # empty string to write into
    if LOG_TIMING_DATA:
        write_string = "{}:\n".format(env_cfg.env.num_envs)
    # run simulations
    for k in range(num_runs):
        print(f"------ Starting: {k + 1}/{num_runs} ------")
        # buffer to log stats
        times = defaultdict(lambda: 0.0)
        # reset environment
        # perform sim steps
        num_steps = 100
        for _ in range(num_steps):
            with timer(times_dict=times, name="inference"):
                actions = policy(obs.detach())

            obs, _, _, _ = timed_step(env, actions.detach(), times)
        # print dictionary of times
        print(f"------ Finished: {k + 1}/{num_runs} ------")
        total = sum([val for val in times.values()])
        for key, value in times.items():
            print(f"{key}: {value / num_steps:0.5f}s ({100*value/total:0.5f}%)")
        print(f"total: {total / num_steps:0.5f}s")
        # write down the results into string for logging
        if LOG_TIMING_DATA:
            write_string = "{"
            for key, value in times.items():
                write_string += """'{}': {}, """.format(key, value / num_steps)
            write_string += "}"
        # end of run

    # log into a file
    if LOG_TIMING_DATA:
        # log directory
        write_timings_path = os.path.join(ppo_runner.log_dir, "timings.txt")
        # write into file
        with open(write_timings_path, "w") as f:
            f.write(write_string + "\n")


def run():
    # load configurations for task
    env_cfg: LeggedEnvCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # modify parameters
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.observations.policy.add_noise = False
    # args.headless = True
    args.num_envs = 100

    # run comparisons based on number of robots
    # num_robots = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    num_robots = [4096]
    for num in num_robots:
        print(f">>>> Evaluation with {num} environments.")
        env_cfg.env.num_envs = num
        p = Process(target=test_timings, args=(args, env_cfg, train_cfg))
        p.start()
        p.join()
        p.kill()


if __name__ == "__main__":
    args = get_args()
    run()
