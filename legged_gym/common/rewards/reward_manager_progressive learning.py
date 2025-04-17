import torch


class ProgressiveRewardManager:
    def __init__(self, env, num_tasks):
        """
        Reward manager that supports multiple tasks and progressive learning.
        Each task has its own set of relevant reward functions.
        """
        self.num_tasks = num_tasks
        self.only_positive_rewards = env.cfg.rewards.only_positive_rewards
        self.reward_functions = {}
        self.reward_params = {}
        self.episode_sums = [{} for _ in range(num_tasks)]

        # Prepare task-specific reward masks
        for name, params in env.cfg.rewards.__dict__.items():
            if not isinstance(params, dict) or params["scale"] == 0:
                continue
            params["scale"] *= env.dt
            func = params["func"]
            if "dofs" in params:
                params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
            if "bodies" in params:
                params["body_indices"], _ = env.robot.find_bodies(params["bodies"])
            task_code = params.get("task_code", -1)

            self.reward_functions[name] = func
            self.reward_params[name] = params

            # Register sensors if needed
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])

            # Assign this reward function to its relevant task(s)
            if task_code == 0:
                # Task 0: shared across all tasks
                for task_id in range(num_tasks):
                    self.episode_sums[task_id][name] = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            else:
                self.episode_sums[task_code][name] = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

        # Active task ID per environment (set externally during training)
        self.active_task_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def set_active_tasks(self, task_ids):
        """Sets the current task index (0..N-1) for each environment."""
        self.active_task_ids.copy_(task_ids)

    def compute_progressive_reward(self, env):
        """Compute task-specific rewards for each environment."""
        total_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

        for name, func in self.reward_functions.items():
            if name == "termination":
                continue  # handled separately
            params = self.reward_params[name]
            task_code = params.get("task_code", -1)

            rew = func(env, params) * params["scale"]

            # Apply reward only to environments training on the right task
            if task_code == 0:
                total_reward += rew
                for task_id in range(self.num_tasks):
                    self.episode_sums[task_id][name] += rew
            else:
                task_mask = self.active_task_ids == task_code
                total_reward[task_mask] += rew[task_mask]
                self.episode_sums[task_code][name][task_mask] += rew[task_mask]

        if self.only_positive_rewards:
            total_reward = total_reward.clip(min=0.0)

        # Add termination rewards after clipping
        if "termination" in self.reward_functions:
            params = self.reward_params["termination"]
            rew = self.reward_functions["termination"](env, params) * params["scale"]
            total_reward += rew
            for task_id in range(self.num_tasks):
                self.episode_sums[task_id]["termination"][self.active_task_ids == task_id] += rew[self.active_task_ids == task_id]

        return total_reward

    def log_info(self, env, env_ids, extras_dict):
        """Logs mean rewards for each task and resets episode sums for those env_ids."""
        for task_id in range(self.num_tasks):
            for name in self.episode_sums[task_id].keys():
                mean_reward = torch.mean(self.episode_sums[task_id][name][env_ids]) / env.max_episode_length_s
                extras_dict[f"rew_task{task_id}_{name}"] = mean_reward
                self.episode_sums[task_id][name][env_ids] = 0.0