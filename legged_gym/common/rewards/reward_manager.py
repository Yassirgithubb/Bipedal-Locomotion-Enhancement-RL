import torch


class RewardManager:
    def __init__(self, env):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self.<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_functions = {}
        self.reward_params = {}
        self.reward = 0.0
        self.only_positive_rewards = env.cfg.rewards.only_positive_rewards
        for name, params in env.cfg.rewards.__dict__.items():
            if not isinstance(params, dict) or params["scale"] == 0:
                continue
            params["scale"] *= env.dt
            # function = getattr(self, params["func_name"])
            function = params["func"]
            if "dofs" in params.keys():
                params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
            if "bodies" in params.keys():
                params["body_indices"], _ = env.robot.find_bodies(params["bodies"])

            self.reward_functions[name] = function
            self.reward_params[name] = params
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                env.num_envs,
                dtype=torch.float,
                device=env.device,
                requires_grad=False,
            )
            for name in self.reward_functions.keys()
        }

    def compute_reward(self, env):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self.__init__())
        adds each terms to the episode sums and to the total reward
        """
        self.reward = 0.0
        for name, function in self.reward_functions.items():
            if name == "termination":
                continue  # handled separately after clipping
            params = self.reward_params[name]
            rew = function(env, params) * params["scale"]
            self.reward += rew
            self.episode_sums[name] += rew
        if self.only_positive_rewards:
            self.reward = self.reward.clip(min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_functions:
            params = self.reward_params["termination"]
            rew = self.reward_functions["termination"](env, params) * params["scale"]
            self.reward += rew
            self.episode_sums["termination"] += rew
        return self.reward

    def log_info(self, env, env_ids, extras_dict):
        # fill env extras with episode sum of each reward
        for key in self.episode_sums.keys():
            extras_dict["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / env.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
