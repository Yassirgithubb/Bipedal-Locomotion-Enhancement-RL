import torch


class ObsManager:
    def __init__(self, env):
        self.obs_per_group = {}
        self.obs_dims_per_group = {}
        self.obs = {}
        self.cfg = env.cfg.observations
        obs_groups = self.cfg.__dict__
        for group_name, obs_group in obs_groups.items():
            self.obs_per_group[group_name] = []
            obs_dim = 0
            add_noise = obs_group.add_noise
            for _, params in obs_group.__dict__.items():
                if not isinstance(params, dict):
                    continue
                if not add_noise:  # turn off all noise
                    params["noise"] = None
                # function = getattr(self, params["func_name"])
                function = params["func"]
                # if function is a string evaluate it, note: it must be imported in the manager module
                if "dofs" in params.keys():
                    params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
                if "bodies" in params.keys():
                    params["body_indices"], _ = env.robot.find_bodies(params["bodies"])
                self.obs_per_group[group_name].append((function, params))
                if params.get("sensor") is not None:
                    env.enable_sensor(params["sensor"])
                obs_dim += function(env, params).shape[1]
            self.obs_dims_per_group[group_name] = obs_dim

    def compute_obs(self, env):
        self.obs = {}
        for group, function_list in self.obs_per_group.items():
            obs_list = []
            for function, params in function_list:
                obs = function(env, params)
                noise = params.get("noise")
                clip = params.get("clip")
                scale = params.get("scale")
                if noise:
                    obs = self._add_uniform_noise(obs, noise)
                if clip:
                    obs = obs.clip(min=clip[0], max=clip[1])
                if scale:
                    obs = scale * obs
                obs_list.append(obs)
            self.obs[group] = torch.cat(obs_list, dim=1)
        return self.obs

    def _add_uniform_noise(self, obs, noise_level):
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
