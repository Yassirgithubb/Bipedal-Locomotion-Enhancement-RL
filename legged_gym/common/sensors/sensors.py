import torch

from isaacgym.torch_utils import quat_apply
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.visualization_utils import BatchWireframeSphereGeometry
from legged_gym.utils.warp_utils import ray_cast
from .sensors_cfg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs import BaseEnv


class SensorBase:
    def __init__(self, cfg, env):
        # prepare some buffers
        # enable corresponding sensors in sim
        raise NotImplementedError()

    def update(self, dt: float, env_ids=None):
        # compute stuff
        raise NotImplementedError()

    def get_data(self):
        # returns sensor data
        raise NotImplementedError()

    def reset(self):
        pass


class Raycaster(SensorBase):
    def __init__(self, cfg: RaycasterCfg, env: "BaseEnv"):
        self.cfg = cfg
        self.terrain_mesh = env.terrain.wp_meshes[self.cfg.terrain_mesh_name]
        self.robot = getattr(env, cfg.robot_name)
        self.body_idx, _ = self.robot.find_bodies(cfg.body_attachement_name)
        self.num_envs = self.robot.num_envs
        self.device = self.robot.device

        self.ray_starts, self.ray_directions = cfg.pattern_cfg.pattern_func(cfg.pattern_cfg, self.device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(cfg.attachement_pos), device=self.device)
        offset_quat = torch.tensor(list(cfg.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)

        self.sphere_geom = None

    def update(self, dt, env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        states = self.robot.rigid_body_states[env_ids, self.body_idx, :].squeeze(1)
        pos = states[..., :3]
        quats = states[..., 3:7]
        if self.cfg.attach_yaw_only:
            ray_starts_world = quat_apply_yaw(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(
                1
            )
            ray_directions_world = self.ray_directions[env_ids]
        else:
            ray_starts_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)
            ray_directions_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self.ray_hits_world[env_ids] = ray_cast(ray_starts_world, ray_directions_world, self.terrain_mesh)

    def get_data(self):
        return torch.nan_to_num(self.ray_hits_world, posinf=self.cfg.default_hit_value)

    def debug_vis(self, env: "BaseEnv"):
        if self.sphere_geom is None:
            self.sphere_geom = BatchWireframeSphereGeometry(
                self.num_envs * self.num_rays, 0.02, 4, 4, None, color=(0, 1, 0)
            )
        self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0])
