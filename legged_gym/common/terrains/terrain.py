import torch
import numpy as np
import trimesh
from collections import defaultdict

from legged_gym.utils import warp_utils
from legged_gym.common.gym_interface import GymInterface
from .terrain_cfg import TerrainCfg

from isaacgym import gymapi


class Terrain:
    def __init__(self, cfg: TerrainCfg, num_envs: int, gym_iface: GymInterface) -> None:
        self.cfg = cfg
        self.gym = gym_iface.gym
        self.sim = gym_iface.sim
        self.device = gym_iface.device
        self.num_envs = num_envs
        self.meshes = defaultdict(trimesh.Trimesh)
        self.wp_meshes = {}
        self.terrain_origins = None

    def add_mesh(self, mesh, name="terrain"):
        if self.meshes[name]:
            mesh = trimesh.util.concatenate(self.meshes[name], mesh)
        self.meshes[name] = mesh
        wp_device = "cuda" if "cuda" in self.device else "cpu"
        self.wp_meshes[name] = warp_utils.convert_to_wp_mesh(mesh.vertices, mesh.faces, wp_device)

    def set_terrain_origins(self, terrain_origins):
        if isinstance(terrain_origins, np.ndarray):
            terrain_origins = torch.from_numpy(terrain_origins).to(torch.float)
        self.terrain_origins = terrain_origins.to(self.device)
        self.env_origins = self._compute_env_origins()

    def add_to_sim(self, name="terrain"):
        mesh_type = self.cfg.mesh_type
        if mesh_type == "plane":
            self._add_ground_plane_to_sim()
        elif mesh_type == "trimesh":
            self._add_trimesh_to_sim(self.meshes[name])
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, trimesh]")

    def _add_ground_plane_to_sim(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.static_friction
        plane_params.dynamic_friction = self.cfg.dynamic_friction
        plane_params.restitution = self.cfg.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _add_trimesh_to_sim(self, mesh):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.faces).astype(np.uint32)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]

        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(order="C"), triangles.flatten(order="C"), tm_params)

    def _compute_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.mesh_type in ["heightfield", "trimesh"]:
            env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            # put robots at the origins defined by the terrain
            max_init_level = min(self.cfg.max_init_terrain_level, self.cfg.num_rows - 1)
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.num_rows
            env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env_spacing
            env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            #env_origins[:, 2] = 0.0
            #increase height when spawning since the robot is tilted
            env_origins[:, 2] = 0.0+0.2
        return env_origins

    def update_terrain_levels(self, env_ids, move_up, move_down):
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
