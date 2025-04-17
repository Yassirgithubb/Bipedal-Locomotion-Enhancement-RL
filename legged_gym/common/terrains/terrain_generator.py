# python
import numpy as np
import trimesh

# legged-gym
from legged_gym.common.terrains.heightfield_terrains import *
from legged_gym.common.terrains.mesh_terrains import *
from .terrain_cfg import TerrainCfg, SubTerrainsCfg


class TerrainGenerator:
    def __init__(self, cfg: TerrainCfg, curriculum: bool = False):

        self.cfg = cfg
        self.type = cfg.mesh_type
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # set some common values to all sub-terrains
        for key, val in self.cfg.sub_terrains.__dict__.items():
            val.length = self.cfg.terrain_length
            val.width = self.cfg.terrain_width
            val.horizontal_scale = self.cfg.horizontal_scale
            val.vertical_scale = self.cfg.vertical_scale
            val.slope_threshold = self.cfg.slope_threshold
        # list of all meshes
        self.terrain_meshes = []
        if self.type == "plane":  # FIXME
            cfg = SubTerrainsCfg.PlaneCfg(length=2.0e6, width=2.0e6)
            mesh, _ = plane(0.0, cfg)
            self.terrain_meshes.append(mesh)
        else:
            if curriculum:
                self._curriculum()
            else:
                self._randomized_terrain()

            self._add_world_border()

        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

    def _add_world_border(self):
        """Add a surrounding border around all the terrains."""
        border_meshes = border_mesh(
            self.cfg.num_rows * self.cfg.terrain_length + 2 * self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width + 2 * self.cfg.border_size,
            self.cfg.num_rows * self.cfg.terrain_length,
            self.cfg.num_cols * self.cfg.terrain_width,
            1.0,
            [self.cfg.num_rows * self.cfg.terrain_length / 2, self.cfg.num_cols * self.cfg.terrain_width / 2, -0.5],
        )
        border = trimesh.util.concatenate(border_meshes)
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        self.terrain_meshes.append(border)

    def _randomized_terrain(self):
        sub_terrain_dict = self.cfg.sub_terrains.__dict__
        proportions = np.array([val.proportion for key, val in sub_terrain_dict.items()])
        proportions /= np.sum(proportions)
        sub_terrain_names = list(sub_terrain_dict.keys())

        for k in range(self.cfg.num_rows * self.cfg.num_cols):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            sub_terrain_idx = np.random.choice(np.arange(len(proportions)), p=proportions)

            difficulty = np.random.choice([0.5, 0.75, 0.9])
            name = sub_terrain_names[sub_terrain_idx]
            cfg = sub_terrain_dict[name]
            function = cfg.func

            # create the terrain
            terrain_mesh, env_origin = function(difficulty, cfg)
            self._add_sub_terrain(terrain_mesh, env_origin, i, j)

    def _curriculum(self):
        sub_terrain_dict = self.cfg.sub_terrains.__dict__
        proportions = np.array([val.proportion for key, val in sub_terrain_dict.items()])
        proportions /= np.sum(proportions)

        sub_terrain_indices = np.array(
            [
                np.min(np.where(i / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                for i in range(self.cfg.num_cols)
            ]
        ).astype(np.int)
        sub_terrain_cfgs = [val for key, val in sub_terrain_dict.items()]

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                cfg = sub_terrain_cfgs[sub_terrain_indices[j]]
                function = cfg.func
                # create the terrain
                terrain_mesh, env_origin = function(difficulty, cfg)
                self._add_sub_terrain(terrain_mesh, env_origin, i, j)

    def _add_sub_terrain(self, mesh, env_origin, row, col):
        pose = np.eye(4)
        pose[:2, -1] = [row * self.cfg.terrain_length, col * self.cfg.terrain_width]
        mesh.apply_transform(pose)
        self.terrain_meshes.append(mesh)

        self.terrain_origins[row, col] = [
            env_origin[0] + row * self.cfg.terrain_length,
            env_origin[1] + col * self.cfg.terrain_width,
            env_origin[2],
        ]
