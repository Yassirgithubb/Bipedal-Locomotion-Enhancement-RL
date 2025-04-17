# python
import numpy as np
import trimesh

# isaacgym
from functools import wraps
from isaacgym import terrain_utils

# legged_gym
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_cfg import SubTerrainsCfg


def heightfield_terrain(func):
    @wraps(func)
    def wrapper(difficulty, cfg):
        # prepare helper class
        width_pixels = int(cfg.width / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.length / cfg.horizontal_scale) + 1
        border_pixels = int(cfg.border_size / cfg.horizontal_scale) + 1
        height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=width_pixels - 2 * border_pixels,
            length=length_pixels - 2 * border_pixels,
            vertical_scale=cfg.vertical_scale,
            horizontal_scale=cfg.horizontal_scale,
        )
        # generate the heightfield
        func(terrain, difficulty, cfg)
        height_field[border_pixels:-border_pixels, border_pixels:-border_pixels] = terrain.height_field_raw
        # convert to trimesh
        vertices, triangles, = terrain_utils.convert_heightfield_to_trimesh(
            height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_treshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # compute origin
        x1 = int((cfg.length / 2.0 - 1) / cfg.horizontal_scale)
        x2 = int((cfg.length / 2.0 + 1) / cfg.horizontal_scale)
        y1 = int((cfg.width / 2.0 - 1) / cfg.horizontal_scale)
        y2 = int((cfg.width / 2.0 + 1) / cfg.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * cfg.vertical_scale
        origin = np.array([0.5 * cfg.length, 0.5 * cfg.width, env_origin_z])
        return mesh, origin

    return wrapper


@heightfield_terrain
def hf_pyramid_slope(terrain, difficulty, cfg: "SubTerrainsCfg.HfPyramidSlopeCfg"):
    slope = cfg.min_slope + difficulty * (cfg.max_slope - cfg.min_slope)
    terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=cfg.platform_size)
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=cfg.min_height_noise,
        max_height=cfg.max_height_noise,
        step=cfg.noise_step,
        downsampled_scale=cfg.downsampled_scale,
    )


@heightfield_terrain
def hf_pyramid_slope_inv(terrain, difficulty, cfg):
    slope = -cfg.min_slope - difficulty * (cfg.max_slope - cfg.min_slope)
    terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=cfg.platform_size)
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=cfg.min_height_noise,
        max_height=cfg.max_height_noise,
        step=cfg.noise_step,
        downsampled_scale=cfg.downsampled_scale,
    )


@heightfield_terrain
def hf_pyramid_stairs(terrain, difficulty, cfg):
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    terrain_utils.pyramid_stairs_terrain(
        terrain, step_width=cfg.step_width, step_height=step_height, platform_size=cfg.platform_size
    )


@heightfield_terrain
def hf_pyramid_stairs_inv(terrain, difficulty, cfg):
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    step_height *= -1
    terrain_utils.pyramid_stairs_terrain(
        terrain, step_width=cfg.step_width, step_height=step_height, platform_size=cfg.platform_size
    )


@heightfield_terrain
def hf_discrete_obstacles(terrain, difficulty, cfg):
    height = cfg.rectangle_min_height + (cfg.rectangle_max_height - cfg.rectangle_min_height) * difficulty
    terrain_utils.discrete_obstacles_terrain(
        terrain,
        height,
        cfg.rectangle_min_size,
        cfg.rectangle_max_size,
        cfg.num_rectangles,
        platform_size=cfg.platform_size,
    )


@heightfield_terrain
def hf_stepping_stones(terrain, difficulty, cfg):
    stepping_stones_size = cfg.stones_max_size - (cfg.stones_max_size - cfg.stones_min_size) * (1 - difficulty)
    stone_distance = cfg.stones_min_distance + (cfg.stones_max_distance - cfg.stones_min_distance) * difficulty
    terrain_utils.stepping_stones_terrain(
        terrain,
        stone_size=stepping_stones_size,
        stone_distance=stone_distance,
        max_height=cfg.max_height,
        platform_size=cfg.platform_size,
    )
