# python
import numpy as np
import torch
import trimesh

from scipy.spatial.transform import Rotation as Rot

# legged-gym
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_cfg import SubTerrainsCfg


def border_mesh(
    outer_length: float, outer_width: float, inner_length: float, inner_width: float, height: float, pos=[0.0, 0.0, 0.0]
):
    """
    Creates a rectangular border mesh with a rectangular hole

         ________________________
        |                        |
        |      ____________      | w
        |     |            |     | i
        |     |            |     | d
        |     |____________|     | t
        |                        | h
        |________________________|
                 length


    """
    thickness_x = (outer_length - inner_length) / 2
    tickness_y = (outer_width - inner_width) / 2

    dims = [outer_length, tickness_y, height]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1] + (tickness_y + inner_width) / 2, pos[2]]
    box_north = trimesh.creation.box(dims, pose)
    pose[:3, -1] = [pos[0], pos[1] - (tickness_y + inner_width) / 2, pos[2]]
    box_south = trimesh.creation.box(dims, pose)
    dims = [thickness_x, outer_width, height]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0] + (thickness_x + inner_length) / 2, pos[1], pos[2]]
    box_east = trimesh.creation.box(dims, pose)
    pose[:3, -1] = [pos[0] - (thickness_x + inner_length) / 2, pos[1], pos[2]]
    box_west = trimesh.creation.box(dims, pose)
    boxes = [box_east, box_west, box_north, box_south]
    return boxes


def pyramid_stairs(difficulty, cfg: "SubTerrainsCfg.PyramidStairsCfg"):
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    num_steps = (
        min(cfg.length - 2 * cfg.border_size - cfg.platform_size, cfg.width - 2 * cfg.border_size - cfg.platform_size)
        // (2 * cfg.step_width)
        + 1
    )

    pos = [0.5 * cfg.length, 0.5 * cfg.width, 0.0]
    meshes = []
    border_meshes = border_mesh(
        cfg.length,
        cfg.width,
        cfg.length - 2 * (cfg.border_size),
        cfg.width - 2 * (cfg.border_size),
        step_height,
        [pos[0], pos[1], -step_height / 2],
    )
    meshes += border_meshes

    env_length = cfg.length - 2 * cfg.border_size
    env_width = cfg.width - 2 * cfg.border_size
    for k in range(int(num_steps)):

        if cfg.holes:
            box_length = cfg.platform_size
            box_width = cfg.platform_size
        else:
            box_length = env_length - k * 2 * cfg.step_width
            box_width = env_width - k * 2 * cfg.step_width

        dims = [box_length, cfg.step_width, (k + 2) * step_height]
        pose = np.eye(4)
        pose[:3, -1] = [pos[0], pos[1] + env_width / 2 - (k + 0.5) * cfg.step_width, pos[2] + k * step_height / 2]
        box_north = trimesh.creation.box(dims, pose)
        pose[:3, -1] = [pos[0], pos[1] - env_width / 2 + (k + 0.5) * cfg.step_width, pos[2] + k * step_height / 2]
        box_south = trimesh.creation.box(dims, pose)

        dims = [cfg.step_width, box_width, (k + 2) * step_height]
        pose[:3, -1] = [pos[0] - env_length / 2 + (k + 0.5) * cfg.step_width, pos[1], pos[2] + k * step_height / 2]
        box_east = trimesh.creation.box(dims, pose)
        pose[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * cfg.step_width, pos[1], pos[2] + k * step_height / 2]
        box_west = trimesh.creation.box(dims, pose)

        meshes += [box_north, box_south, box_east, box_west]

    dims = [env_length - 2 * (k + 1) * cfg.step_width, env_width - 2 * (k + 1) * cfg.step_width, (k + 2) * step_height]
    pose[:3, -1] = [pos[0], pos[1], pos[2] + k * step_height / 2]
    box_middle = trimesh.creation.box(dims, pose)
    meshes.append(box_middle)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], (num_steps + 1) * step_height])
    return meshes, origin


def pyramid_stairs_inv(difficulty, cfg: "SubTerrainsCfg.PyramidStairsInvCfg"):
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    num_steps = (
        min(cfg.length - 2 * cfg.border_size - cfg.platform_size, cfg.width - 2 * cfg.border_size - cfg.platform_size)
        // (2 * cfg.step_width)
        + 2
    )

    pos = [0.5 * cfg.length, 0.5 * cfg.width, 0.0]
    tot_height = num_steps * step_height
    meshes = []
    border_meshes = border_mesh(
        cfg.length,
        cfg.width,
        cfg.length - 2 * (cfg.border_size),
        cfg.width - 2 * (cfg.border_size),
        tot_height,
        [pos[0], pos[1], -tot_height / 2],
    )
    meshes += border_meshes

    env_length = cfg.length - 2 * cfg.border_size
    env_width = cfg.width - 2 * cfg.border_size
    for k in range(int(num_steps) - 1):

        if cfg.holes:
            box_length = cfg.platform_size
            box_width = cfg.platform_size
        else:
            box_length = env_length - k * 2 * cfg.step_width
            box_width = env_width - k * 2 * cfg.step_width

        dims = [box_length, cfg.step_width, tot_height - (k + 1) * step_height]
        pose = np.eye(4)
        pose[:3, -1] = [
            pos[0],
            pos[1] + env_width / 2 - (k + 0.5) * cfg.step_width,
            pos[2] - tot_height / 2 - (k + 1) * step_height / 2,
        ]
        box_north = trimesh.creation.box(dims, pose)
        pose[:3, -1] = [
            pos[0],
            pos[1] - env_width / 2 + (k + 0.5) * cfg.step_width,
            pos[2] - tot_height / 2 - (k + 1) * step_height / 2,
        ]
        box_south = trimesh.creation.box(dims, pose)

        dims = [cfg.step_width, box_width, tot_height - (k + 1) * step_height]
        pose[:3, -1] = [
            pos[0] - env_length / 2 + (k + 0.5) * cfg.step_width,
            pos[1],
            pos[2] - tot_height / 2 - (k + 1) * step_height / 2,
        ]
        box_east = trimesh.creation.box(dims, pose)
        pose[:3, -1] = [
            pos[0] + env_length / 2 - (k + 0.5) * cfg.step_width,
            pos[1],
            pos[2] - tot_height / 2 - (k + 1) * step_height / 2,
        ]
        box_west = trimesh.creation.box(dims, pose)

        meshes += [box_north, box_south, box_east, box_west]

    dims = [
        env_length - 2 * (k + 1) * cfg.step_width,
        env_width - 2 * (k + 1) * cfg.step_width,
        tot_height - (k + 1) * step_height,
    ]
    pose[:3, -1] = [pos[0], pos[1], pos[2] - tot_height / 2 - (k + 1) * step_height / 2]
    box_middle = trimesh.creation.box(dims, pose)
    meshes.append(box_middle)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], -(num_steps - 1) * step_height])
    return meshes, origin


def boxes(difficulty, cfg: "SubTerrainsCfg.BoxesCfg"):
    box_height = cfg.min_box_height + (cfg.max_box_height - cfg.min_box_height) * difficulty

    pos = [0.5 * cfg.length, 0.5 * cfg.width, -0.5]

    meshes = []
    num_boxes_x = int(cfg.length / cfg.box_size)
    border = cfg.length - num_boxes_x * cfg.box_size
    border_meshes = border_mesh(cfg.length, cfg.width, cfg.length - border, cfg.width - border, 1, pos)

    meshes += border_meshes

    box_dim = [cfg.box_size, cfg.box_size, 1.0]
    pose = np.eye(4)
    pose[:3, -1] = [cfg.box_size * 0.5, cfg.box_size * 0.5, -0.5]
    box = trimesh.creation.box(box_dim, pose)
    vertices = box.vertices
    faces = box.faces

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch_vertices = torch.tensor(vertices, device=device).repeat(num_boxes_x * num_boxes_x, 1, 1)
    xx, yy = torch.meshgrid(torch.arange(0, num_boxes_x, device=device), torch.arange(0, num_boxes_x, device=device))
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    offsets = cfg.box_size * xx_yy + border / 2
    torch_vertices[:, :, :2] += offsets.unsqueeze(1)

    if cfg.holes:
        mask_x = torch.logical_and(
            (torch_vertices[:, :, 0] > (cfg.length - border - cfg.platform_size) / 2).all(dim=1),
            (torch_vertices[:, :, 0] < (cfg.length + border + cfg.platform_size) / 2).all(dim=1),
        )
        torch_vertices_x = torch_vertices[mask_x]

        mask_y = torch.logical_and(
            (torch_vertices[:, :, 1] > (cfg.width - border - cfg.platform_size) / 2).all(dim=1),
            (torch_vertices[:, :, 1] < (cfg.width + border + cfg.platform_size) / 2).all(dim=1),
        )
        torch_vertices_y = torch_vertices[mask_y]

        torch_vertices = torch.cat((torch_vertices_x, torch_vertices_y))

    num_boxes = len(torch_vertices)
    noise = torch.zeros((num_boxes, 3), device=device)
    noise[:, 2].uniform_(-box_height, box_height)
    noise2 = torch.zeros((num_boxes, 4, 3), device=device)
    noise2 += noise.unsqueeze(1)
    noise2 = noise2.view(-1, 3)
    torch_vertices[torch_vertices[:, :, 2] == 0] += noise2
    vertices = torch_vertices.reshape(-1, 3).cpu().numpy()

    torch_faces = torch.tensor(faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    torch_faces += face_offsets.unsqueeze(2)
    faces = torch_faces.view(-1, 3).cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes.append(mesh)

    dim = [cfg.platform_size, cfg.platform_size, 1.0 + box_height]
    pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.width, -0.5 + box_height / 2]
    platform = trimesh.creation.box(dim, pose)
    meshes.append(platform)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], box_height])
    return meshes, origin


def pit(difficulty, cfg: "SubTerrainsCfg.PitCfg"):
    height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    inner_width = cfg.platform_size
    inner_length = cfg.platform_size
    pose = np.eye(4)

    if cfg.is_double_pit:
        height *= 2.0
        inner_width = cfg.platform_size + (cfg.width - cfg.platform_size) * 0.6
        inner_length = cfg.platform_size + (cfg.length - cfg.platform_size) * 0.6

    # Outer ring
    pos = [0.5 * cfg.length, 0.5 * cfg.width, -height * 0.5]
    meshes = border_mesh(cfg.length, cfg.width, inner_length, inner_width, height, pos)

    # Inner ring
    if cfg.is_double_pit:
        pos[2] = -height
        meshes += border_mesh(inner_length, inner_width, cfg.platform_size, cfg.platform_size, height, pos)

    # Ground
    box_dim = [cfg.length, cfg.width, 1.0]
    pose[:3, -1] = [pos[0], pos[1], -height - 0.5]
    ground = trimesh.creation.box(box_dim, pose)
    meshes.append(ground)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], -height])
    return meshes, origin


def box(difficulty, cfg: "SubTerrainsCfg.BoxCfg"):
    height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    pose = np.eye(4)

    if cfg.is_double_box:
        height *= 2.0

    # Upper box
    dim = [cfg.platform_size, cfg.platform_size, 1.0 + height]
    pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.width, -0.5 + height / 2]
    meshes = trimesh.creation.box(dim, pose)

    # Lower box
    if cfg.is_double_box:
        outer_width = cfg.platform_size + (cfg.width - cfg.platform_size) * 0.6
        outer_length = cfg.platform_size + (cfg.length - cfg.platform_size) * 0.6
        dim = [outer_width, outer_length, 1.0 + height / 2]
        pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.width, -0.5 + height / 4]
        meshes += trimesh.creation.box(dim, pose)

    # Ground
    pos = [0.5 * cfg.length, 0.5 * cfg.width, height]
    box_dim = [cfg.length, cfg.width, 1.0]
    pose[:3, -1] = [pos[0], pos[1], -0.5]
    meshes += trimesh.creation.box(box_dim, pose)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], height])
    return meshes, origin


def gap(difficulty, cfg: "SubTerrainsCfg.GapCfg"):
    gap_size = cfg.min_gap + (cfg.max_gap - cfg.min_gap) * difficulty

    height = 1.0
    pos = [0.5 * cfg.length, 0.5 * cfg.width, -height / 2]
    meshes = []

    border_meshes = border_mesh(
        cfg.length, cfg.width, cfg.platform_size + 2 * gap_size, cfg.platform_size + 2 * gap_size, height, pos
    )
    meshes += border_meshes

    box_dim = [cfg.platform_size, cfg.platform_size, height]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1], -height / 2]
    box = trimesh.creation.box(box_dim, pose)
    meshes.append(box)

    meshes = trimesh.util.concatenate(meshes)
    origin = np.array([pos[0], pos[1], 0.0])
    return meshes, origin


def table(difficulty, cfg: "SubTerrainsCfg.TableCfg"):
    table_height = cfg.max_table_height - (cfg.max_table_height - cfg.min_table_height) * difficulty
    table_length = cfg.min_table_length + (cfg.max_table_length - cfg.min_table_length) * difficulty
    meshes = []

    pos = [0.5 * cfg.length, 0.5 * cfg.width, table_height]
    table_meshes = border_mesh(
        cfg.platform_size + 2 * table_length,
        cfg.platform_size + 2 * table_length,
        cfg.platform_size,
        cfg.platform_size,
        cfg.table_thickness,
        pos,
    )
    meshes += table_meshes

    dim = [cfg.length, cfg.width, 1]
    pose = np.eye(4)
    pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.length, -0.5]
    platform = trimesh.creation.box(dim, pose)
    meshes.append(platform)

    meshes = trimesh.util.concatenate(meshes)
    origin = [pos[0], pos[1], 0.0]
    return meshes, origin


def rails(difficulty, cfg: "SubTerrainsCfg.RailsCfg"):
    height = cfg.max_height - (cfg.max_height - cfg.min_height) * difficulty
    meshes = []

    pos = [0.5 * cfg.length, 0.5 * cfg.width, height * 0.5]
    meshes += border_mesh(
        cfg.platform_size + 2.0 * cfg.min_thickness,
        cfg.platform_size + 2.0 * cfg.min_thickness,
        cfg.platform_size,
        cfg.platform_size,
        height,
        pos,
    )

    outer_width = cfg.platform_size + (cfg.width - cfg.platform_size) * 0.6
    outer_length = cfg.platform_size + (cfg.length - cfg.platform_size) * 0.6
    meshes += border_mesh(
        outer_length + 2.0 * cfg.max_thickness,
        outer_width + 2.0 * cfg.max_thickness,
        outer_length,
        outer_width,
        height,
        pos,
    )

    dim = [cfg.length, cfg.width, 1]
    pose = np.eye(4)
    pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.length, -0.5]
    platform = trimesh.creation.box(dim, pose)
    meshes.append(platform)

    meshes = trimesh.util.concatenate(meshes)
    origin = [pos[0], pos[1], 0.0]
    return meshes, origin


def plane(difficulty, cfg: "SubTerrainsCfg.PlaneCfg"):
    x0 = [cfg.length, cfg.width, 0.0]
    x1 = [cfg.length, 0.0, 0.0]
    x2 = [0.0, cfg.width, 0.0]
    x3 = [0.0, 0.0, 0.0]
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh, [0.5 * cfg.length, 0.5 * cfg.width, 0.0]


def repeated_objects(difficulty, cfg):
    """
    Helper function to create a terrain, with objects constructed through cfg.object_func.
    """

    def make_plane(length, width):
        # Create a plane with dimensions length and width
        x0 = [length, width, 0.0]
        x1 = [length, 0.0, 0.0]
        x2 = [0.0, width, 0.0]
        x3 = [0.0, 0.0, 0.0]
        vertices = np.array([x0, x1, x2, x3])
        faces = np.array([[1, 0, 2], [2, 3, 1]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    # Construct ground plane
    meshes = make_plane(cfg.length, cfg.width)

    # Process parameters
    d = 0.5 * cfg.platform_size  # dimension of platform
    h = cfg.cp0.height + (cfg.cp1.height - cfg.cp0.height) * difficulty  # obstacle height
    num = cfg.cp0.num + np.int((cfg.cp1.num - cfg.cp0.num) * difficulty)  # number of obstacles

    # Center of terrain
    origin = [0.5 * cfg.length, 0.5 * cfg.width, 0.5 * h]

    # Center of obstacles
    c = np.zeros((num, 3))
    c[:, 0] = np.random.uniform(0.0, cfg.length, num)
    c[:, 1] = np.random.uniform(0.0, cfg.width, num)

    # Construct obstacles (but keep platform clean)
    for id in range(num):
        is_plaform = (
            c[id, 0] < origin[0] + d
            and c[id, 0] > origin[0] - d
            and c[id, 1] < origin[1] + d
            and c[id, 1] > origin[1] - d
        )
        if not is_plaform:
            delta_h = np.random.uniform(-cfg.max_rand_height, cfg.max_rand_height)
            new_h = h + delta_h
            if new_h > 0.0:
                meshes += cfg.object_func(c[id, :], cfg, new_h, difficulty)

    # Construct platform
    dim = [cfg.platform_size, cfg.platform_size, 0.5 * h]
    pose = np.eye(4)
    pose[:3, -1] = [0.5 * cfg.length, 0.5 * cfg.width, h * 0.25]
    meshes += trimesh.creation.box(dim, pose)

    return meshes, origin


def make_pyramid(center, cfg: "SubTerrainsCfg.PyramidsCfg", h, difficulty):
    pose = np.eye(4)
    pose[0:3, -1] = center
    R = Rot.random().as_euler("zyx")
    max_rot_deg = cfg.cp0.rot + (cfg.cp1.rot - cfg.cp0.rot) * difficulty
    R[1:3] *= max_rot_deg / 180.0
    pose[0:3, 0:3] = Rot.from_euler("zyx", R).as_matrix()
    return trimesh.creation.cone(radius=cfg.radius, height=h, sections=np.random.randint(4, 6), transform=pose)


def make_box(center, cfg: "SubTerrainsCfg.RotatedBoxesCfg", h, difficulty):
    dim = [cfg.obj_length, cfg.obj_width, h]
    pose = np.eye(4)
    pose[0:3, -1] = center
    R = Rot.random().as_euler("zyx")
    max_rot_deg = cfg.cp0.rot + (cfg.cp1.rot - cfg.cp0.rot) * difficulty
    R[1:3] *= max_rot_deg / 180.0
    pose[0:3, 0:3] = Rot.from_euler("zyx", R).as_matrix()
    return trimesh.creation.box(dim, pose)


def make_stepping_stone(center, cfg: "SubTerrainsCfg.SteppingStonesCylindersCfg", h, difficulty):
    pose = np.eye(4)
    pose[0:3, -1] = center
    R = Rot.random().as_euler("zyx")
    max_rot_deg = cfg.cp0.rot + (cfg.cp1.rot - cfg.cp0.rot) * difficulty
    R[1:3] *= max_rot_deg / 180.0
    pose[0:3, 0:3] = Rot.from_euler("zyx", R).as_matrix()
    return trimesh.creation.cylinder(radius=cfg.radius, height=h, sections=np.random.randint(4, 6), transform=pose)
