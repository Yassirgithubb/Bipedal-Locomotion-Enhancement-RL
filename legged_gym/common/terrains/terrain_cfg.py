from legged_gym.utils.config_utils import configclass
from . import mesh_terrains as mt
from . import heightfield_terrains as ht


@configclass
class SubTerrainsCfg:
    @configclass
    class BaseCfg:
        func = ""
        proportion = 0.0
        length = 0.0  # set from the Terrain config
        width = 0.0  # set from the Terrain config
        platform_size = 3.0
        border_size = 0.25
        holes = False
        # heightfield only
        slope_treshold = 0.75
        horizontal_scale = 0.0  # set from the Terrain config
        vertical_scale = 0.0  # set from the Terrain config

    @configclass
    class PlaneCfg(BaseCfg):
        func = mt.plane

    @configclass
    class PyramidsCfg(BaseCfg):
        func = mt.repeated_objects
        object_func = mt.make_pyramid

        @configclass
        class CurriculumParams:
            height = 0.05  # height of cone pyramid at curriculum start/end [m]
            num = 100  # number of objects at curriculum start/end [-]
            rot = 0.0  # max roll/pitch at curriculum start/end [deg]

        max_rand_height = 0.0  # value sampled from uniform distribution, added to height [m]
        radius = 0.6  # radius of the cone pyramide at curriculum start/end [m]
        cp0 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum start
        cp1 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum end

    @configclass
    class RotatedBoxesCfg(BaseCfg):
        func = mt.repeated_objects
        object_func = mt.make_box

        @configclass
        class CurriculumParams:
            height = 0.05  # height of cone pyramid at curriculum start/end [m]
            num = 100  # number of objects at curriculum start/end [-]
            rot = 0.0  # max roll/pitch at curriculum start/end [deg]

        max_rand_height = 0.0  # value sampled from uniform distribution, added to height [m]
        obj_length = 0.6  # box length [m]
        obj_width = 0.6  # box width [m]
        cp0 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum start
        cp1 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum end

    @configclass
    class SteppingStonesCylindersCfg(BaseCfg):
        func = mt.repeated_objects
        object_func = mt.make_stepping_stone

        @configclass
        class CurriculumParams:
            height = 0.05  # height of cone pyramid at curriculum start/end [m]
            num = 100  # number of objects at curriculum start/end [-]
            rot = 0.0  # max roll/pitch at curriculum start/end [deg]

        max_rand_height = 0.0  # value sampled from uniform distribution, added to height [m]
        radius = 0.6  # radius of stepping stones [m]
        cp0 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum start
        cp1 = CurriculumParams(height=0.05, num=100, rot=0.0)  # parameters at curriculum end

    @configclass
    class PyramidStairsCfg(BaseCfg):
        func = mt.pyramid_stairs
        min_step_height = 0.05
        max_step_height = 0.23
        step_width = 0.3

    @configclass
    class PyramidStairsInvCfg(BaseCfg):
        func = mt.pyramid_stairs_inv
        min_step_height = 0.05
        max_step_height = 0.23
        step_width = 0.3

    @configclass
    class BoxesCfg(BaseCfg):
        func = mt.boxes
        box_size = 0.8
        min_box_height = 0.025
        max_box_height = 0.2

    @configclass
    class PitCfg(BaseCfg):
        func = mt.pit
        min_height = 0.05
        max_height = 1.1
        is_double_pit = False

    @configclass
    class BoxCfg(BaseCfg):
        func = mt.box
        min_height = 0.05
        max_height = 1.1
        is_double_box = False

    @configclass
    class GapCfg(BaseCfg):
        func = mt.gap
        min_gap = 0.05
        max_gap = 1.1

    @configclass
    class TableCfg(BaseCfg):
        # TODO check values
        func = mt.table
        min_table_height = 0.4
        max_table_height = 1.0
        min_table_length = 0.5
        max_table_length = 1.0
        table_thickness = 0.05

    @configclass
    class RailsCfg(BaseCfg):
        func = mt.rails
        min_height = 0.3
        max_height = 0.05
        min_thickness = 0.05
        max_thickness = 0.1

    @configclass
    class HfPyramidStairsCfg(BaseCfg):
        func = ht.hf_pyramid_stairs
        min_step_height = 0.05
        max_step_height = 0.23
        step_width = 0.301  # 0.3 would be rounded down to 0.2

    @configclass
    class HfPyramidStairsInvCfg(BaseCfg):
        func = ht.hf_pyramid_stairs_inv
        min_step_height = 0.05
        max_step_height = 0.23
        step_width = 0.301  # 0.3 would be rounded down to 0.2

    @configclass
    class HfPyramidSlopeCfg(BaseCfg):
        func = ht.hf_pyramid_slope
        max_slope = 0.4
        min_slope = 0.0
        min_height_noise = -0.05
        max_height_noise = 0.05
        noise_step = 0.005
        downsampled_scale = 0.2

    @configclass
    class HfPyramidSlopeInvCfg(BaseCfg):
        func = ht.hf_pyramid_slope_inv
        max_slope = 0.4
        min_slope = 0.0
        min_height_noise = -0.05
        max_height_noise = 0.05
        noise_step = 0.005
        downsampled_scale = 0.2

    @configclass
    class HfDiscreteObstaclesCfg(BaseCfg):
        func = ht.hf_discrete_obstacles
        num_rectangles = 20
        rectangle_min_size = 1.0
        rectangle_max_size = 2.0
        rectangle_min_height = 0.05
        rectangle_max_height = 0.25

    @configclass
    class HfSteppingStonesCfg(BaseCfg):
        func = ht.hf_stepping_stones
        stones_min_size = 1.575
        stones_max_size = 0.25
        stones_min_distance = 0.05
        stones_max_distance = 0.1
        max_height = 0.0

    pyramid_stairs = PyramidStairsCfg()
    pyramid_stairs_inv = PyramidStairsInvCfg()
    boxes = BoxesCfg()
    hf_pyramid_stairs = HfPyramidStairsCfg()
    hf_pyramid_stairs_inv = HfPyramidStairsInvCfg()
    hf_pyramid_slope = HfPyramidSlopeCfg()
    hf_pyramid_slope_inv = HfPyramidSlopeInvCfg()
    hf_discrete_obstacles = HfDiscreteObstaclesCfg()
    hf_stepping_stones = HfSteppingStonesCfg()


@configclass
class TerrainCfg:
    mesh_type: str = "trimesh"  # none, plane, heightfield or trimesh
    env_spacing = 3.0  # not used with heightfields/trimeshes
    border_size = 25  # [m]
    curriculum = True
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.0
    max_init_terrain_level = 5  # starting curriculum state
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 10  # number of terrain rows (levels)
    num_cols = 20  # number of terrain cols (types)
    # heightfield specific
    horizontal_scale = 0.1  # [m]
    vertical_scale = 0.005  # [m]
    slope_threshold = 0.75

    # slopes above this threshold will be corrected to vertical surfaces

    sub_terrains = SubTerrainsCfg()
