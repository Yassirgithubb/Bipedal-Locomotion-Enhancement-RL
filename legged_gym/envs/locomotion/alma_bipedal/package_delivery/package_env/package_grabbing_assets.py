# python
from dataclasses import MISSING

# utils
from legged_gym.utils import configclass

# legged-gym
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.aow.aow_camera_cfg import AowCameraCfg
from legged_gym.common.assets.asset_cfg import CuboidAssetCfg
from legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import (
    anymal_c_actuator_cfg,
    wheel_actuator,
)
from legged_gym.common.actuators import (
    anymal_d_actuator_cfg,
    baboon_actuator,
    coyote_actuator,
)


# dof_names: 0=LF_HAA
#            1=LF_HFE
#            2=LF_KFE
#            3=LF_WHEEL
#            4=LH_HAA
#            5=LH_HFE
#            6=LH_KFE
#            7=LH_WHEEL
#            8=RF_HAA
#            9=RF_HFE
#            10=RF_KFE
#            11=RF_WHEEL
#            12=RH_HAA
#            13=RH_HFE
#            14=RH_KFE
#            15=RH_WHEEL

# body_names: 0=base
#             1=LF_HIP
#             2=LF_THIGH
#             3=LF_SHANK
#             4=LF_WHEEL_L
#             5=LH_HIP
#             6=LH_THIGH
#             7=LH_SHANK
#             8=LH_WHEEL_L
#             9=RF_HIP
#             10=RF_THIGH
#             11=RF_SHANK
#             12=RF_WHEEL_L
#             13=RH_HIP
#             14=RH_THIGH
#             15=RH_SHANK
#             16=RH_WHEEL_L


@configclass
class PackageCfg(CuboidAssetCfg):
    max_yaw: float = MISSING


@configclass
class TableCfg(CuboidAssetCfg):
    max_height: float = MISSING


package_cfg = PackageCfg(
    asset_name="Package",
    depth=0.45,  # x-axis
    width=0.45,  # y-axis
    height=0.45,  # z-xis
    density=80,
    init_state=CuboidAssetCfg.InitState(
        pos=(0.0, 0.0, 0.0),  # package spawns above table position
        rot=(0.5, 0.5, 0.5, 0.5),  # turn package to have height in z direction
    ),
    randomization=CuboidAssetCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(2, 2.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-0.5, 0.5),
    ),
    max_yaw=3.14 / 2,
)

table_cfg = TableCfg(
    asset_name="Table",
    depth=0.2,  # x-axis
    width=0.2,  # y-axis
    height=0.1,  # z-xis
    density=100,
    disable_gravity=True,
    fix_base_link=True,
    init_state=CuboidAssetCfg.InitState(
        pos=(1.2, 0.0, 0.4),
        rot=(0.5, 0.5, 0.5, 0.5),  # turn package to have height in z direction
    ),
    randomization=CuboidAssetCfg.Randomization(
        randomize_added_mass=False,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
    ),
    max_height=0.7,
)

""" standing_robot_cfg = AowCameraCfg(
    cls_name="AowCamera",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_wheels/urdf/anymal_wheels_chimera.urdf",
    feet_names=".*WHEEL",
    self_collisions=True,
    replace_cylinder_with_capsule=False,
    feet_position_offset=[0.0, 0.0, 0.14],
    # standing state
    init_state=AowCameraCfg.InitState(
        pos=(0.0, 0.0, 0.95),
        rot=(0.0, -0.707, 0.0, 0.707),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.3,  # both front HFE
            ".*H_HFE": 0.6,  # both hind HFE
            ".*F_KFE": -0.6,
            ".*H_KFE": 0.6,
        },
    ),
    actuators=[
        {"actuator": anymal_c_actuator_cfg, "dof_names": [".*HAA", ".*HFE", ".*KFE"]},
        {
            "actuator": wheel_actuator,
            "dof_names": [".*WHEEL"],
            "p_gains": {".*": 0.0},
            "d_gains": {".*": 8.0},
        },
    ],
    randomization=AowCameraCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5.0),
    ),
) """

standing_robot_cfg = AowCameraCfg(
    cls_name="AowCamera",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/alma/urdf/alma_d_bipedal.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    replace_cylinder_with_capsule=False,
    feet_position_offset=[0.0, 0.0, 0.4],
    # standing state
    init_state=AowCameraCfg.InitState(
        pos=(0.0, 0.0, 0.6),
        rot=(0.0, -0.707, 0.0, 0.707),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0,  # both front HFE
            ".*H_HFE": 0.15,  # both hind HFE
            ".*F_KFE": -0.7,
            ".*H_KFE": 1.6,
            "SH_ROT": 3.1416,
            "SH_FLE": 0.9,
            "EL_FLE": 0.8,
            "FA_ROT": 0.0,
            "WRIST_1": 0.0,
            "WRIST_2": 0.0,
        },
    ),
     actuators=[
        {"actuator": anymal_d_actuator_cfg, "dof_names": [".*HAA", ".*HFE", ".*KFE"]},
        {
            "actuator": baboon_actuator,
            "dof_names": ["SH_ROT", "SH_FLE", "EL_FLE"],
            "p_gains": {".*": 50.0},
            "d_gains": {".*": 4.0},
        },
        {
            "actuator": coyote_actuator,
            "dof_names": ["FA_ROT", "WRIST_1", "WRIST_2"],
            "p_gains": {".*": 50.0},
            "d_gains": {".*": 4.0},
        },
    ],
    randomization=AowCameraCfg.Randomization(
        randomize_added_mass=False,
        randomize_friction=False,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5.0),
    ),
)