from legged_gym.utils.config_utils import configclass
from legged_gym.common.assets.robots.manipulators.manipulators_cfg import ManipulatorCfg
from legged_gym.common.assets.robots.legged_robots.legged_robots_cfg import LeggedRobotCfg
from legged_gym.common.actuators import (
    anymal_d_actuator_cfg,
    baboon_actuator,
    coyote_actuator,
)


@configclass
class LeggedMobileManipulatorCfg(ManipulatorCfg, LeggedRobotCfg):
    cls_name = "LeggedMobileManipulator"
    ee_names = ".*wheel"  # name of the feet rigid bodies (from URDF), used to index body state and contact force tensors
    penalize_contacts_on = {}  # list of rigid body names (from URDF)
    terminate_after_contacts_on = []  # list of rigid body names (from URDF)
    fix_base_link = False
    arm_drive_configuration = "serial"
    arm_drive_start_idx = 12


# Ready to use robots

alma_robot_cfg = LeggedMobileManipulatorCfg(
    asset_name="alma",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/alma/urdf/alma_d.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.65),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
            "SH_ROT": 0.0,
            "SH_FLE": -0.7,
            "EL_FLE": 1.4,
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
    arm_drive_configuration="dynaarm",
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),
        added_mass_range=(-1.0, 1),
    ),
)

alma_bipedal_robot_cfg = LeggedMobileManipulatorCfg(
    asset_name="alma",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/alma/urdf/alma_d_bipedal.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.65),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.7,  # both front HFE
            ".*H_HFE": 0.15,  # both hind HFE
            ".*F_KFE": 1.74,
            ".*H_KFE": 1.6,
            "SH_ROT": 0,
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
    arm_drive_configuration="dynaarm",
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=False,
        randomize_friction=False,
        friction_range=(0.0, 1.5),
        added_mass_range=(-1.0, 1),
    ),
)
