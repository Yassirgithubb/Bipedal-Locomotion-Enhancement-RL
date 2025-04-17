from legged_gym.utils.config_utils import configclass
from legged_gym.common.assets.robots.articulation_cfg import ArticulationCfg
from legged_gym.common.actuators import (
    baboon_actuator,
    coyote_actuator,
)


@configclass
class ManipulatorCfg(ArticulationCfg):
    cls_name = "Manipulator"
    ee_names = ".*END_EFFECTOR"  # name of the feet rigid bodies (from URDF), used to index body state and contact force tensors
    penalize_contacts_on = {}  # list of rigid body names (from URDF)
    terminate_after_contacts_on = []  # list of rigid body names (from URDF)
    fix_base_link = True
    arm_drive_configuration = "serial"
    arm_drive_start_idx = 0


# Ready to use robots

dynaarm_robot_cfg = ManipulatorCfg(
    asset_name="dynaarm",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/dynaarm/urdf/dynaarm.urdf",
    self_collisions=True,
    init_state=ManipulatorCfg.InitState(
        pos=(0.0, 0.0, 0.01),
        dof_pos={
            "SH_ROT": 0.0,
            "SH_FLE": -0.7,
            "EL_FLE": 1.4,
            "FA_ROT": 0.0,
            "WRIST_1": 0.0,
            "WRIST_2": 0.0,
        },
    ),
    actuators=[
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
    randomization=ManipulatorCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),
        added_mass_range=(-1.0, 1),
    ),
)
