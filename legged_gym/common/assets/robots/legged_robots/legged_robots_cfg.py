from legged_gym.utils.config_utils import configclass
from legged_gym.common.assets.robots.articulation_cfg import ArticulationCfg
from legged_gym.common.actuators import (
    anymal_c_actuator_cfg,
    anymal_d_actuator_cfg,
    barry_hip_actuator,
    barry_knee_actuator,
    wheel_actuator,
)


@configclass
class LeggedRobotCfg(ArticulationCfg):
    cls_name = "LeggedRobot"
    feet_names = (
        ".*foot"  # name of the feet rigid bodies (from URDF), used to index body state and contact force tensors
    )
    feet_position_offset = [0.0, 0.0, 0.0]


# Ready to use robots

anymal_b_robot_cfg = LeggedRobotCfg(
    asset_name="anymal_b",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_b/urdf/anymal_b.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    replace_cylinder_with_capsule=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.6),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators=[{"actuator": anymal_c_actuator_cfg, "dof_names": [".*"]}],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5.0),
    ),
)

anymal_c_robot_cfg = LeggedRobotCfg(
    asset_name="anymal_c",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    replace_cylinder_with_capsule=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.6),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators=[{"actuator": anymal_c_actuator_cfg, "dof_names": [".*"]}],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5),
    ),
)

anymal_d_robot_cfg = LeggedRobotCfg(
    asset_name="anymal_d",
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_d/urdf/anymal_d.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.6),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
        },
    ),
    actuators=[{"actuator": anymal_d_actuator_cfg, "dof_names": [".*"]}],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5),
    ),
)

# barry
barry = LeggedRobotCfg(
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_barry.urdf",
    feet_names=".*FOOT",
    self_collisions=True,
    soft_dof_limit_factor=0.95,
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.8),
        dof_pos={
            "L._HAA": 0.1,  # all HAA
            "R._HAA": -0.1,  # both front HFE
            ".*HFE": 0.7,  # both hind HFE
            ".*_KFE": -1.5,
        },
    ),
    actuators=[
        {
            "actuator": barry_hip_actuator,
            "dof_names": [".*HAA", ".*HFE"],
            "p_gains": {".*": 140.0},
            "d_gains": {".*": 3.5},
        },
        {"actuator": barry_knee_actuator, "dof_names": [".*KFE"], "p_gains": {".*": 140.0}, "d_gains": {".*": 3.5}},
    ],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_rigid_body_indices=(0,),
        added_mass_range=(-5.0, 25.0),
    ),
)

# anymal wheels
anymal_wheels_robot_cfg = LeggedRobotCfg(
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_wheels/urdf/anymal_wheels_chimera.urdf",
    feet_names=".*WHEEL",
    self_collisions=True,
    replace_cylinder_with_capsule=False,
    feet_position_offset=[0.0, 0.0, 0.14],
    init_state=LeggedRobotCfg.InitState(
        pos=(0.0, 0.0, 0.7),
        dof_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.3,  # both front HFE
            ".*H_HFE": -0.3,  # both hind HFE
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
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=True,
        randomize_friction=True,
        friction_range=(0.0, 1.5),  # friction coefficients are averaged, mu = 0.5*(mu_terrain + mu_foot)
        added_mass_range=(-5.0, 5.0),
    ),
)

# cassie
cassie_robot_cfg = LeggedRobotCfg(
    file="{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf",
    feet_names=".*toe",
    soft_dof_limit_factor=0.95,
    self_collisions=True,
    flip_visual_attachments=False,
    replace_cylinder_with_capsule=True,
    init_state=LeggedRobotCfg.InitState(
        pos = (0.0, 0.0, 1.0),
        dof_pos = { 
            "hip_abduction_left": 0.1,
            "hip_rotation_left": 0.0,
            "hip_flexion_left": 1.0,
            "thigh_joint_left": -1.8,
            "ankle_joint_left": 1.57,
            "toe_joint_left": -1.57,
            "hip_abduction_right": -0.1,
            "hip_rotation_right": 0.0,
            "hip_flexion_right": 1.0,
            "thigh_joint_right": -1.8,
            "ankle_joint_right": 1.57,
            "toe_joint_right": -1.57,
        },
    ),
    actuators=[{
            "actuator": anymal_d_actuator_cfg,
            "dof_names": [".*"],
            "p_gains": {".*": 140.0},
            "d_gains": {".*": 3.5},
        },],
    randomization=LeggedRobotCfg.Randomization(
        randomize_added_mass=False,
        randomize_friction=False,
        # !!!! in Preview 3 friction Combine mode=MULTIPLY, in Preview 4 it will be changed to AVERAGE !!!!!!!!!!
        friction_range=(-0.98, 3.0), # [0.01, 2.0] in old multiply mode (static friction = 1.0)
        added_mass_range=(-0.8, 0.8),
    ),
)