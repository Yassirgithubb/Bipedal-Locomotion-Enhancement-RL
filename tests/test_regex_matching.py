#
import re

# ground truth!
ROBOT_DOF_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
ROBOT_DOF_INDICES = dict()
for index, name in enumerate(ROBOT_DOF_NAMES):
    ROBOT_DOF_INDICES[name] = index


def test_regex_parsing(actuator_dof_names):
    # find actuator DOF names through regex string matching
    old_dof_names = list()
    for re_key in actuator_dof_names:
        old_dof_names.extend([name for name in ROBOT_DOF_NAMES if re.match(re_key, name)])
    # find corresponding actuator DOF indices
    old_dof_indices = list()
    for name in old_dof_names:
        dof_index = ROBOT_DOF_INDICES[name]
        old_dof_indices.append(dof_index)

    old_dof_indices.sort()
    print("Existing solution: ")
    print(f"dof_names: {old_dof_names}")
    print(f"dof_indices: {old_dof_indices}")
    print()

    # test
    new_dof_names = list()
    new_dof_indices = list()

    for name in ROBOT_DOF_NAMES:
        for re_key in actuator_dof_names:
            if re.match(re_key, name):
                new_dof_names.append(name)
                new_dof_indices.append(ROBOT_DOF_INDICES[name])

    print("New solution: ")
    print(f"dof_names: {new_dof_names}")
    print(f"dof_indices: {new_dof_indices}")

    # assert old_dof_names == new_dof_names
    # assert old_dof_indices == new_dof_indices


if __name__ == "__main__":

    actuator_dof_names = [".*"]
    print("-" * 40)
    print(f"\nTest Case 1: {actuator_dof_names}\n")
    test_regex_parsing(actuator_dof_names)

    actuator_dof_names = [".*hip.*", ".*thigh.*"]
    print("-" * 40)
    print(f"\nTest Case 2: {actuator_dof_names}\n")
    test_regex_parsing(actuator_dof_names)

    actuator_dof_names = ["FL.*", "FR.*"]
    print("-" * 40)
    print(f"\nTest Case 3: {actuator_dof_names}\n")
    test_regex_parsing(actuator_dof_names)

    actuator_dof_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
    ]
    print("-" * 40)
    print(f"\nTest Case 4: {actuator_dof_names}\n")
    test_regex_parsing(actuator_dof_names)

    actuator_dof_names = [""]
    print("-" * 40)
    print(f"\nTest Case 5: {actuator_dof_names}\n")
    test_regex_parsing(actuator_dof_names)
