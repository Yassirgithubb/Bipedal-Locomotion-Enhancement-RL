from typing import Tuple
from legged_gym.utils import configclass


@configclass
class UnifromVelocityCommandCfg:
    class_name: str = "UnifromVelocityCommand"
    robot_name: str = "robot"
    curriculum = False
    max_curriculum = 1.0
    num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
    resampling_time = 10.0  # time before commands are changed [s]
    heading_command = True  # if true: compute ang vel command from heading error
    rel_standing_envs = 0.02  # percentage of the robots are standing
    rel_heading_envs = 1.0  # percentage of the robots follow heading command (the others follow angular velocity)

    @configclass
    class Ranges:
        lin_vel_x: Tuple = (-1.0, 1.0)  # min max [m/s]
        lin_vel_y: Tuple = (-1.0, 1.0)  # min max [m/s]
        ang_vel_yaw: Tuple = (-1.5, 1.5)  # min max [rad/s]
        heading: Tuple = (-3.14, 3.14)  # [rad]

    ranges = Ranges()


@configclass
class NormalVelocityCommandCfg(UnifromVelocityCommandCfg):
    class_name: str = "NormalVelocityCommand"
    heading_command = False

    @configclass
    class Ranges:
        mean_vel: Tuple = (0.8, 0.8, 0.8)  # linear x, linear y, angular yaw [m/s]
        std_vel: Tuple = (0.4, 0.4, 0.4)  # linear x, linear y, angular yaw [m/s]
        zero_prob: Tuple = (0.01, 0.01, 0.01)  # linear x, linear y, angular yaw [percentage of zero velocity]

    ranges = Ranges()
