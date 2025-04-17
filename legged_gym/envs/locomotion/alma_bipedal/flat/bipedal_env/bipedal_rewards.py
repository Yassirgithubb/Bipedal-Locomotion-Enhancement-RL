
import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv

from legged_gym.envs.rewards import *


class TASK_VARIABLE:
    static_var = 0
 
    def __init__(self):
        TASK_VARIABLE.static_var += 1
        self.instance_var = TASK_VARIABLE.static_var

def link_linear_velocity(env: "ANY_ENV", params):
     # Penalize link velocities
    
     return torch.sum(torch.square( env.robot.rigid_body_states[:, params["body_indices"], [7,8,9]]), dim=1)

def tracking_lin_vel_standing(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.square(env.robot.root_lin_vel_b[:, 0] )  + torch.square(env.robot.root_lin_vel_b[:, 1] )
    task_var =  TASK_VARIABLE()
    
    if  ( task_var.static_var  >  6000):
        return torch.exp(-lin_vel_error / params["std"])
    else:
        return 0


def dist_from_goal(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    goal_x  =env.terrain.env_origins[:, [0]]  -  10
    goal_y = env.terrain.env_origins[:, [1]]  - 6

    dist_error = torch.square( goal_x - env.robot.rigid_body_states[:, [0], [0]])
    task_var =  TASK_VARIABLE()
    if task_var.static_var  >  10000:
        return torch.exp(-torch.squeeze(dist_error/30,1) )
    else:
        return 0
    


def stand_still_standing(env: "LeggedEnv", params):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * (
        torch.norm(env.command_generator.get_command()[:, :2], dim=1) < 0.1
    )


def tracking_ang_vel_standing(env: "LeggedEnv", params):
    # Tracking of angular velocity commands (yaw) is the roll angle of the base 
    # "std" defines the width of the bel curve
    #ang_vel_error = torch.square(env.command_generator.get_command()[:, 0] - env.robot.root_ang_vel_b[:, 0])
    z_ang_tensor = torch.reshape( env.robot.rigid_body_states[:, [0], 12],(-1,))
    #z_ang_tensor = env.robot.root_ang_vel_b[:, 0]
    ang_vel_error = torch.square((0*env.command_generator.get_command()[:, 0]+0.5) -  z_ang_tensor)
    task_var =  TASK_VARIABLE()
    #print(task_var.static_var)
    if  ( task_var.static_var  >  3000):
        return torch.exp(-ang_vel_error / params["std"])
    else:
        return 0
    
    