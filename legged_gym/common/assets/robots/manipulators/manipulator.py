# isaac-gym
from isaacgym.torch_utils import get_axis_params, quat_rotate_inverse, to_torch, quat_apply

# python
import torch
from torch import Tensor

# legged-gym
from legged_gym.common.gym_interface import GymInterface
from legged_gym.common.assets.robots.articulation import Articulation
from .manipulators_cfg import ManipulatorCfg


class Manipulator(Articulation):

    projected_gravity_b: Tensor = None
    """ Projection of the Gravity vector in base frame (Tensor), shape=(num_envs, 3)"""

    ee_indices: torch.Tensor
    """ Indices of the feet rigid bodies"""

    def __init__(self, cfg: ManipulatorCfg, num_envs: int, gym_iface: GymInterface) -> None:
        super().__init__(cfg, num_envs, gym_iface)
        # note: we reassign cfg here for PyLance to recognize the class object
        self.cfg = cfg

    def init_buffers(self):
        super().init_buffers()

        # process body names and indices TODO change to list for better performance
        self.ee_indices, _ = self.find_bodies(self.cfg.ee_names)

        up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self._gravity_vec_w = to_torch(get_axis_params(-1.0, up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.root_ang_vel_b = quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)
        self.projected_gravity_b = quat_rotate_inverse(self.root_quat_w, self._gravity_vec_w)
        # base is fixed and aligned with world axis
        print(self.ee_indices[0])
        self.ee_pos_b = quat_rotate_inverse(
            self.root_quat_w, self.rigid_body_states[:, self.ee_indices[0], :3] - self.root_states[:, :3]
        )
        self.ee_pos_w = self.rigid_body_states[:, self.ee_indices[0], :3]
        self.ee_vel_w = self.rigid_body_states[:, self.ee_indices[0], 7:10]
        self.ee_facing_w = quat_apply(self.rigid_body_states[:, self.ee_indices[0], 3:7], self._gravity_vec_w)
        if self.cfg.arm_drive_configuration == "dynaarm":
            self.serial_dof_vel = torch.zeros_like(self.dof_vel)

    def update_buffers(self, dt: float, env_ids=None):
        super().update_buffers(dt, env_ids)
        if env_ids is None:
            env_ids = ...  # all elements of the tensor
        self.root_ang_vel_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self.root_ang_vel_w[env_ids])
        self.projected_gravity_b[env_ids] = quat_rotate_inverse(self.root_quat_w[env_ids], self._gravity_vec_w[env_ids])
        self.ee_pos_b[env_ids] = quat_rotate_inverse(
            self.root_quat_w[env_ids],
            self.rigid_body_states[env_ids, self.ee_indices[0], :3] - self.root_states[env_ids, :3],
        )
        self.ee_pos_w[env_ids] = self.rigid_body_states[env_ids, self.ee_indices[0], :3]
        self.ee_vel_w[env_ids] = self.rigid_body_states[env_ids, self.ee_indices[0], 7:10]
        self.ee_facing_w[env_ids] = quat_apply(
            self.rigid_body_states[env_ids, self.ee_indices[0], 3:7], self._gravity_vec_w[env_ids]
        )

    def apply_actions(self, actions: torch.Tensor):
        if self.cfg.arm_drive_configuration == "dynaarm":
            self.serial_dof_vel[:] = self.dof_vel[:]
            self.dof_vel[:, self.cfg.arm_drive_start_idx + 2] += self.serial_dof_vel[
                :, self.cfg.arm_drive_start_idx + 1
            ]
            super().apply_actions(actions)
            self.dof_vel[:] = self.serial_dof_vel[:]
            self.dof_torques[:, self.cfg.arm_drive_start_idx + 1] += self.dof_torques[
                :, self.cfg.arm_drive_start_idx + 2
            ]
        else:
            super().apply_actions(actions)
