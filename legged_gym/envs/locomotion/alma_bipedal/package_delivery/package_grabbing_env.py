# isaac-gym
from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_euler_xyz, quat_rotate_inverse, quat_mul, quat_rotate

# python
import torch
import numpy as np

# legged-gym
from legged_gym.envs.locomotion.anymal_wheels.wheeled_legged_env import WheeledLeggedEnv
from legged_gym.common.assets.robots.legged_robots.legged_robot import LeggedRobot
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.aow.aow_camera import AowCamera
from legged_gym.common.assets.asset import CuboidAsset
from legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_grabbing_env_config import (
    PackageGrabbingEnvCfg,
)
import legged_gym.envs.locomotion.alma_bipedal.package_delivery.package_env.package_grabbing_viz as viz
from legged_gym.common.terrains import Terrain, TerrainGenerator

from legged_gym.common.observations.observation_manager import ObsManager
from legged_gym.envs.locomotion.legged_env_config import LeggedEnvCfg
from legged_gym.common.rewards.reward_manager import RewardManager
from legged_gym.common.curriculum.curriculum_manager import CurriculumManager
from legged_gym.common.terminations.termination_manager import TerminationManager


class PackageGrabbingEnv(WheeledLeggedEnv):

    cfg: PackageGrabbingEnvCfg

    def _init_external_forces(self):

        self.external_forces = torch.zeros(
            (self.num_envs, self.robot.num_bodies + self.package.num_bodies , 3),
            device=self.device,
        )
        self.external_torques = torch.zeros(
            (self.num_envs, self.robot.num_bodies + self.package.num_bodies , 3),
            device=self.device,
        ) 

    def reset_idx(self, env_ids):
        # -- reset package and table state
        self._reset_package_and_table(env_ids)
        # -- write to simulator
        self.gym_iface.write_states_to_sim()
        # -- reset package buffers
        self.package.reset_buffers(env_ids)
        #self.table.reset_buffers(env_ids)
        super().reset_idx(env_ids)

    def _create_envs(self):
        """Design the environment instances."""
        # add terrain instance
        terrain_curriculum = self.cfg.curriculum.__dict__.get("terrain_levels", None) is not None
        terrain_generator = TerrainGenerator(self.cfg.terrain, curriculum=terrain_curriculum)
        self.terrain = Terrain(self.cfg.terrain, self.num_envs, self.gym_iface)
        self.terrain.set_terrain_origins(terrain_generator.terrain_origins)
        self.terrain.add_mesh(terrain_generator.terrain_mesh, name="terrain")
        self.terrain.add_to_sim()
        # add robot class
        robot_cls = eval(self.cfg.robot.cls_name)
        self.robot: AowCamera = robot_cls(self.cfg.robot, self.num_envs, self.gym_iface)
        # add package class
        package_cls = eval(self.cfg.package.cls_name)
        self.package: CuboidAsset = package_cls(self.cfg.package, self.num_envs, self.gym_iface)
        # add package class
        #table_cls = eval(self.cfg.table.cls_name)
        #self.table: CuboidAsset = table_cls(self.cfg.table, self.num_envs, self.gym_iface)
        # create environments
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = list()
        for i in range(self.num_envs):
            # create env instance
            env_package = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_package)
            # spawn
            pos = self.terrain.env_origins[i].clone()
            self.robot.spawn(i, pos)
            self.package.spawn(i, pos)


    def _post_physics_step(self):
        # compute if package is in fov of camera
        self.check_package_in_fov()
        # update package specific buffers
        self.update_package()
        super()._post_physics_step()

    def _init_buffers(self):
        # package specific buffers
        self.package_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.package_to_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.package_lifted = torch.zeros(self.num_envs, device=self.device)

        super()._init_buffers()
        self.package.init_buffers()
        #self.table.init_buffers()

    def _reset_package_and_table(self, env_ids):
        """Resets root states of packages and tables in selected environments."""
        # reset table and randomize height
        root_state = self.package.get_default_root_state(env_ids)
        root_state[:, :3] = self.terrain.env_origins[env_ids]
        """ if self.cfg.table.max_height > 0.0:
            root_state[:, 2] += torch.empty_like(root_state[:, 2]).uniform_(0.0, self.cfg.table.max_height) """
        table_pos = root_state[:, :3]
        #self.table.set_root_state(env_ids, root_state)

        # reset package to table position and randomize orientation
        root_state [:, :3]= self.terrain.env_origins[env_ids]
        #root_state[:, :3] = table_pos
        #offsets for the lifted box
        root_state[:, 0] +=   0.4
        root_state[:, 1] +=   0
        root_state[:, 2] +=   1.5
        if self.cfg.package.max_yaw > 0.0:
            yaw = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.package.max_yaw, self.cfg.package.max_yaw
            )
            root_state[:, 3:7] = quat_mul(
                quat_from_euler_xyz(
                    torch.zeros(len(env_ids), device=self.device), torch.zeros(len(env_ids), device=self.device), yaw
                ),
                root_state[:, 3:7],
            )
        self.package.set_root_state(env_ids, root_state)

    def check_package_in_fov(self):
        # compute vector from camera to package in camera frame
        camera_to_package_c = quat_rotate_inverse(
            self.robot.camera_quat_w,
            self.package.root_pos_w - self.robot.camera_pos_w,
        )

        # check if vector is within the camera cone
        self.robot.package_in_fov = torch.logical_and(
            torch.logical_and(
                torch.abs(torch.atan(camera_to_package_c[:, 0] / camera_to_package_c[:, 2]))
                < torch.deg2rad(torch.tensor(self.robot.cfg.camera_fov[0] / 2)),
                torch.abs(torch.atan(camera_to_package_c[:, 1] / camera_to_package_c[:, 2]))
                < torch.deg2rad(torch.tensor(self.robot.cfg.camera_fov[1] / 2)),
            ),
            camera_to_package_c[:, 2] > 0,
        )

    def update_package(self):
        # compute target package position
        self.package_target_pos_w = self.robot.root_pos_w + quat_rotate(
            self.robot.root_quat_w,
            torch.tensor([0, 0.0, -0.5], device=self.device).repeat(self.num_envs, 1),
        )

        # compute distance between package and goal position
        self.package_to_goal_dist = torch.norm(self.package_target_pos_w - self.package.root_pos_w, dim=1)
        # packages that are lifted high enough and do not interact with the table count as lifted
        self.package_lifted = torch.logical_and(
            torch.norm(self.package.rigid_body_states[:,:,2], dim=-1).squeeze() > 1,  # not in contact with table
            self.package_to_goal_dist < 0.4,  # close to goal position
        ) 

    def _draw_debug_vis(self):
        """ for i in range(self.num_envs):
            viz.draw_camera(self, i)
            viz.draw_target(self, i) """
