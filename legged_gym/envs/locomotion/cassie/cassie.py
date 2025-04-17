"""HAS NOT BEEN ADAPTED TO THE NEW STRUCTURE"""

# python
import torch

# legged-gym
from legged_gym.envs.locomotion import LeggedRobot
from legged_gym.common.rewards.reward_manager import RewardManager


class Cassie(LeggedRobot):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, reward_manager_cls=CassieRewardManager)


class CassieRewardManager(RewardManager):
    def no_fly(self, env, params):
        contacts = env.contact_forces[:, env.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact
