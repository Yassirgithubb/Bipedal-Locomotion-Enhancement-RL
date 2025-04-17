from typing import List, Tuple, Union
from legged_gym.utils.config_utils import configclass


@configclass
class PolicyCfg:
    init_noise_std: float = 1.0
    actor_hidden_dims: Tuple = (512, 256, 128)
    critic_hidden_dims: Tuple = (512, 256, 128)
    activation: str = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    # only for 'ActorCriticRecurrent':
    # rnn_type = 'lstm'
    # rnn_hidden_size = 512
    # rnn_num_layers = 1


@configclass
class AlgorithmCfg:
    # training params
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4  # mini batch size = num_envs * nsteps / nminibatches
    learning_rate: float = 1.0e-3  # 5.e-4
    schedule: str = "adaptive"  # adaptive, fixed
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class RunnerCfg:
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "PPO"
    num_steps_per_env: int = 24  # per iteration
    max_iterations: int = 1500  # number of policy updates
    empirical_normalization: bool = False

    # logging
    save_interval: int = 50  # check for potential saves every this many iterations
    experiment_name: str = "test"
    run_name: Union[int, str] = ""
    # load and resume
    resume: bool = False
    load_run: Union[int, str] = -1  # -1 = last run
    checkpoint: int = -1  # -1 = last saved model
    resume_path: str = None  # updated from load_run and chkpt

    logger: str = "tensorboard"  # tensorboard, wandb, or neptune
    wandb_project: str = "legged_gym"
    neptune_project: str = "legged_gym"


@configclass
class PPOCfg:
    seed: int = 1
    runner_class_name: str = "OnPolicyRunner"
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    runner: RunnerCfg = RunnerCfg()