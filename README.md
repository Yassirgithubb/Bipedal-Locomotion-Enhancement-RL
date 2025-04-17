# Isaac Gym Environments for Legged Robots

This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction randomization, noisy observations and random pushes during training.

**Project website:** https://leggedrobotics.github.io/legged_gym/   
**Paper:** https://arxiv.org/abs/2109.11978    

**Maintainer**: Nikita Rudin  
**Affiliation**: Robotic Systems Lab, ETH Zurich  
**Contact**: rudinn@ethz.ch  


## Installation
1. install dependencies `sudo apt-get install openscad`
2. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
3. Create the folder for the repositories

        mkdir ~/issac_ws

4. Install pytorch 1.12 with cuda-11.6 (you don't need to cuda installed separately):

        pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

5. Install Isaac Gym

      - Download [isaacgym](https://developer.nvidia.com/isaac-gym) and move it to ~/isaac_ws
   
            cd ~/isaac_ws/isaacgym/python
            pip install -e .
            
      - Try running an example:
   
            cd examples
            python 1080_balls_of_solitude.py 
            
      - For troubleshooting, check docs at: `isaacgym/docs/index.html`

6. Install [rsl_rl](https://bitbucket.org/leggedrobotics/rsl_rl) (PPO implementation):

        cd ~/isaac_ws
        git clone git@bitbucket.org:leggedrobotics/rsl_rl.git
        pip install -e rsl_rl

7. Install [legged_gym](https://bitbucket.org/leggedrobotics/legged_gym)

        cd ~/isaac_ws
        git clone git@bitbucket.org:leggedrobotics/legged_gym.git
        pip install -e legged_gym

### Code Structure

1. Each environment is defined by a python file (ex: [`legged_env.py`](legged_gym/envs/anymal_c/anymal.py)) and two python configuration classes: one for the environment `cfg ` and one for training `train_cfg` (ex: [`anymal_c_rough_config.py`](legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py)).

## Usage

### Train:

```bash
python scripts/train.py --task=anymal_c_flat
```

The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`, where `<experiment_name>` and `<run_name>` are defined in the train config.

Furthermore, the `git status` and `git diff` outputs of this repository (`legged_gym`) and of the RL repository (`rsl_rl`) are stored as .diff files in the logdirectory next to the model files. It is suggested to use Visual Studio Code (or other code editors) to view them, since they support diff syntax highlighting. You can use `code logs/<experiment_name>/<date_time>_<run_name>/legged_gym_git.diff` to view the file. This way the exact code configuration is always stored with every run.

#### Arguments:

- To run on CPU, append following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
- To run headless (no rendering), append `--headless`.
- The following command line arguments override the values set in the config files:
  - `--task TASK`: Task name.
  - `--resume`: Resume training from a checkpoint
  - `--experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load.
  - `--run_name RUN_NAME`: Name of the run.
  - `--load_run LOAD_RUN`: Name of the run to load when resume=True. If -1: will load the last run.
  - `--checkpoint CHECKPOINT`: Saved model checkpoint number. If -1: will load the last checkpoint.
  - `--num_envs NUM_ENVS`: Number of environments to create.
  - `--seed SEED`: Random seed.
  - `--max_iterations MAX_ITERATIONS`: Maximum number of training iterations.

#### Logging
For logging, either [Tensorboard](https://www.tensorflow.org/tensorboard/get_started), [Neptune.ai](https://neptune.ai/), or [Weights & Biases](https://wandb.ai/) can be used. The desired logger can be specified in `RunnerCfg.logger` (default: tensorboard).

Tensorboard can be launched from the command line. Logging is saved in the specified`log_dir` (default: `<legged_gym>/logs`). To open tensorboard, run:

```bash
# from top of the repository
`tensorboard --logdir logs/<experiment_name>`
```
For Neptune and Weights & Biases, create a profile and a project on the desired service. Specify your project's name in `RunnerCfg.{wandb/neptune}_project` (default: `legged_gym`). Your username is read from the environment variable `{WANDB/NEPTUNE}_USERNAME`. Add one of the following lines to your ~/.bashrc:
```bash
export NEPTUNE_USERNAME=YOUR_USERNAME
export WANDB_USERNAME=YOUR_USERNAME
```
### Play a trained policy:  

```bash
python scripts/play.py --task=anymal_c_flat
```

- By default the loaded policy is the last model of the last run of the experiment folder.
- Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

- If you have a joystick device (such as a gampead), then it is possible to also control a given robot in the environment, through the joystick. Following are the settings available for [PlayStation controller](https://www.playstation.com/en-us/accessories/dualshock-4-wireless-controller/)

      Robot Steering
      --------------
            Left joystick   :   move the robot around in x-y-direction.
            Right joystick  :   move the robot around its yaw axis.

      Camera Steering
      ---------------
            L1 button       :   cinematic mode clockwise
            R1 button       :   cinematic mode counter clockwise
            L1 + R1 button  :   stop cinematic mode
            L2 button       :   manually rotate the camera clockwise
            R2 button       :   manually rotate the camera counter clockwise
            Square button   :   zoom in
            Circle button   :   zoom out
            X button        :   move camera down
            Triangle button :   move camera up

            Options Button  :   randomly switch to another robot
            Share button    :   reset camera to initial state

> **Note:** You can also change how the joystick controller interacts with your environment (e.g. instead of giving target velocities, you can move the target robot position). If you are using a different joystick, just make sure to adjust the mapping of joystick axis / button-ids to your new joystick in legged_gym/utils/joystick.py:120 and following lines. In the same file, you can adjust the camera movement velocity in Lines 26-30.

### Adding a new environment:

The base environment [`legged_env`](legged_gym/envs/base/legged_robot.py) implements a rough terrain locomotion task. The [corresponding cfg](legged_gym/envs/base/legged_robot_config.py) does not specify a robot asset (URDF/ MJCF).

1. Add a new folder to [`legged_gym/envs`](legged_gym/envs) with the name of your environment.
2. Add the `<your_env>_config.py`, which inherits from an existing environment cfgs.
3. If adding a new robot:
    - Add the corresponding assets to [`resources`](resources)
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
4. (If needed) Implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
5. Register your env in [`legged_gym/envs/__init__.py`](legged_gym/envs/__init__.py).
6. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!
7. (If needed) Customize the keyboard controller for the [play-mode](scripts/play.py). More Infos [here](legged_gym/utils/README.md).

## Contribution Guidelines

We use [`black`](https://github.com/psf/black) formatter for formatting the python code and [`flake8`](https://flake8.pycqa.org/en/latest/) for linting. To run the formatter:

```bash
# for formatting
pip install black
black --line-length 120 .
# for checking lints
pip install flake8
flake8 .
```

## Troubleshooting

To fix the following error:
```
ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory
```
install python-dev:
```bash
sudo apt install libpython3.8
```

### Known Issues

The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`.

Example:

```python
sensor_pose = gymapi.Transform()
for name in feet_names:
    sensor_options = gymapi.ForceSensorProperties()
    sensor_options.enable_forward_dynamics_forces = False # for example gravity
    sensor_options.enable_constraint_solver_forces = True # for example contacts
    sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
    index = self.gym.find_asset_rigid_body_index(robot_asset, name)
    self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
(...)

sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
self.gym.refresh_force_sensor_tensor(self.sim)
force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
(...)

self.gym.refresh_force_sensor_tensor(self.sim)
contact = self.sensor_forces[:, :, 2] > 1.
```
