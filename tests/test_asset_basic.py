# isaacgym
from isaacgym import gymapi

# python
import torch
import math

# legged-gym
from legged_gym.common.gym_interface import GymInterface, GymInterfaceCfg
from legged_gym.common.assets.asset import CuboidAsset, SphereAsset
from legged_gym.common.assets.asset_cfg import CuboidAssetCfg, SphereAssetCfg

if __name__ == "__main__":
    # configure simulator
    gym_cfg = GymInterfaceCfg()
    gym_cfg.sim_params.dt = 1.0 / 60.0
    gym_cfg.sim_params.substeps = 2
    gym_cfg.sim_params.num_position_iterations = 4
    gym_cfg.sim_params.num_velocity_iterations = 1
    # create simulation app
    gym_interface = GymInterface(gym_cfg)

    # set up the env grid
    num_envs = 1024
    num_per_row = int(math.sqrt(num_envs))
    spacing = 0.25
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, 1.0)
    spawn_sphere = True

    # assets
    # -- ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym_interface.gym.add_ground(gym_interface.sim, plane_params)
    # -- cube
    cube_config = CuboidAssetCfg(asset_name="cube", width=0.1, height=0.1, depth=0.2, density=1200)
    cube_config.init_state.pos = (0.0, 0.0, 2.0)
    cube_config.randomization.randomize_color = 2
    cube = CuboidAsset(cube_config, num_envs, gym_interface)
    # --sphere
    if spawn_sphere:
        sphere_config = SphereAssetCfg(asset_name="sphere", radius=0.1, density=20000)
        sphere_config.init_state.pos = (0.0, 0.0, 5.0)
        sphere_config.randomization.randomize_color = 2
        sphere = SphereAsset(sphere_config, num_envs, gym_interface)

    envs = []
    # create env
    for env_id in range(num_envs):
        # create env
        env = gym_interface.gym.create_env(gym_interface.sim, env_lower, env_upper, num_per_row)
        envs.append(env)
        # spawn assets
        cube.spawn(env_id)
        if spawn_sphere:
            sphere.spawn(env_id)

    # prepare simulation buffers
    gym_interface.prepare_sim()
    cube.init_buffers()
    cube.reset_buffers()
    if spawn_sphere:
        sphere.init_buffers()
        sphere.reset_buffers()
    # set camera view
    gym_interface.set_camera_view((20, 20, 5), (0, 0, 1))

    # perform stepping
    step_count = 0

    while not gym_interface.gym.query_viewer_has_closed(gym_interface.viewer):
        step_count += 1

        # Get input actions from the viewer and handle them appropriately
        if step_count % 200 == 0:
            ids = torch.randint(0, num_envs - 1, (2 * num_per_row,), device=gym_interface.device)
            root_states = cube.get_default_root_state(ids)
            cube.set_root_state(ids, root_states)
            cube.reset_buffers()
            if spawn_sphere:
                root_states = sphere.get_default_root_state(ids)
                sphere.set_root_state(ids, root_states)
                sphere.reset_buffers()
            # set things into sim
            # gym_interface.gym.set_actor_root_state_tensor(gym_interface.sim, sim_root_states)
            gym_interface.write_states_to_sim()
        # step the physics
        gym_interface.simulate()
        # refresh tensors
        gym_interface.refresh_tensors(root_state=True)
        # update assets
        cube.update_buffers()
        if spawn_sphere:
            sphere.update_buffers()
        # update the viewer
        gym_interface.render()
