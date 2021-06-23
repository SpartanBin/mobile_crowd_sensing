import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from new_reward_project.simulation_environment.environment import *
from new_reward_project.multi_agent_dispatching.synchronous_timestep.PPO.multi_agent_PPO_algorithm import *


if __name__ == '__main__':

    # generate network
    height = 20
    width = 20
    low_second = 30
    high_second = 300
    per_grid_height = 2
    per_grid_width = 2
    seed = 4000
    experienced_travel_time, node_id_to_grid_id = generate_rectangle_network(
        height=height,
        width=width,
        low_second=low_second,
        high_second=high_second,
        per_grid_height=per_grid_height,
        per_grid_width=per_grid_width,
        seed=seed,
    )

    # initial synchronous env
    ac_dim = 8
    action_interval = 180
    num_of_action_interval = 4
    num_of_cal_reward = 5
    BETA = 0.5
    vehicle_num = 10
    env = generate_synchronous_timestep_environment_with_directional_action(
        experienced_travel_time=experienced_travel_time,
        node_id_to_grid_id=node_id_to_grid_id,
        ac_dim=ac_dim,
        action_interval=action_interval,
        num_of_action_interval=num_of_action_interval,
        num_of_cal_reward=num_of_cal_reward,
        BETA=BETA,
        vehicle_num=vehicle_num,
        seed=seed,
    )

    weight_shape = height * width
    share_policy = True
    ortho_init = True
    conv_params = [
        {
            'in_channels': vehicle_num * 3 + 3,
            'out_channels': 64,
            'kernel_size': 5,
            'stride': 2,
            'padding': 2,
            'dilation': 1,
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 2,
            'padding': 2,
            'dilation': 1,
        },
    ]
    add_BN = True
    output_dim = [64, 32]
    share_params = False
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = n_steps
    n_epochs = 10
    gamma = 0.99  # In OpenAI Five, when set this to 0.99, policy performs best
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.01  # In OpenAI Five, when set this to 0.01, policy performs best
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    device = 'cuda'

    model = multi_agent_PPO(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        share_policy=share_policy,
        ortho_init=ortho_init,
        conv_params=conv_params,
        add_BN=add_BN,
        output_dim=output_dim,
        share_params=share_params,
        action_dim=ac_dim,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        seed=seed,
        device=device,
    )
    model.learn(
        total_timesteps=1000000,
        test_episode_times=100,
        grid_weight=None,
    )