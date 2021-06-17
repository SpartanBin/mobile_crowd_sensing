import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_environment.environment import *
from multi_agent_dispatching.asynchronous_timestep.PPO.multi_agent_PPO_algorithm import *


if __name__ == '__main__':

    height = 20
    width = 20
    ts_duration = 600
    num_of_ts = 6 * 4
    BETA = 0.5
    vehicle_num = 2

    # allowed reward_type values are 'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance'
    weight_shape = height * width
    ortho_init = True
    conv_params = [
        {
            'in_channels': vehicle_num * 3 + 3,
            'out_channels': 64,
            'kernel_size': 3,
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
    action_dim = 4
    learning_rate = 3e-4
    buffer_size_episodes = 200
    batch_size_proportion = 0.4
    n_epochs = 10
    gamma = 1  # In OpenAI Five, when set this to 0.99, policy performs best
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.01  # In OpenAI Five, when set this to 0.01, policy performs best
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    device = 'cuda'
    seed = 4000

    # generate network
    experienced_travel_time, node_id_to_grid_id = generate_rectangle_network(
        height=height,
        width=width,
        low_second=30,
        high_second=300,
        per_grid_height=2,
        per_grid_width=2,
        seed=seed,
    )

    # initial env
    env = generate_asynchronous_timestep_environment(
        experienced_travel_time=experienced_travel_time,
        node_id_to_grid_id=node_id_to_grid_id,
        ts_duration=ts_duration,
        num_of_ts=num_of_ts,
        BETA=BETA,
        vehicle_num=vehicle_num,
        seed=seed,
    )

    model = multi_agent_PPO(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        ortho_init=ortho_init,
        conv_params=conv_params,
        add_BN=add_BN,
        output_dim=output_dim,
        share_params=share_params,
        action_dim=action_dim,
        learning_rate=learning_rate,
        buffer_size_episodes=buffer_size_episodes,
        batch_size_proportion=batch_size_proportion,
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
        total_episodes=10000,
        test_episode_times=100,
        grid_weight=None,
    )