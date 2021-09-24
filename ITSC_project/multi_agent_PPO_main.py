import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from ITSC_project.simulation_environment.rectangular_environment import *
from ITSC_project.multi_agent_dispatching.PPO.multi_agent_PPO_algorithm import *


if __name__ == '__main__':

    for seed in [4000]:  # , 8000, 12000, 16000, 20000

        height = 20
        width = 20
        low_second = 30
        high_second = 300
        grid_height = 2
        grid_width = 2
        action_interval = 180
        left_reward_to_stop = 0.01
        episode_duration = int(3600)
        vehicle_num = 4

        train_link_weight_distribution = 'UD'
        test_link_weight_distribution = 'UD'

        # allowed reward_type values are 'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance'
        reward_type = 'greedy'
        cooperative_weight = 1 / (vehicle_num * 1.5)
        negative_constant_reward = 0
        weight_shape = height * width
        share_policy = True
        ortho_init = True
        conv_params = [
            {
                'in_channels': 3,
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
                'padding': 1,
                'dilation': 1,
            },
        ]
        add_BN = True
        output_dim = [64, 32]
        share_params = False
        action_dim = 4
        learning_rate = 3e-4
        n_steps = 2048
        batch_size = int(n_steps)
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

        env = generate_rectangle_network_action_destination_env(
            height=height,
            width=width,
            low_second=low_second,
            high_second=high_second,
            grid_height=grid_height,
            grid_width=grid_width,
            action_interval=action_interval,
            left_reward_to_stop=left_reward_to_stop,
            episode_duration=episode_duration,
            link_weight_distribution=train_link_weight_distribution,
            vehicle_num=vehicle_num,
            seed=seed,
        )
        with open(project_path + '/ITSC_project/simulation_environment/experienced_travel_time/experienced_travel_time_height{}_width{}_seed{}.pickle'.format(height, width, seed), 'rb') as file:
            env.experienced_travel_time = pickle.load(file)

        model = multi_agent_PPO(
            env=env,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            vehicle_num=vehicle_num,
            weight_shape=weight_shape,
            share_policy=share_policy,
            ortho_init=ortho_init,
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            share_params=share_params,
            action_dim=action_dim,
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

        total_timesteps = 1000000
        model.learn(
            total_timesteps=total_timesteps,
            test_episode_times=100,
            train_link_weight_distribution=train_link_weight_distribution,
            test_link_weight_distribution=test_link_weight_distribution,
        )