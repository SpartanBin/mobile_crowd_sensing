import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from ITSC_project.simulation_environment.rectangular_environment import *
from ITSC_project.multi_agent_dispatching.DQN.multi_agent_DQN_algorithm import *


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
        episode_duration = 3600
        vehicle_num = 2
        num_episodes = 100000

        train_link_weight_distribution = 'UD'
        test_link_weight_distribution = 'UD'

        # allowed reward_type values are 'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance'
        reward_type = 'greedy'
        cooperative_weight = 1 / (vehicle_num * 1.5)
        negative_constant_reward = 0
        weight_shape = height * width
        share_policy = True
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
        action_dim = 4
        learning_rate = 0.001
        gamma = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        max_grad_norm = 0.5
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

        model = multi_agent_DQN(
            env=env,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            vehicle_num=vehicle_num,
            weight_shape=weight_shape,
            share_policy=share_policy,
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            EPS_START=EPS_START,
            EPS_END=EPS_END,
            EPS_DECAY=1 / num_episodes,
            max_grad_norm=max_grad_norm,
            seed=seed,
            device=device,
        )
        model.learn(
            num_episodes=num_episodes,
            test_episode_times=100,
            train_link_weight_distribution=train_link_weight_distribution,
            test_link_weight_distribution=test_link_weight_distribution,
        )