import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from simulation_environment.environment import *
from multi_agent_dispatching.asynchronous_timestep.QL.multi_agent_QL_algorithm import *


if __name__ == '__main__':

    height = 20
    width = 20
    ts_duration = 600
    num_of_ts = 6 * 12
    BETA = 0.5
    vehicle_num = 2
    num_episodes = 100000

    # allowed reward_type values are 'greedy', 'sum', 'greedy_mean', 'team_spirit', 'distance'
    reward_type = 'greedy'
    cooperative_weight = 1 / (vehicle_num * 1.5)
    negative_constant_reward = 0
    weight_shape = height * width
    share_policy = True
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
    action_dim = height * width
    learning_rate = 0.001
    gamma = 1
    EPS_START = 0.9
    EPS_END = 0.05
    max_grad_norm = 0.5
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

    model = multi_agent_QL(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        conv_params=conv_params,
        add_BN=add_BN,
        output_dim=output_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        EPS_START=EPS_START,
        EPS_END=EPS_END,
        EPS_DECAY=num_episodes / 4,
        max_grad_norm=max_grad_norm,
        seed=seed,
        device=device,
    )
    model.learn(
        num_episodes=num_episodes,
        grid_weight=None,
    )