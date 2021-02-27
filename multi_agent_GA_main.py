import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from simulation_env.environment import *
from multi_agent_dispatching.GA.multi_agent_GA_algorithm import *


if __name__ == '__main__':

    height = 20
    width = 20
    low_second = 30
    high_second = 300
    grid_height = 2
    grid_width = 2
    action_interval = 180
    episode_duration = 3600 * 24 * 4
    vehicle_num = 4

    weight_shape = (height, width)
    share_policy = True
    weight_feature_params = [
        {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': (5, 5),
            'stride': (2, 2),
            'padding': (2, 2),
            'dilation': (1, 1),
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': (3, 3),
            'stride': (2, 2),
            'padding': (2, 2),
            'dilation': (1, 1),
        },
    ]
    output_dim = [64, 32]
    action_dim = 4
    num_episodes_to_cal = 10
    pop_size = 20
    bound = None
    GenomeClass = GenomeBinary
    cross_prob = 0.8
    mutate_prob = 0.03
    seed = 4000

    env = generate_rectangle_network_action_destination_env(
        height=height,
        width=width,
        low_second=low_second,
        high_second=high_second,
        grid_height=grid_height,
        grid_width=grid_width,
        action_interval=action_interval,
        episode_duration=episode_duration,
        vehicle_num=vehicle_num,
        seed=seed,
    )

    model = multi_agent_GA(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        share_policy=share_policy,
        conv_params=weight_feature_params,
        output_dim=output_dim,
        action_dim=action_dim,
        num_episodes_to_cal=num_episodes_to_cal,
        pop_size=pop_size,
        bound=bound,
        GenomeClass=GenomeClass,
        cross_prob=cross_prob,
        mutate_prob=mutate_prob,
        seed=seed,
    )
    model.genetic(num_gen=1000)