import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from new_reward_project.simulation_environment.environment import *
from new_reward_project.multi_agent_dispatching.synchronous_timestep.GA.multi_agent_GA_algorithm import *


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
    vehicle_num = 2
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
            'padding': 2,
            'dilation': 1,
        },
    ]
    output_dim = [64, 32]
    num_episodes_to_cal = 100
    pop_size = 50
    bound = None
    GenomeClass = GenomeBinary
    cross_prob = 0.8
    mutate_prob = 0.03

    model = multi_agent_GA(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        share_policy=share_policy,
        conv_params=conv_params,
        add_BN=False,
        output_dim=output_dim,
        action_dim=ac_dim,
        num_episodes_to_cal=num_episodes_to_cal,
        pop_size=pop_size,
        bound=bound,
        GenomeClass=GenomeClass,
        cross_prob=cross_prob,
        mutate_prob=mutate_prob,
        seed=seed,
    )
    model.genetic(num_gen=1000)