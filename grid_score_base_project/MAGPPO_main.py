import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from grid_score_base_project.simulation_environment.environment import *
from grid_score_base_project.multi_agent_dispatching.MAGPPO.multi_agent_PPO_algorithm import *


if __name__ == '__main__':

    # initial env
    T = 120
    vehicle_num = 2

    vehicle_speed = 8
    with open(project_path + '/grid_score_base_project/lq_network_data/node_base_features/adjacency_matrix_with_edge_id.pickle', 'rb') as file:
        adjacency_matrix_with_edge_id = pickle.load(file)
    edge_info = pd.read_excel(
        project_path + '/grid_score_base_project/lq_network_data/node_base_features/edge_info.xls')[[
        'edge_id', 'length']]
    endpoints_of_each_edge = pd.read_excel(
        project_path + '/grid_score_base_project/lq_network_data/node_base_features/endpoints_of_each_edge.xls')[[
        'edge_id', 'node_id_one', 'node_id_two']]
    node_info = pd.read_excel(
        project_path + '/grid_score_base_project/lq_network_data/node_base_features/node_info.xls')[[
        'node_id', 'grid_id', 'project_x', 'project_y']]
    node_info.rename(columns={'project_x': 'node_coordinate_x', 'project_y': 'node_coordinate_y'}, inplace=True)
    grid_weight = pd.read_csv(
        project_path + '/grid_score_base_project/lq_network_data/weight.csv')[[
        'grid_id', 'weight']]
    grid_weight.rename(columns={'weight': 'grid_weight'}, inplace=True)
    taxi_cover_times = pd.read_csv(
        project_path + '/grid_score_base_project/lq_network_data/taxi_cover_times.csv')[[
        'grid_id', '9', '10', '11', '12', '13', '14']]
    ac_dim = 8
    action_interval = 180
    seed = 4000
    env = generate_environment_with_grid_score_directional_action_node_base_features(
        vehicle_speed=vehicle_speed,
        adjacency_matrix_with_edge_id=adjacency_matrix_with_edge_id,
        edge_info=edge_info,
        node_info=node_info,
        grid_weight=grid_weight,
        taxi_cover_times=taxi_cover_times,
        ac_dim=ac_dim,
        action_interval=action_interval,
        T=T,
        vehicle_num=vehicle_num,
        seed=seed,
    )

    weight_shape = len(env.node)
    ortho_init = True
    in_channels = 5
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = int(2048 * 1.4)  # 1.4
    n_epochs = 10
    gamma = 0.99  # In OpenAI Five, when set this to 0.99, policy performs best
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.002
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    device = 'cuda'

    model = multi_agent_PPO(
        env=env,
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        ortho_init=ortho_init,
        in_channels=in_channels,
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
        total_timesteps=4000000,
        test_episode_times=100,
    )