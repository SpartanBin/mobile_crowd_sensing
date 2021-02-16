import sys
import os

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from simulation_env.environment import *
from multi_agent_dispatching.PPO.multi_agent_PPO_algorithm import *


if __name__ == '__main__':

    height = 20
    width = 20
    low_second = 30
    high_second = 300
    grid_height = 2
    grid_width = 2
    action_interval = 180
    episode_duration = 24 * 3600
    vehicle_num = 1

    loc_dim = 4
    weight_shape = (height, width)
    share_policy = True
    ortho_init = True
    loc_feature_dim = [128]
    weight_feature_params = [
        {
            'in_channels': 1,
            'out_channels': 64,
            'kernel_size': (5, 5),
            'stride': (2, 2),
            'padding': (2, 2),
            'dilation': (1, 1),
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': (5, 5),
            'stride': (2, 2),
            'padding': (2, 2),
            'dilation': (1, 1),
        },
    ]
    output_dim = [128, 128]
    share_params = False
    action_dim = 4
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 1024
    n_epochs = 10
    gamma = 0.9
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    device = 'cuda'
    seed = 10000

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

    model = multi_agent_PPO(
        env=env,
        vehicle_num=vehicle_num,
        loc_dim=loc_dim,
        weight_shape=weight_shape,
        share_policy=share_policy,
        ortho_init=ortho_init,
        loc_feature_dim=loc_feature_dim,
        weight_feature_params=weight_feature_params,
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
    model.learn(total_timesteps=1000000)