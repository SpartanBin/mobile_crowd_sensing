from typing import Optional, Union
import random

import numpy as np
import torch


class multi_agent():

    def __init__(
        self,
        env,
        vehicle_num: int,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)
        if device == 'cuda':
            # Deterministic operations for CuDNN, it may impact performances
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.num_timesteps = 0

        self.env = env

        self.vehicle_num = vehicle_num
        self.device = torch.device(device)

    def make_one_step_forward_for_env(self, env, distributions, episode_time_cost):

        reselect_agent = np.array(range(self.vehicle_num))
        ac_probs_dict = {}
        ac_dict = {}
        actions = []

        for i in reselect_agent:
            self.select_action_time += 1
            distribution = distributions[i]
            action = distribution.get_actions().item()
            ac_dict[i] = action
            actions.append(action)
            ac_probs_dict[i] = distribution.all_probs()[0]
        actions = np.array(actions)

        vehicle_states, node_weight, grid_cover, p, reward, done, episode_time_cost = env.step(
            ac_dict=ac_dict,
            episode_time_cost=episode_time_cost,
        )

        log_probs = np.array([np.nan] * self.vehicle_num)
        ac = torch.tensor(actions).view((1, -1)).to(self.device)
        for i in reselect_agent:
            distribution = distributions[i]
            log_probs[i] = distribution.log_prob(ac[:, i]).cpu().numpy()[0]

        return actions, log_probs, vehicle_states, node_weight, grid_cover, p, reward, done, episode_time_cost

    def init_learn(self, grid_weight):

        self.num_timesteps = 0
        self.episode_time_cost = 0
        self.episode = 0
        self.the_last_100_episodes_got_rewards = []
        self.last_100_episodes_mean_got_reward = - 100000000000000000000
        self.the_best_last_100_episodes_mean_got_reward = - 100000000000000000000
        self.the_best_100_episodes_got_rewards = []
        self.random_policy_100_episodes_mean_got_reward = None
        vehicle_states, node_weight, grid_cover, p = self.env.reset(grid_weight=grid_weight)
        vehicle_states = vehicle_states.astype(np.float32).reshape((1,) + vehicle_states.shape)
        node_weight = node_weight.astype(np.float32).reshape((1,) + node_weight.shape)
        grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
        p = p.astype(np.float32).reshape((1,) + p.shape)
        self._last_obs = [vehicle_states, node_weight, grid_cover, p]
        self._last_done = False
        self.select_action_time = 0