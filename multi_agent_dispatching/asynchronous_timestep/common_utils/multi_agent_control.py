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

        self.num_episodes = 0

        self.env = env

        self.vehicle_num = vehicle_num
        self.device = torch.device(device)

    def make_one_step_forward_for_env(self, env, distributions: dict, episode_time_cost):

        ac_probs_dict = {}

        for i in distributions.keys():
            self.select_action_time += 1
            distribution = distributions[i]
            ac_probs_dict[i] = distribution.all_probs()[0]

        actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = \
            env.step_by_action_probs(
                ac_probs_dict=ac_probs_dict,
                episode_time_cost=episode_time_cost,
            )

        log_probs = {}
        for i in actions.keys():
            distribution = distributions[i]
            ac = torch.tensor(actions[i]).to(self.device)
            log_probs[i] = distribution.log_prob(ac).cpu().numpy()[0]

        return actions, log_probs, vehicle_states, node_weight, grid_cover, \
               p, need_move, reward, done, episode_time_cost

    def init_learn(self, grid_weight):

        self.episode_time_cost = 0
        self.episode = 0
        self.the_last_100_episodes_rewards = []
        self.last_100_episodes_mean_reward = - 10000000000000000000
        self.the_best_last_100_episodes_mean_reward = - 10000000000000000000
        self.the_best_100_episodes_rewards = []
        self.random_policy_100_episodes_mean_reward = None
        vehicle_states, node_weight, grid_cover, p, need_move = self.env.reset(grid_weight=grid_weight)
        vehicle_states = vehicle_states.astype(np.float32).reshape((1, ) + vehicle_states.shape)
        node_weight = node_weight.astype(np.float32).reshape((1, ) + node_weight.shape)
        grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
        p = p.astype(np.float32).reshape((1,) + p.shape)
        self._last_obs = [vehicle_states, node_weight, grid_cover, p, need_move]
        self._last_done = False
        self.select_action_time = 0