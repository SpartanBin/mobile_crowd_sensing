from typing import Optional, Union
import random

import numpy as np
import torch


class multi_agent():

    def __init__(
        self,
        env,
        reward_type,
        cooperative_weight,
        negative_constant_reward,
        vehicle_num: int,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        self.seed = seed
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
        self.reward_type = reward_type
        self.cooperative_weight = cooperative_weight
        self.negative_constant_reward = negative_constant_reward

        self.vehicle_num = vehicle_num
        self.device = torch.device(device)

    def make_one_step_forward_for_env(self, env, distributions, episode_time_cost):

        reselect_agent = np.array(range(self.vehicle_num))
        ac_probs_dict = {}

        for i in reselect_agent:
            self.select_action_time += 1
            distribution = distributions[i]
            ac_probs_dict[i] = distribution.all_probs()[0]

        actions, new_obs, rewards, done, episode_time_cost = env.step_by_action_probs(
            ac_probs_dict=ac_probs_dict,
            reward_type=self.reward_type,
            cooperative_weight=self.cooperative_weight,
            negative_constant_reward=self.negative_constant_reward,
            episode_time_cost=episode_time_cost,
        )

        log_probs = np.array([np.nan] * self.vehicle_num)
        ac = torch.tensor(actions).view((1, -1)).to(self.device)
        for i in reselect_agent:
            distribution = distributions[i]
            log_probs[i] = distribution.log_prob(ac[:, i]).cpu().numpy()[0]

        return actions, log_probs, new_obs, rewards, done, episode_time_cost

    def init_learn(self, train_link_weight_distribution):

        self.num_timesteps = 0
        self.episode_time_cost = 0
        self.episode = 0
        self.the_last_100_episodes_total_scores = []
        self.last_100_episodes_mean_total_score = 0
        self.the_best_last_100_episodes_mean_total_score = 0
        self.the_best_100_episodes_total_scores = []
        self.random_policy_100_episodes_mean_total_score = None
        self._last_obs = self.env.reset(link_weight_distribution=train_link_weight_distribution)
        self._last_obs[0] = self._last_obs[0].astype(np.float32).reshape((1, -1))
        self._last_obs[1] = self._last_obs[1].astype(np.float32).reshape(
            (1, 1,) + self._last_obs[1].shape)
        self._last_done = False
        self.select_action_time = 0