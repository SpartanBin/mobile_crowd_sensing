import sys
import os
from typing import Union
import copy
import math
import pickle
from itertools import count

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from ITSC_project.multi_agent_dispatching.common_utils import policies, multi_agent_control


class buffer():

    def __init__(self):
        self.loc = []
        self.weight = []
        self.actions = []
        self.rewards = []

    def push(self, state, actions, reward):
        self.loc.append(state[0])
        self.weight.append(state[1])
        if len(actions.shape) == 2:
            self.actions.append(actions)
        elif len(actions.shape) == 1:
            self.actions.append(actions.reshape((1, -1)))
        else:
            assert False, 'pushed actions must to have one-dim or two-dim'
        if len(reward.shape) == 2:
            self.rewards.append(reward)
        elif len(reward.shape) == 1:
            self.rewards.append(reward.reshape((1, -1)))
        else:
            assert False, 'pushed reward must to have one-dim or two-dim'

    def get(self):
        loc = np.concatenate(self.loc, axis=0)
        weight = np.concatenate(self.weight, axis=0)
        actions = np.concatenate(self.actions, axis=0)
        rewards = np.concatenate(self.rewards, axis=0)
        return loc, weight, actions, rewards


class multi_agent_DQN(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        reward_type,
        cooperative_weight,
        negative_constant_reward,
        vehicle_num: int,
        weight_shape: int,
        share_policy: bool,
        conv_params: Union[list, tuple],
        add_BN: bool,
        output_dim: Union[list, tuple],
        action_dim: int,
        learning_rate: Union[float, int] = 0.001,
        gamma: float = 0.99,
        EPS_START: float = 0.9,
        EPS_END: float = 0.05,
        EPS_DECAY: float = 1 / 20000,
        max_grad_norm: float = 0.5,
        seed: int = 4000,
        device: Union[torch.device, str] = "cpu",
    ):

        super(multi_agent_DQN, self).__init__(
            env=env,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            vehicle_num=vehicle_num,
            seed=seed,
            device=device,
        )

        self.learning_rate = learning_rate

        self.weight_shape = weight_shape
        self.gamma = gamma
        self.share_policy = share_policy
        self.conv_params = conv_params
        self.add_BN = add_BN
        self.output_dim = output_dim
        self.action_dim = action_dim

        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.max_grad_norm = max_grad_norm

        self.policy = policies.multi_agent_QLP(
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            share_policy=self.share_policy,
            conv_params=self.conv_params,
            add_BN=self.add_BN,
            output_dim=self.output_dim,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
        ).to(self.device).eval()

    def values_to_ac_probs(self, values):
        eps_threshold = (self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.num_timesteps / self.EPS_DECAY)) / (self.action_dim - 1)
        ac_probs = torch.zeros(values.shape)
        ac_probs[:] = eps_threshold
        max_index_last = values.max(-1)[1].flatten()
        max_indexs = []
        max_index = np.repeat(np.array(range(values.shape[0])).reshape(1, -1),
                              int(len(max_index_last) / values.shape[0]),
                              axis=1).flatten()
        max_indexs.append(max_index)
        max_index = np.repeat(np.array(range(values.shape[1])).reshape(1, -1),
                              int(len(max_index_last) / values.shape[1]),
                              axis=0).flatten()
        max_indexs.append(max_index)
        max_indexs.append(max_index_last)
        ac_probs[max_indexs] = 1 - (eps_threshold * (self.action_dim - 1))
        return ac_probs

    def make_one_step_forward_for_env(self, env, values, episode_time_cost, random_policy: bool):

        reselect_agent = np.array(range(self.vehicle_num))
        ac_probs_dict = {}
        if random_policy == True:
            ac_probs = torch.zeros((self.vehicle_num, self.action_dim))
            ac_probs[:] = 1 / self.action_dim
        else:
            ac_probs = self.values_to_ac_probs(values=values)[0]

        for i in reselect_agent:
            self.select_action_time += 1
            ac_probs_dict[i] = ac_probs[i]

        actions, new_obs, rewards, done, episode_time_cost = env.step_by_action_probs(
            ac_probs_dict=ac_probs_dict,
            reward_type=self.reward_type,
            cooperative_weight=self.cooperative_weight,
            negative_constant_reward=self.negative_constant_reward,
            episode_time_cost=episode_time_cost,
        )

        return actions, new_obs, rewards, done, episode_time_cost

    def sampling(self, link_weight_distribution):
        """
        """
        self.policy.eval()
        need_test = False

        one_batch = buffer()
        next_state = self.env.reset(link_weight_distribution=link_weight_distribution)

        for _ in count():

            state = next_state
            state[0] = state[0].astype(np.float32).reshape((1, -1))
            state[1] = state[1].astype(np.float32).reshape((1, 1,) + state[1].shape)

            with torch.no_grad():
                # Convert to pytorch tensor
                loc_features = state[0]
                weight_features = state[1]
                values = self.policy.forward(
                    loc_features=loc_features,
                    weight_features=weight_features,
                )

            actions, next_state, reward, done, self.episode_time_cost = self.make_one_step_forward_for_env(
                env=self.env,
                values=values,
                episode_time_cost=self.episode_time_cost,
                random_policy=False,
            )

            # Select and perform an action
            reward = np.array([reward])

            # Store the transition in memory
            one_batch.push(
                state=state,
                actions=actions,
                reward=reward,
            )

            self.num_timesteps += 1

            # Perform one step of the optimization (on the target network)
            if done:
                break

        self.episode += 1
        episode_total_score = 1 - self.env.left_reward
        self.the_last_100_episodes_total_scores.append(episode_total_score)
        self.the_best_100_episodes_total_scores.append(episode_total_score)
        self.last_100_episodes_mean_total_score = np.mean(self.the_last_100_episodes_total_scores)
        if len(self.the_last_100_episodes_total_scores) > 100:
            if self.last_100_episodes_mean_total_score > self.the_best_last_100_episodes_mean_total_score:
                self.the_best_last_100_episodes_mean_total_score = self.last_100_episodes_mean_total_score
                need_test = True
            self.the_last_100_episodes_total_scores.pop(0)
            self.the_best_100_episodes_total_scores.sort()
            self.the_best_100_episodes_total_scores.pop(0)
        if self.episode % 100 == 0:
            print('''
            ******************************************************************************************************
            in {}th episode, the number of vehicle is {}, reward_type is '{}', 
            cooperative_weight is {}, negative_constant_reward is {}, 
            seed is {}, 
            ------------------------------------------------------------------------------------------------------
            episode_time_cost = {}, episode_total_score = {}, select_action_times = {}, 
            random_policy_100_episodes_mean_total_score = {}, 
            the_best_100_episodes_mean_total_score = {}, 
            last_100_episodes_mean_total_score = {}, 
            the_best_last_100_episodes_mean_total_score = {}
            ******************************************************************************************************
            '''.format(
                self.episode, self.vehicle_num, self.reward_type, self.cooperative_weight,
                self.negative_constant_reward, self.seed, self.episode_time_cost, episode_total_score,
                self.select_action_time, self.random_policy_100_episodes_mean_total_score,
                np.mean(self.the_best_100_episodes_total_scores),
                self.last_100_episodes_mean_total_score,
                self.the_best_last_100_episodes_mean_total_score,
            ))
        self.episode_time_cost = 0
        self.select_action_time = 0

        return one_batch, need_test

    def train(self, one_episode_sample) -> None:

        # Update optimizer learning rate
        for i in range(self.vehicle_num):
            for param_group in self.policy.QLP[i].optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        loc, weight, actions, rewards = one_episode_sample.get()

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).long().unsqueeze(2)
        reward_batch = torch.tensor(rewards, device=self.device)

        non_final_next_loc = loc[1:]
        non_final_next_weight = weight[1:]
        next_state_values = torch.zeros((len(loc), self.vehicle_num), device=self.device)
        self.policy.eval()
        with torch.no_grad():
            next_state_values[: -1] = self.policy.forward(
                loc_features=non_final_next_loc,
                weight_features=non_final_next_weight,
            ).max(-1)[0].detach()

        # Compute the max q values
        max_q_values = (next_state_values * self.gamma) + reward_batch
        self.policy.train()
        estimated_q_values = self.policy.forward(
            loc_features=loc,
            weight_features=weight,
        ).gather(-1, actions)

        # Compute Huber loss
        # To minimise this error, we will use the Huber loss.
        # The Huber loss acts like the mean squared error when the error is small,
        # but like the mean absolute error when the error is large -
        # this makes it more robust to outliers when the estimates of Q are very noisy.
        loss = F.smooth_l1_loss(estimated_q_values, max_q_values.unsqueeze(2))

        # Optimize the model
        self.policy.optimize(
            loss=loss,
            max_grad_norm=self.max_grad_norm,
        )

    def test(self, test_episode_times: int, test_link_weight_distribution: str, random_policy: bool):
        self.policy.eval()
        env = copy.deepcopy(self.env)
        new_obs = env.reset(link_weight_distribution=test_link_weight_distribution)
        episodes_total_scores = []
        episodes_grid_scores = []
        episodes_got_scores = []
        all_timesteps_socre = []
        all_values = []
        for _ in range(test_episode_times):
            done = False
            episode_time_cost = 0
            episodes_grid_scores += [copy.deepcopy(env.grid_weight)]
            timesteps_socre = []
            while not done:
                new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
                new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)
                with torch.no_grad():
                    loc_features = new_obs[0]
                    weight_features = new_obs[1]
                    values = self.policy.forward(
                        loc_features=loc_features,
                        weight_features=weight_features,
                    )
                all_values.append(values.cpu().data.clone())
                actions, new_obs, _, done, episode_time_cost = self.make_one_step_forward_for_env(
                    env=env,
                    values=values,
                    episode_time_cost=episode_time_cost,
                    random_policy=random_policy,
                )
                timesteps_socre.append(1 - env.left_reward)
            episode_total_score = 1 - env.left_reward
            episodes_total_scores.append(episode_total_score)
            all_timesteps_socre.append(timesteps_socre)
            episodes_got_scores += [copy.deepcopy(env.episode_got_scores)]
            new_obs = env.reset(link_weight_distribution=test_link_weight_distribution)
        return np.mean(episodes_total_scores), episodes_grid_scores, episodes_got_scores, \
               all_timesteps_socre, all_values

    def save(self):
        self.best_state['best_episode'] = self.best_episode
        self.best_state['best_train_session'] = self.best_train_session
        self.best_state['random_policy_100_episodes_mean_total_score'] = \
            self.random_policy_100_episodes_mean_total_score
        self.best_state['the_best_100_episodes_mean_total_score'] = np.mean(self.the_best_100_episodes_total_scores)
        self.test_state['best_state'] = self.best_state
        with open('DQN_state_vehicle{}_env_{}_{}_seed{}_ID_allID_DNN_structure2_b.pickle'.format(
                self.vehicle_num, self.env.height, self.env.width, self.seed), 'wb') as file:
            pickle.dump(self.test_state, file)

    def learn(self, num_episodes: int, test_episode_times: int, train_link_weight_distribution,
              test_link_weight_distribution):

        self.init_learn(train_link_weight_distribution)
        self.random_policy_100_episodes_mean_total_score, _, _, _, _ = self.test(
            test_episode_times=test_episode_times,
            test_link_weight_distribution=test_link_weight_distribution,
            random_policy=True,
        )
        self.cur_state = self.random_policy_100_episodes_mean_total_score
        self.best_state = {'test_100_episodes_mean_total_score': self.random_policy_100_episodes_mean_total_score,
                           'policy_params': self.policy.state_dict()}
        self.test_state = {}
        self.best_episode = 0

        train_session = 0
        test_session = 0
        self.best_train_session = train_session
        while self.episode <= num_episodes:
            one_batch, need_test = self.sampling(link_weight_distribution=train_link_weight_distribution)
            self.train(one_batch)
            train_session += 1
            print('training successful in {}th training session'.format(train_session))

            if need_test:
                # ------------------------------------------test--------------------------------------------------
                test_session += 1
                self.cur_state, episodes_grid_scores, episodes_got_scores, all_timesteps_socre, all_values = \
                    self.test(
                        test_episode_times=test_episode_times,
                        test_link_weight_distribution=test_link_weight_distribution,
                        random_policy=False,
                    )

                self.test_state[test_session] = {}
                self.test_state[test_session]['test_100_episodes_mean_total_score'] = self.cur_state
                self.test_state[test_session]['episodes_grid_scores'] = episodes_grid_scores
                self.test_state[test_session]['episodes_got_scores'] = episodes_got_scores
                self.test_state[test_session]['all_timesteps_socre'] = all_timesteps_socre

                if self.cur_state > self.best_state['test_100_episodes_mean_total_score']:
                    self.best_state['test_100_episodes_mean_total_score'] = self.cur_state
                    self.best_state['policy_params'] = self.policy.state_dict()
                    self.best_state['test_session'] = test_session
                    self.best_state['all_values'] = all_values
                    self.best_episode = self.episode
                    self.best_train_session = train_session
                print('''
                **------------------------------------------------------------------------------------------**
                **------------------------------------------------------------------------------------------**
                {}th test: seed is {}, 
                now have been {}th episode, {}th training, current test 100_episodes_mean_total_score = {}; 
                best test 100_episodes_mean_total_score = {}, the result of {}th episode, {}th training is best
                **------------------------------------------------------------------------------------------**
                **------------------------------------------------------------------------------------------**
                '''.format(
                    test_session, self.seed, self.episode, train_session, self.cur_state,
                    self.best_state['test_100_episodes_mean_total_score'], self.best_episode, self.best_train_session))
                self.save()
                # ------------------------------------------------------------------------------------------------

            if self.last_100_episodes_mean_total_score <= self.the_best_last_100_episodes_mean_total_score / 2:
                break

        self.save()