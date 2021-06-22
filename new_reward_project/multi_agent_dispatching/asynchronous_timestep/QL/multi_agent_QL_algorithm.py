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

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from multi_agent_dispatching.asynchronous_timestep.common_utils import policies, multi_agent_control


class buffer():

    def __init__(self, vehicle_num):
        self.vehicle_num = vehicle_num
        self.vehicle_states = []
        self.node_weight = []
        self.grid_cover = []
        self.p = []
        self.actions = []
        self.finial_reward = None
        self.rewards = []

        for vehicle_id in range(self.vehicle_num):
            self.vehicle_states.append([])
            self.node_weight.append([])
            self.grid_cover.append([])
            self.p.append([])
            self.actions.append([])

    def push(self, state, actions, done, reward):

        if done:
            self.finial_reward = reward

        vehicle_states, node_weight, grid_cover, p, need_move = state

        for vehicle_id in need_move:
            self.vehicle_states[vehicle_id].append(vehicle_states[:, [vehicle_id] + list(range(vehicle_id)) + list(
                range(vehicle_id + 1, self.vehicle_num)), :].copy())
            self.node_weight[vehicle_id].append(node_weight.copy())
            self.grid_cover[vehicle_id].append(grid_cover.copy())
            self.p[vehicle_id].append(p.copy())
            self.actions[vehicle_id].append(copy.deepcopy(actions[vehicle_id]))

    def get(self):

        for vehicle_id in range(self.vehicle_num):
            self.vehicle_states[vehicle_id] = np.array(self.vehicle_states[vehicle_id]).squeeze()
            self.node_weight[vehicle_id] = np.array(self.node_weight[vehicle_id]).squeeze()
            self.grid_cover[vehicle_id] = np.array(self.grid_cover[vehicle_id]).squeeze()
            self.p[vehicle_id] = np.array(self.p[vehicle_id]).squeeze()
            self.actions[vehicle_id] = np.array(self.actions[vehicle_id]).squeeze()
            self.rewards.append(np.zeros(len(self.vehicle_states[vehicle_id]), dtype=np.float32))
            self.rewards[-1][-1] = self.finial_reward

        return self.vehicle_states, self.node_weight, self.grid_cover, self.p, self.actions, self.rewards


class multi_agent_QL(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        vehicle_num: int,
        weight_shape: int,
        conv_params: Union[list, tuple],
        add_BN: bool,
        output_dim: Union[list, tuple],
        action_dim: int,
        learning_rate: Union[float, int] = 0.001,
        gamma: float = 0.99,
        EPS_START: float = 0.9,
        EPS_END: float = 0.05,
        EPS_DECAY: float = 5000,
        max_grad_norm: float = 0.5,
        seed: int = 4000,
        device: Union[torch.device, str] = "cpu",
    ):

        super(multi_agent_QL, self).__init__(
            env=env,
            vehicle_num=vehicle_num,
            seed=seed,
            device=device,
        )

        self.learning_rate = learning_rate

        self.weight_shape = weight_shape
        self.gamma = gamma
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
            conv_params=self.conv_params,
            add_BN=self.add_BN,
            output_dim=self.output_dim,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
        ).to(self.device).eval()

    def transform_and_concat_all_features(
            self, vehicle_states: np.ndarray, node_weight: np.ndarray,
            grid_cover: np.ndarray, p: np.ndarray
    ) -> torch.Tensor:
        '''
        :param vehicle_states: shape = (batch_size, self.vehicle_num, 3), the third dimension is the origin node, the
        destination node and the remaining time to destination.
        :param node_weight: shape = (batch_size, number_of_node)
        :param grid_cover: shape = (batch_size, number_of_node)
        :param p: shape = (batch_size, number_of_node)
        :return: all_features, the tensor to input model directly
        '''
        batch_size = vehicle_states.shape[0]
        node_num = node_weight.shape[1]
        state_features = np.zeros((
            batch_size, 3 * self.vehicle_num, node_num), dtype=np.float32)
        # origin location
        index0 = np.repeat(np.array(range(batch_size)), self.vehicle_num, axis=0)
        index1 = np.repeat(np.array(range(0, state_features.shape[1], 3)).reshape(
            (1, -1)), batch_size, axis=0).flatten()
        index2 = vehicle_states[:, :, 0].flatten()
        index = tuple(np.vstack((index0, index1, index2)).astype(int).tolist())
        state_features[index] = 1
        # destination location
        index1 = np.repeat(np.array(range(1, state_features.shape[1], 3)).reshape(
            (1, -1)), batch_size, axis=0).flatten()
        index2 = vehicle_states[:, :, 1].flatten()
        index = tuple(np.vstack((index0, index1, index2)).astype(int).tolist())
        state_features[index] = 1
        # remaining time to destination
        index1 = range(2, state_features.shape[1], 3)
        state_features[:, index1, :] = np.repeat(vehicle_states[:, :, 2].reshape((batch_size, self.vehicle_num, 1)),
                                               repeats=node_num, axis=2)
        # concat with node_weight, grid_cover and p
        state_features = np.concatenate((
            state_features,
            node_weight.reshape((batch_size, 1, node_num)),
            grid_cover.reshape((batch_size, 1, node_num)),
            p.reshape((batch_size, 1, node_num))
        ), axis=1)
        state_features = torch.tensor(state_features, dtype=torch.float32, device=self.device)
        return state_features

    def policy_model_predict(self, vehicle_states: np.ndarray, node_weight: np.ndarray,
            grid_cover: np.ndarray, p: np.ndarray, need_move: list, train):
        '''
        :param vehicle_states: shape = (batch_size, self.vehicle_num, 3), the third dimension is the origin node, the
        destination node and the remaining time to destination.
        :param node_weight: shape = (batch_size, number_of_node)
        :param grid_cover: shape = (batch_size, number_of_node)
        :param p: shape = (batch_size, number_of_node)
        :param need_move:
        :return:

        the shape of values, log_probs, entropys = (batch_size, ).

        distributions, values are dict. values' structure is {vehicle_id1: [x], vehicle_id2: [x]...}. distributions'
        structure is {vehicle_id1: __main__.CategoricalDistribution object,
        vehicle_id2: __main__.CategoricalDistribution object...}

        '''
        if train:
            self.policy.train()
            state_features = self.transform_and_concat_all_features(
                vehicle_states=vehicle_states,
                node_weight=node_weight,
                grid_cover=grid_cover,
                p=p,
            )
            values = self.policy.forward(
                state_features=state_features,
            )
        else:
            self.policy.eval()
            if len(need_move) > 0:
                values = {}
                for vehicle_id in need_move:
                    vehicle_state = vehicle_states[:, [vehicle_id] + list(range(vehicle_id)) + list(range(
                        vehicle_id + 1, self.vehicle_num)), :]
                    state_features = self.transform_and_concat_all_features(
                        vehicle_states=vehicle_state,
                        node_weight=node_weight,
                        grid_cover=grid_cover,
                        p=p,
                    )
                    with torch.no_grad():
                        value = self.policy.forward(
                            state_features=state_features,
                        )
                    values[vehicle_id] = value.cpu().numpy().squeeze()
            else:
                with torch.no_grad():
                    state_features = self.transform_and_concat_all_features(
                        vehicle_states=vehicle_states,
                        node_weight=node_weight,
                        grid_cover=grid_cover,
                        p=p,
                    )
                    values = self.policy.forward(
                        state_features=state_features,
                    )
        return values

    def value_to_ac_probs(self, value):
        eps_threshold = (self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * (self.episode + 1) / self.EPS_DECAY)) / (self.action_dim - 1)
        ac_probs = torch.zeros(value.shape)
        ac_probs[:] = eps_threshold
        max_index = np.where(value == value.max())
        ac_probs[max_index] = 1 - (eps_threshold * (self.action_dim - 1))
        return ac_probs

    def make_one_step_forward_for_env(self, env, values: dict, episode_time_cost):

        ac_probs_dict = {}

        for i in values.keys():
            ac_prob = self.value_to_ac_probs(value=values[i])
            ac_probs_dict[i] = ac_prob

        actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = \
            env.step_by_action_probs(
                ac_probs_dict=ac_probs_dict,
                episode_time_cost=episode_time_cost,
            )

        return actions, vehicle_states, node_weight, grid_cover, \
               p, need_move, reward, done, episode_time_cost

    def sampling(self, grid_weight):
        """
        """
        self.policy.eval()
        need_test = False

        one_batch = buffer(vehicle_num=self.vehicle_num)
        vehicle_states, node_weight, grid_cover, p, need_move = self.env.reset(grid_weight=grid_weight)
        vehicle_states = vehicle_states.astype(np.float32).reshape((1,) + vehicle_states.shape)
        node_weight = node_weight.astype(np.float32).reshape((1,) + node_weight.shape)
        grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
        p = p.astype(np.float32).reshape((1,) + p.shape)
        next_state = [vehicle_states, node_weight, grid_cover, p, need_move]

        for _ in count():

            state = next_state

            vehicle_states, node_weight, grid_cover, p, need_move = state
            values = self.policy_model_predict(
                vehicle_states=vehicle_states,
                node_weight=node_weight,
                grid_cover=grid_cover,
                p=p,
                need_move=need_move,
                train=False,
            )

            actions, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, episode_time_cost = \
                self.make_one_step_forward_for_env(
                    env=self.env,
                    values=values,
                    episode_time_cost=self.episode_time_cost,
                )
            vehicle_states = vehicle_states.astype(np.float32).reshape((1,) + vehicle_states.shape)
            node_weight = node_weight.astype(np.float32).reshape((1,) + node_weight.shape)
            grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
            p = p.astype(np.float32).reshape((1,) + p.shape)
            next_state = [vehicle_states, node_weight, grid_cover, p, need_move]

            # Select and perform an action
            reward = np.array([reward])

            # Store the transition in memory
            one_batch.push(
                state=state,
                actions=actions,
                done=done,
                reward=reward,
            )

            # Perform one step of the optimization (on the target network)
            if done:
                break

        self.episode += 1
        self.the_last_100_episodes_rewards.append(reward)
        self.the_best_100_episodes_rewards.append(reward)
        self.last_100_episodes_mean_reward = np.mean(self.the_last_100_episodes_rewards)
        if len(self.the_last_100_episodes_rewards) > 100:
            if self.last_100_episodes_mean_reward > self.the_best_last_100_episodes_mean_reward:
                self.the_best_last_100_episodes_mean_reward = self.last_100_episodes_mean_reward
                need_test = True
            self.the_last_100_episodes_rewards.pop(0)
            self.the_best_100_episodes_rewards.sort()
            self.the_best_100_episodes_rewards.pop(0)
        if self.episode % 100 == 0:
            print('''
            ******************************************************************************************************
            in {}th episode, the number of vehicle is {}
            ------------------------------------------------------------------------------------------------------
            episode_time_cost = {}, episode_reward = {}, 
            random_policy_100_episodes_mean_reward = {},
            the_best_100_episodes_rewards = {},
            last_100_episodes_mean_reward = {},
            the_best_last_100_episodes_mean_reward = {}
            ******************************************************************************************************
            '''.format(
                self.episode, self.vehicle_num, self.episode_time_cost, reward,
                self.random_policy_100_episodes_mean_reward,
                np.mean(self.the_best_100_episodes_rewards),
                self.last_100_episodes_mean_reward,
                self.the_best_last_100_episodes_mean_reward,
            ))
        self.episode_time_cost = 0
        self.select_action_time = 0

        return one_batch, need_test

    def train(self, one_episode_sample) -> None:

        # Update optimizer learning rate
        self.policy.QLP.optimizer.param_groups[0]["lr"] = self.learning_rate

        vehicle_states, node_weight, grid_cover, p, actions, rewards = one_episode_sample.get()

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        max_q_values = []
        estimated_q_values = []
        for vehicle_id in range(self.vehicle_num):

            action = torch.as_tensor(
                actions[vehicle_id], dtype=torch.float32, device=self.device).long().unsqueeze(1)
            reward_batch = torch.tensor(rewards[vehicle_id], dtype=torch.float32, device=self.device)

            non_final_next_vehicle_states = vehicle_states[vehicle_id][1:]
            non_final_next_node_weight = node_weight[vehicle_id][1:]
            non_final_next_grid_cover = grid_cover[vehicle_id][1:]
            non_final_next_p = p[vehicle_id][1:]

            next_state_values = torch.zeros((len(vehicle_states[vehicle_id])), dtype=torch.float32, device=self.device)
            next_state_values[: -1] = self.policy_model_predict(
                vehicle_states=non_final_next_vehicle_states,
                node_weight=non_final_next_node_weight,
                grid_cover=non_final_next_grid_cover,
                p=non_final_next_p,
                need_move=[],
                train=False,
            ).max(-1)[0].detach()

            # Compute the max q values
            max_q_values.append((next_state_values * self.gamma) + reward_batch)

            estimated_q_values.append(
                self.policy_model_predict(
                    vehicle_states=vehicle_states[vehicle_id],
                    node_weight=node_weight[vehicle_id],
                    grid_cover=grid_cover[vehicle_id],
                    p=p[vehicle_id],
                    need_move=[],
                    train=True,
                ).gather(-1, action)
            )

        # Compute Huber loss
        # To minimise this error, we will use the Huber loss.
        # The Huber loss acts like the mean squared error when the error is small,
        # but like the mean absolute error when the error is large -
        # this makes it more robust to outliers when the estimates of Q are very noisy.

        loss = F.smooth_l1_loss(torch.cat(estimated_q_values, dim=0).squeeze(),
                                torch.cat(max_q_values, dim=0).squeeze())

        # Optimize the model
        self.policy.optimize(
            loss=loss,
            max_grad_norm=self.max_grad_norm,
        )

    # def test(self, test_episode_times: int):
    #     self.policy.eval()
    #     env = copy.deepcopy(self.env)
    #     new_obs = env.reset()
    #     episodes_total_scores = []
    #     episodes_grid_scores = []
    #     episodes_got_scores = []
    #     all_timesteps_socre = []
    #     for _ in range(test_episode_times):
    #         done = False
    #         episode_time_cost = 0
    #         episodes_grid_scores += [copy.deepcopy(env.grid_weight)]
    #         timesteps_socre = []
    #         while not done:
    #             new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
    #             new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)
    #             with torch.no_grad():
    #                 loc_features = new_obs[0]
    #                 weight_features = new_obs[1]
    #                 values = self.policy.forward(
    #                     loc_features=loc_features,
    #                     weight_features=weight_features,
    #                 )
    #             actions, new_obs, _, done, episode_time_cost = self.make_one_step_forward_for_env(
    #                 env=env,
    #                 values=values,
    #                 episode_time_cost=episode_time_cost,
    #             )
    #             timesteps_socre.append(1 - env.left_reward)
    #         episode_total_score = 1 - env.left_reward
    #         episodes_total_scores.append(episode_total_score)
    #         all_timesteps_socre.append(timesteps_socre)
    #         episodes_got_scores += [copy.deepcopy(env.episode_got_scores)]
    #         new_obs = env.reset()
    #     return np.mean(episodes_total_scores), episodes_grid_scores, episodes_got_scores, all_timesteps_socre

    # def save(self):
    #     self.best_state['best_episode'] = self.best_episode
    #     self.best_state['best_train_session'] = self.best_train_session
    #     self.best_state['random_policy_100_episodes_mean_total_score'] = \
    #         self.random_policy_100_episodes_mean_total_score
    #     self.best_state['the_best_100_episodes_mean_total_score'] = np.mean(self.the_best_100_episodes_total_scores)
    #     self.test_state['best_state'] = self.best_state
    #     with open('QL_state_vehicle{}_env_{}_{}.pickle'.format(
    #             self.vehicle_num, self.env.height, self.env.width), 'wb') as file:
    #         pickle.dump(self.test_state, file)

    def learn(self, num_episodes: int, grid_weight):

        self.init_learn(grid_weight=grid_weight)
        self.random_policy_100_episodes_mean_total_score = 0
        self.cur_state = self.random_policy_100_episodes_mean_total_score
        self.best_state = {'test_100_episodes_mean_total_score': self.random_policy_100_episodes_mean_total_score,
                           'policy_params': self.policy.state_dict()}
        self.test_state = {}
        self.best_episode = 0

        train_session = 0
        test_session = 0
        self.best_train_session = train_session
        while self.episode <= num_episodes:
            one_batch, need_test = self.sampling(grid_weight=grid_weight)
            self.train(one_batch)
            train_session += 1
            print('training successful in {}th training session'.format(train_session))

        #     if need_test:
        #         # ------------------------------------------test--------------------------------------------------
        #         test_session += 1
        #         self.cur_state, episodes_grid_scores, episodes_got_scores, all_timesteps_socre = self.test(
        #             test_episode_times=test_episode_times)
        #
        #         self.test_state[test_session] = {}
        #         self.test_state[test_session]['test_100_episodes_mean_total_score'] = self.cur_state
        #         self.test_state[test_session]['episodes_grid_scores'] = episodes_grid_scores
        #         self.test_state[test_session]['episodes_got_scores'] = episodes_got_scores
        #         self.test_state[test_session]['all_timesteps_socre'] = all_timesteps_socre
        #
        #         if self.cur_state > self.best_state['test_100_episodes_mean_total_score']:
        #             self.best_state['test_100_episodes_mean_total_score'] = self.cur_state
        #             self.best_state['policy_params'] = self.policy.state_dict()
        #             self.best_state['test_session'] = test_session
        #             self.best_episode = self.episode
        #             self.best_train_session = train_session
        #         print('''
        #         **------------------------------------------------------------------------------------------**
        #         **------------------------------------------------------------------------------------------**
        #         {}th test:
        #         now have been {}th episode, {}th training, current test 100_episodes_mean_total_score = {};
        #         best test 100_episodes_mean_total_score = {}, the result of {}th episode, {}th training is best
        #         **------------------------------------------------------------------------------------------**
        #         **------------------------------------------------------------------------------------------**
        #         '''.format(
        #             test_session, self.episode, train_session, self.cur_state,
        #             self.best_state['test_100_episodes_mean_total_score'], self.best_episode, self.best_train_session))
        #         self.save()
        #         # ------------------------------------------------------------------------------------------------
        #
        #     if self.last_100_episodes_mean_total_score <= self.the_best_last_100_episodes_mean_total_score / 2:
        #         break
        #
        # self.save()