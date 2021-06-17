import sys
import os
from typing import Optional, Union
import copy
import pickle

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from multi_agent_dispatching.asynchronous_timestep.common_utils import policies, multi_agent_control
from multi_agent_dispatching.asynchronous_timestep.PPO import buffers


class multi_agent_PPO(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        vehicle_num: int,
        weight_shape: int,
        ortho_init: bool,
        conv_params: Union[list, tuple],
        add_BN: bool,
        output_dim: Union[list, tuple],
        share_params: bool,
        action_dim: int,
        learning_rate: Union[float, int] = 3e-4,
        buffer_size_episodes: int = 200,
        batch_size_proportion: Optional[int] = 0.4,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        seed: int = 4000,
        device: Union[torch.device, str] = "cpu",
    ):

        super(multi_agent_PPO, self).__init__(
            env=env,
            vehicle_num=vehicle_num,
            seed=seed,
            device=device,
        )

        self.learning_rate = learning_rate

        self.buffer_size_episodes = buffer_size_episodes
        self.weight_shape = weight_shape
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ortho_init = ortho_init
        self.conv_params = conv_params
        self.add_BN = add_BN
        self.output_dim = output_dim
        self.share_params = share_params
        self.action_dim = action_dim

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size_proportion = batch_size_proportion
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        self.rollout_buffer = buffers.RolloutBuffer(
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.multi_agent_ACP(
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            ortho_init=self.ortho_init,
            conv_params=self.conv_params,
            add_BN=self.add_BN,
            output_dim=self.output_dim,
            share_params=self.share_params,
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
            grid_cover: np.ndarray, p: np.ndarray, need_move: list, actions: Union[np.ndarray, None]):
        '''
        :param vehicle_states: shape = (batch_size, self.vehicle_num, 3), the third dimension is the origin node, the
        destination node and the remaining time to destination.
        :param node_weight: shape = (batch_size, number_of_node)
        :param grid_cover: shape = (batch_size, number_of_node)
        :param p: shape = (batch_size, number_of_node)
        :param need_move:
        :param actions: shape = (batch_size, )
        :return:

        the shape of values, log_probs, entropys = (batch_size, ).

        distributions, values are dict. values' structure is {vehicle_id1: [x], vehicle_id2: [x]...}. distributions'
        structure is {vehicle_id1: __main__.CategoricalDistribution object,
        vehicle_id2: __main__.CategoricalDistribution object...}

        '''
        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).long()
            state_features = self.transform_and_concat_all_features(
                vehicle_states=vehicle_states,
                node_weight=node_weight,
                grid_cover=grid_cover,
                p=p,
            )
            self.policy.train()
            values, log_probs, entropys = self.policy.forward(
                state_features=state_features,
                actions=actions,
            )
            return values, log_probs, entropys
        else:
            self.policy.eval()
            distributions = {}
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
                    distributions[vehicle_id], value, _ = self.policy.forward(
                        state_features=state_features,
                        actions=None,
                    )
                values[vehicle_id] = value.cpu().numpy()
            return distributions, values, None

    def collect_rollouts(self, grid_weight):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        need_test = False
        episode = 0
        done = False
        self.rollout_buffer.reset()

        while episode < self.buffer_size_episodes:

            vehicle_states, node_weight, grid_cover, p, need_move = self._last_obs
            distributions, values, _ = self.policy_model_predict(
                vehicle_states=vehicle_states,
                node_weight=node_weight,
                grid_cover=grid_cover,
                p=p,
                need_move=need_move,
                actions=None,
            )

            actions, log_probs, vehicle_states, node_weight, grid_cover, p, need_move, reward, done, \
            self.episode_time_cost = self.make_one_step_forward_for_env(
                env=self.env,
                distributions=distributions,
                episode_time_cost=self.episode_time_cost,
            )
            vehicle_states = vehicle_states.astype(np.float32).reshape((1,) + vehicle_states.shape)
            node_weight = node_weight.astype(np.float32).reshape((1,) + node_weight.shape)
            grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
            p = p.astype(np.float32).reshape((1,) + p.shape)

            if done:
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

                if episode == self.buffer_size_episodes - 1:
                    _, last_values, _ = self.policy_model_predict(
                        vehicle_states=vehicle_states,
                        node_weight=node_weight,
                        grid_cover=grid_cover,
                        p=p,
                        need_move=list(range(self.vehicle_num)),
                        actions=None,
                    )

                vehicle_states, node_weight, grid_cover, p, need_move = self.env.reset(grid_weight=grid_weight)
                vehicle_states = vehicle_states.astype(np.float32).reshape((1,) + vehicle_states.shape)
                node_weight = node_weight.astype(np.float32).reshape((1,) + node_weight.shape)
                grid_cover = grid_cover.astype(np.float32).reshape((1,) + grid_cover.shape)
                p = p.astype(np.float32).reshape((1,) + p.shape)
                self.episode += 1
                episode += 1
            new_obs = [vehicle_states, node_weight, grid_cover, p, need_move]

            self.rollout_buffer.add(
                obs=self._last_obs,
                actions=actions,
                reward=reward,
                done=self._last_done,
                values=values,
                log_probs=log_probs)
            self._last_obs = new_obs
            self._last_done = done

        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values, done=done)

        return need_test

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        for i in range(self.vehicle_num):
            for param_group in self.policy.ACP.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        self.rollout_buffer.concat()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size_proportion):

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)

                values, log_prob, entropy = self.policy_model_predict(
                    vehicle_states=rollout_data.vehicle_states,
                    node_weight=rollout_data.node_weight,
                    grid_cover=rollout_data.grid_cover,
                    p=rollout_data.p,
                    need_move=[],
                    actions=rollout_data.actions,
                )

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # flatten data
                values = torch.flatten(values)
                log_prob = torch.flatten(log_prob)
                entropy = torch.flatten(entropy)
                old_log_prob = torch.flatten(torch.as_tensor(
                    rollout_data.old_log_prob, dtype=torch.float32, device=self.device))
                advantages = torch.flatten(torch.as_tensor(
                    advantages, dtype=torch.float32, device=self.device))
                old_values = rollout_data.old_values
                returns = torch.flatten(torch.as_tensor(
                    rollout_data.returns, dtype=torch.float32, device=self.device))

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    old_values = torch.flatten(torch.as_tensor(old_values, dtype=torch.float32, device=self.device))
                    values_pred = old_values + torch.clamp(
                        values - old_values, - self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = - torch.mean(- log_prob)
                else:
                    entropy_loss = - torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimize(
                    loss=loss,
                    max_grad_norm=self.max_grad_norm,
                )
                approx_kl_divs.append(torch.mean(old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

    def test(self, test_episode_times: int, grid_weight: str):
        self.policy.eval()
        env = copy.deepcopy(self.env)
        new_obs = env.reset(grid_weight=grid_weight)
        episodes_total_scores = []
        episodes_grid_scores = []
        episodes_got_scores = []
        all_timesteps_socre = []
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
                    distributions, _, _ = self.policy.forward(
                        loc_features=loc_features,
                        weight_features=weight_features,
                    )
                _, _, new_obs, _, done, episode_time_cost = self.make_one_step_forward_for_env(
                    env=env,
                    distributions=distributions,
                    episode_time_cost=episode_time_cost,
                )
                timesteps_socre.append(1 - env.left_reward)
            episode_total_score = 1 - env.left_reward
            episodes_total_scores.append(episode_total_score)
            all_timesteps_socre.append(timesteps_socre)
            episodes_got_scores += [copy.deepcopy(env.episode_got_scores)]
            new_obs = env.reset(grid_weight=grid_weight)
        return np.mean(episodes_total_scores), episodes_grid_scores, episodes_got_scores, all_timesteps_socre

    # def save(self, train_link_weight_distribution, test_link_weight_distribution):
    #     self.best_state['best_episode'] = self.best_episode
    #     self.best_state['best_train_session'] = self.best_train_session
    #     self.best_state['random_policy_100_episodes_mean_total_score'] = \
    #         self.random_policy_100_episodes_mean_total_score
    #     self.best_state['the_best_100_episodes_mean_total_score'] = np.mean(self.the_best_100_episodes_total_scores)
    #     self.test_state['best_state'] = self.best_state
    #     with open('PPO_state_vehicle{}_env_{}_{}_ed_{}_trainD_{}_testD_{}.pickle'.format(
    #             self.vehicle_num, self.env.height, self.env.width, self.env.episode_duration,
    #             train_link_weight_distribution, test_link_weight_distribution
    #     ), 'wb') as file:
    #         pickle.dump(self.test_state, file)

    def learn(self, total_episodes: int, test_episode_times: int, grid_weight):

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
        while self.episode < total_episodes:
            need_test = self.collect_rollouts(grid_weight=grid_weight)
            self.train()
            print("""
            *************************************************************************
            *************************************************************************
            train successfully
            *************************************************************************
            *************************************************************************
            """)
            train_session += 1
            print('training successful in {}th training session'.format(train_session))

            # #------------------------------------------test--------------------------------------------------
            # test_session += 1
            # self.cur_state, episodes_grid_scores, episodes_got_scores, all_timesteps_socre = self.test(
            #     test_episode_times=test_episode_times,
            #     grid_weight=grid_weight,
            # )
            #
            # self.test_state[test_session] = {}
            # self.test_state[test_session]['test_100_episodes_mean_total_score'] = self.cur_state
            # self.test_state[test_session]['episodes_grid_scores'] = episodes_grid_scores
            # self.test_state[test_session]['episodes_got_scores'] = episodes_got_scores
            # self.test_state[test_session]['all_timesteps_socre'] = all_timesteps_socre
            #
            # if self.cur_state > self.best_state['test_100_episodes_mean_total_score']:
            #     self.best_state['test_100_episodes_mean_total_score'] = self.cur_state
            #     self.best_state['policy_params'] = self.policy.state_dict()
            #     self.best_state['test_session'] = test_session
            #     self.best_episode = self.episode
            #     self.best_train_session = train_session
            # print('''
            # **------------------------------------------------------------------------------------------**
            # **------------------------------------------------------------------------------------------**
            # {}th test:
            # now have been {}th episode, {}th training, current test 100_episodes_mean_total_score = {};
            # best test 100_episodes_mean_total_score = {}, the result of {}th episode, {}th training is best
            # **------------------------------------------------------------------------------------------**
            # **------------------------------------------------------------------------------------------**
            # '''.format(
            #     test_session, self.episode, train_session, self.cur_state,
            #     self.best_state['test_100_episodes_mean_total_score'], self.best_episode, self.best_train_session))
            # # self.save(
            # #     train_link_weight_distribution=train_link_weight_distribution,
            # #     test_link_weight_distribution=test_link_weight_distribution,
            # # )
            # # ------------------------------------------------------------------------------------------------

            # if self.last_100_episodes_mean_total_score <= self.the_best_last_100_episodes_mean_total_score / 2:
            #     break

        # self.save(
        #     train_link_weight_distribution=train_link_weight_distribution,
        #     test_link_weight_distribution=test_link_weight_distribution,
        # )

    def load_params(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
        ACP_params = state['best_state']['policy_params']
        self.policy.load_state_dict(ACP_params=ACP_params)