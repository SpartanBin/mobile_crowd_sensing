import sys
import os
from typing import Optional, Union
import random
import copy
import pickle

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import DataLoader

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from grid_score_base_project.multi_agent_dispatching.MAGPPO import buffers, policies


class multi_agent_PPO():

    def __init__(
        self,
        env,
        vehicle_num: int,
        weight_shape: int,
        ortho_init: bool,
        in_channels: int,
        action_dim: int,
        learning_rate: Union[float, int] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
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
        ad_mat = self.env.experienced_travel_time
        edge_where = np.where(ad_mat < np.inf)
        self.edge_index = torch.from_numpy(np.vstack(edge_where))
        self.edge_weight = 1 / ad_mat[edge_where]
        self.edge_weight = torch.from_numpy(self.edge_weight)
        self.one_timestep_batch = torch.zeros(len(self.env.node), dtype=torch.long)

        self.vehicle_num = vehicle_num
        self.device = torch.device(device)

        self.learning_rate = learning_rate

        self.n_steps = n_steps
        self.weight_shape = weight_shape
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ortho_init = ortho_init
        self.in_channels = in_channels
        self.action_dim = action_dim

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        self.rollout_buffer = buffers.RolloutBuffer(
            buffer_size=self.n_steps,
            vehicle_num=self.vehicle_num,
            weight_shape=self.weight_shape,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.multi_agent_ACP(
            vehicle_num=self.vehicle_num,
            ortho_init=self.ortho_init,
            in_channels=self.in_channels,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
        ).to(self.device).eval()

    def change_env_output_feature_format(
            self,
            edge_loc_features: np.ndarray,
            the_same_features: np.ndarray,
            loc_ID: np.ndarray,
    ):
        edge_loc_features = torch.from_numpy(edge_loc_features).unsqueeze(-1)
        the_same_features = torch.from_numpy(the_same_features)
        loc_ID = loc_ID.reshape((1,) + loc_ID.shape)
        return edge_loc_features, the_same_features, loc_ID

    def make_one_step_forward_for_env(self, env, distributions, episode_time_cost):

        ac_probs_dict = {}
        ac_dict = {}
        actions = []

        for i in range(self.vehicle_num):
            self.select_action_time += 1
            distribution = distributions[i]
            action = distribution.get_actions().item()
            ac_dict[i] = action
            actions.append(copy.deepcopy(action))
            ac_probs_dict[i] = distribution.all_probs()[0]
        actions = torch.tensor(actions, dtype=torch.long)

        edge_loc_features, the_same_features, reward, done, episode_time_cost, loc_ID = env.step(
            ac_dict=ac_dict,
            episode_time_cost=episode_time_cost,
        )

        # # augment info
        # edge_loc_features = edge_loc_features * 100
        # the_same_features[:, 0] = the_same_features[:, 0] * 100

        edge_loc_features, the_same_features, loc_ID = self.change_env_output_feature_format(
            edge_loc_features, the_same_features, loc_ID)

        log_probs = torch.tensor([np.nan] * self.vehicle_num, dtype=torch.float32)
        ac = actions.view((1, -1)).to(self.device)
        for i in range(self.vehicle_num):
            distribution = distributions[i]
            log_probs[i] = distribution.log_prob(ac[:, i]).cpu()[0]

        return actions, log_probs, edge_loc_features, the_same_features, reward, done, episode_time_cost

    def collect_rollouts(self):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        need_test = False
        timestep = 0
        done = False
        self.rollout_buffer.reset()
        self.policy.eval()

        while timestep < self.n_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                edge_loc_features, the_same_features, loc_ID = self._last_obs

                # import pickle
                # with open('D:/projects/mobile_crowd_sensing/jupyter_notebook/kk', 'wb') as file:
                #     pickle.dump((
                #         self.edge_index, self.edge_weight, self.one_timestep_batch,
                #         edge_loc_features, the_same_features
                #     ), file)
                # exit()

                # edge_loc_features[:] = 0
                # the_same_features[:] = 0
                distributions, values, _ = self.policy.forward(
                    edge_index=self.edge_index,
                    edge_weight=self.edge_weight,
                    batch=self.one_timestep_batch,
                    edge_loc_features=edge_loc_features,
                    the_same_features=the_same_features,
                )

            # for distribution in distributions:
            #     print(distribution.all_probs())

            actions, log_probs, edge_loc_features, the_same_features, reward, done, self.episode_time_cost = \
                self.make_one_step_forward_for_env(
                    env=self.env,
                    distributions=distributions,
                    episode_time_cost=self.episode_time_cost,
                )

            if done:
                self.episode += 1
                got_reward = self.env.got_reward
                self.the_last_100_episodes_got_rewards.append(got_reward)
                self.the_best_100_episodes_got_rewards.append(got_reward)
                self.last_100_episodes_mean_got_reward = np.mean(self.the_last_100_episodes_got_rewards)
                if len(self.the_last_100_episodes_got_rewards) > 100:
                    if self.last_100_episodes_mean_got_reward > self.the_best_last_100_episodes_mean_got_reward:
                        self.the_best_last_100_episodes_mean_got_reward = self.last_100_episodes_mean_got_reward
                        need_test = True
                    self.the_last_100_episodes_got_rewards.pop(0)
                    self.the_best_100_episodes_got_rewards.sort()
                    self.the_best_100_episodes_got_rewards.pop(0)
                if self.episode % 100 == 0:
                    print('''
                    ******************************************************************************************************
                    in {}th episode, the number of vehicle is {} 
                    ------------------------------------------------------------------------------------------------------
                    episode_time_cost = {}, got_reward = {}, select_action_times = {}, 
                    random_policy_100_episodes_mean_got_reward = {}, 
                    the_best_100_episodes_mean_got_reward = {}, 
                    last_100_episodes_mean_got_reward = {}, 
                    the_best_last_100_episodes_mean_got_reward = {}
                    ******************************************************************************************************
                    '''.format(
                        self.episode, self.vehicle_num, self.episode_time_cost, got_reward,
                        self.select_action_time, self.random_policy_100_episodes_mean_got_reward,
                        np.mean(self.the_best_100_episodes_got_rewards),
                        self.last_100_episodes_mean_got_reward,
                        self.the_best_last_100_episodes_mean_got_reward,
                    ))
                self.episode_time_cost = 0
                self.select_action_time = 0
                edge_loc_features, the_same_features, loc_ID = self.env.reset()
                edge_loc_features, the_same_features, loc_ID = self.change_env_output_feature_format(
                    edge_loc_features, the_same_features, loc_ID)
            new_obs = [edge_loc_features, the_same_features, loc_ID]

            self.num_timesteps += 1
            timestep += 1
            self.rollout_buffer.add(self._last_obs, self.edge_index, self.edge_weight, actions, reward,
                                    self._last_done, values, log_probs)
            self._last_obs = new_obs
            self._last_done = done

        with torch.no_grad():
            # Compute value for the last timestep
            edge_loc_features, the_same_features, loc_ID = self._last_obs
            _, values, _ = self.policy.forward(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                batch=self.one_timestep_batch,
                edge_loc_features=edge_loc_features,
                the_same_features=the_same_features,
            )

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)

        return need_test

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.train()
        # Update optimizer learning rate
        for i in range(self.vehicle_num):
            for param_group in self.policy.ACP.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        DL = DataLoader(self.rollout_buffer.thgm_data, batch_size=self.batch_size, shuffle=True)

        # train for n_epochs epochs
        mean_policy_loss = []
        mean_entropy_loss = []
        mean_value_loss = []
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for mini_batch_data in DL:

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)

                state_features = mini_batch_data.x
                actions = mini_batch_data.actions
                values, log_prob, entropy = self.policy.forward(
                    edge_index=mini_batch_data.edge_index,
                    edge_weight=mini_batch_data.edge_attr,
                    batch=mini_batch_data.batch,
                    state_features=state_features,
                    actions=actions,
                )

                # pos index
                pos_index = mini_batch_data.pos_i.flatten()

                # Normalize advantage
                advantages = self.rollout_buffer.advantages_[pos_index].to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # flatten data
                values = torch.flatten(values)
                log_prob = torch.flatten(log_prob)
                entropy = torch.flatten(entropy)
                old_log_prob = self.rollout_buffer.log_probs_[pos_index].to(self.device)
                old_values = self.rollout_buffer.values_[pos_index]
                returns = self.rollout_buffer.returns_[pos_index].to(self.device)

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
                    old_values = old_values.to(self.device)
                    values_pred = old_values + torch.clamp(
                        values - old_values, - self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns.flatten(), values_pred.flatten())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = - torch.mean(- log_prob)
                else:
                    entropy_loss = - torch.mean(entropy)

                mean_policy_loss.append(policy_loss.detach().cpu().numpy().copy())
                mean_entropy_loss.append(entropy_loss.detach().cpu().numpy().copy())
                mean_value_loss.append(value_loss.detach().cpu().numpy().copy())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimize(
                    loss=loss,
                    max_grad_norm=self.max_grad_norm,
                )
                approx_kl_divs.append(torch.mean(old_log_prob - log_prob).detach().cpu().numpy().copy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        return np.mean(mean_policy_loss), np.mean(mean_entropy_loss), np.mean(mean_value_loss)

    def test(self, test_episode_times: int):
        all_state_features = []
        self.policy.eval()
        env = copy.deepcopy(self.env)
        edge_loc_features, the_same_features, loc_ID = env.reset()
        edge_loc_features, the_same_features, loc_ID = self.change_env_output_feature_format(
            edge_loc_features, the_same_features, loc_ID)
        new_obs = [edge_loc_features, the_same_features, loc_ID]
        got_rewards = np.zeros(test_episode_times, dtype=np.float32)
        for episode in range(test_episode_times):
            done = False
            episode_time_cost = 0
            while not done:
                edge_loc_features, the_same_features, loc_ID = new_obs
                all_state_features.append(copy.deepcopy(new_obs))
                with torch.no_grad():
                    distributions, _, _ = self.policy.forward(
                        edge_index=self.edge_index,
                        edge_weight=self.edge_weight,
                        batch=self.one_timestep_batch,
                        edge_loc_features=edge_loc_features,
                        the_same_features=the_same_features,
                        actions=None,
                    )
                _, _, edge_loc_features, the_same_features, _, done, episode_time_cost = \
                    self.make_one_step_forward_for_env(
                        env=env,
                        distributions=distributions,
                        episode_time_cost=episode_time_cost,
                    )
                new_obs = [edge_loc_features, the_same_features, loc_ID]
            got_rewards[episode] = env.got_reward
            edge_loc_features, the_same_features, loc_ID = env.reset()
            edge_loc_features, the_same_features, loc_ID = self.change_env_output_feature_format(
                edge_loc_features, the_same_features, loc_ID)
            new_obs = [edge_loc_features, the_same_features, loc_ID]
        return got_rewards.mean(), all_state_features

    def save(self):
        self.best_state['best_episode'] = self.best_episode
        self.best_state['best_train_session'] = self.best_train_session
        self.best_state['random_policy_100_episodes_mean_got_reward'] = \
            self.random_policy_100_episodes_mean_got_reward
        self.best_state['the_best_100_episodes_mean_got_reward'] = np.mean(self.the_best_100_episodes_got_rewards)
        self.test_state['best_state'] = self.best_state
        with open('PPO_state_vehicle{}_ed_{}_SAGE_GA_Extractor.pickle'.format(
                self.vehicle_num, self.env.episode_duration,
        ), 'wb') as file:
            pickle.dump(self.test_state, file)

    def learn(self, total_timesteps: int, test_episode_times: int):

        self.num_timesteps = 0
        self.episode_time_cost = 0
        self.episode = 0
        self.the_last_100_episodes_got_rewards = []
        self.last_100_episodes_mean_got_reward = - float('inf')
        self.the_best_last_100_episodes_mean_got_reward = - float('inf')
        self.the_best_100_episodes_got_rewards = []
        self.random_policy_100_episodes_mean_got_reward = None
        edge_loc_features, the_same_features, loc_ID = self.env.reset()
        edge_loc_features, the_same_features, loc_ID = self.change_env_output_feature_format(
            edge_loc_features, the_same_features, loc_ID)
        self._last_obs = [edge_loc_features, the_same_features, loc_ID]
        self._last_done = False
        self.select_action_time = 0
        self.random_policy_100_episodes_mean_got_reward = - float('inf')
        self.cur_state = self.random_policy_100_episodes_mean_got_reward
        self.best_state = {'test_100_episodes_mean_got_reward': self.random_policy_100_episodes_mean_got_reward,
                           'policy_params': self.policy.state_dict()}
        self.test_state = {}
        self.best_episode = 0

        train_session = 0
        test_session = 0
        self.best_train_session = train_session
        last_10_train_sessions_policy_loss = []
        last_10_train_sessions_entropy_loss = []
        last_10_train_sessions_value_loss = []
        the_best_last_10_train_sessions_mean_policy_loss = - float('inf')
        the_best_last_10_train_sessions_mean_entropy_loss = - float('inf')
        the_best_last_10_train_sessions_mean_value_loss = - float('inf')
        while self.num_timesteps < total_timesteps:
            need_test = self.collect_rollouts()
            mean_policy_loss, mean_entropy_loss, mean_value_loss = self.train()
            last_10_train_sessions_policy_loss.append(mean_policy_loss)
            last_10_train_sessions_entropy_loss.append(mean_entropy_loss)
            last_10_train_sessions_value_loss.append(mean_value_loss)
            if len(last_10_train_sessions_policy_loss) > 10:
                last_10_train_sessions_policy_loss.pop(0)
                last_10_train_sessions_entropy_loss.pop(0)
                last_10_train_sessions_value_loss.pop(0)
            last_10_train_sessions_mean_policy_loss = np.mean(last_10_train_sessions_policy_loss)
            last_10_train_sessions_mean_entropy_loss = np.mean(last_10_train_sessions_entropy_loss)
            last_10_train_sessions_mean_value_loss = np.mean(last_10_train_sessions_value_loss)
            if abs(last_10_train_sessions_mean_policy_loss) < abs(the_best_last_10_train_sessions_mean_policy_loss):
                the_best_last_10_train_sessions_mean_policy_loss = last_10_train_sessions_mean_policy_loss
            if abs(last_10_train_sessions_mean_entropy_loss) < abs(the_best_last_10_train_sessions_mean_entropy_loss):
                the_best_last_10_train_sessions_mean_entropy_loss = last_10_train_sessions_mean_entropy_loss
            if abs(last_10_train_sessions_mean_value_loss) < abs(the_best_last_10_train_sessions_mean_value_loss):
                the_best_last_10_train_sessions_mean_value_loss = last_10_train_sessions_mean_value_loss
            train_session += 1
            print(
                '''
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                training successful in {}th training session, 
                the last 10 train sessions mean policy loss is {}, 
                the last 10 train sessions mean entropy loss is {}, 
                the last 10 train sessions mean value loss is {}, 
                the best last 10 train sessions mean policy loss is {}, 
                the best last 10 train sessions mean entropy loss is {}, 
                the best last 10 train sessions mean value loss is {}, 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                '''.format(
                    train_session, last_10_train_sessions_mean_policy_loss,
                    last_10_train_sessions_mean_entropy_loss, last_10_train_sessions_mean_value_loss,
                    the_best_last_10_train_sessions_mean_policy_loss,
                    the_best_last_10_train_sessions_mean_entropy_loss,
                    the_best_last_10_train_sessions_mean_value_loss,
                )
            )

            if need_test:
                #------------------------------------------test--------------------------------------------------
                test_session += 1
                self.cur_state, all_state_features = self.test(
                    test_episode_times=test_episode_times,
                )

                self.test_state[test_session] = {}
                self.test_state[test_session]['test_100_episodes_mean_got_reward'] = self.cur_state

                if self.cur_state > self.best_state['test_100_episodes_mean_got_reward']:
                    self.best_state['test_100_episodes_mean_got_reward'] = self.cur_state
                    self.best_state['policy_params'] = self.policy.state_dict()
                    self.best_state['test_session'] = test_session
                    self.best_state['all_state_features'] = all_state_features
                    self.best_episode = self.episode
                    self.best_train_session = train_session
                print('''
                **------------------------------------------------------------------------------------------**
                **------------------------------------------------------------------------------------------**
                {}th test: 
                now have been {}th episode, {}th training, current test 100_episodes_mean_total_score = {}; 
                best test 100_episodes_mean_total_score = {}, the result of {}th episode, {}th training is best
                **------------------------------------------------------------------------------------------**
                **------------------------------------------------------------------------------------------**
                '''.format(
                    test_session, self.episode, train_session, self.cur_state,
                    self.best_state['test_100_episodes_mean_got_reward'], self.best_episode, self.best_train_session))
                self.save()
                # ------------------------------------------------------------------------------------------------

                # if self.last_100_episodes_mean_got_reward <= self.the_best_last_100_episodes_mean_got_reward / 2:
                #     break

            self.save()

    def load_params(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
        ACP_params = state['best_state']['policy_params']
        self.policy.load_state_dict(ACP_params=ACP_params)