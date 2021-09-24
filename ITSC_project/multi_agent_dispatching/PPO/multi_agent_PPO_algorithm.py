import sys
import os
from typing import Optional, Union
import copy
import pickle

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from ITSC_project.multi_agent_dispatching.common_utils import policies, multi_agent_control
from ITSC_project.multi_agent_dispatching.PPO import buffers


class multi_agent_PPO(multi_agent_control.multi_agent):

    def __init__(
        self,
        env,
        reward_type,
        cooperative_weight,
        negative_constant_reward,
        vehicle_num: int,
        weight_shape: int,
        share_policy: bool,
        ortho_init: bool,
        conv_params: Union[list, tuple],
        add_BN: bool,
        output_dim: Union[list, tuple],
        share_params: bool,
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

        super(multi_agent_PPO, self).__init__(
            env=env,
            reward_type=reward_type,
            cooperative_weight=cooperative_weight,
            negative_constant_reward=negative_constant_reward,
            vehicle_num=vehicle_num,
            seed=seed,
            device=device,
        )

        self.learning_rate = learning_rate

        self.n_steps = n_steps
        self.weight_shape = weight_shape
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.share_policy = share_policy
        self.ortho_init = ortho_init
        self.conv_params = conv_params
        self.add_BN = add_BN
        self.output_dim = output_dim
        self.share_params = share_params
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
            weight_shape=self.weight_shape,
            share_policy=self.share_policy,
            ortho_init=self.ortho_init,
            conv_params=self.conv_params,
            add_BN=self.add_BN,
            output_dim=self.output_dim,
            share_params=self.share_params,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
        ).to(torch.float32).to(self.device).eval()

    def collect_rollouts(self, link_weight_distribution):
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
                loc_features = self._last_obs[0]
                weight_features = self._last_obs[1]
                distributions, values, _ = self.policy.forward(
                    loc_features=loc_features,
                    weight_features=weight_features,
                )
            values = values.cpu().numpy()

            for i, distribution in enumerate(distributions):
                print('vehicle id {} prob: '.format(i), distribution.all_probs())

            actions, log_probs, new_obs, reward, done, self.episode_time_cost = self.make_one_step_forward_for_env(
                env=self.env,
                distributions=distributions,
                episode_time_cost=self.episode_time_cost,
            )

            if done:
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
                new_obs = self.env.reset(link_weight_distribution=link_weight_distribution)
                number_of_episode_timestep = 0
            new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
            new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)

            self.num_timesteps += 1
            timestep += 1

            self.rollout_buffer.add(self._last_obs, actions, reward,
                                    self._last_done, values, log_probs)
            self._last_obs = new_obs
            self._last_done = done

        with torch.no_grad():
            # Compute value for the last timestep
            loc_features = self._last_obs[0]
            weight_features = self._last_obs[1]
            _, values, _ = self.policy.forward(
                loc_features=loc_features,
                weight_features=weight_features,
            )
        values = values.cpu().numpy()

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)

        return need_test

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.train()
        # Update optimizer learning rate
        for i in range(self.vehicle_num):
            for param_group in self.policy.ACP[i].optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)

                values, log_prob, entropy = self.policy.forward(
                    loc_features=rollout_data.loc,
                    weight_features=rollout_data.weight[:, np.newaxis],
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

    def test(self, test_episode_times: int, link_weight_distribution: str):
        self.policy.eval()
        env = copy.deepcopy(self.env)
        new_obs = env.reset(link_weight_distribution=link_weight_distribution)
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
            new_obs = env.reset(link_weight_distribution=link_weight_distribution)
        return np.mean(episodes_total_scores), episodes_grid_scores, episodes_got_scores, all_timesteps_socre

    def save(self, train_link_weight_distribution, test_link_weight_distribution):
        self.best_state['best_episode'] = self.best_episode
        self.best_state['best_train_session'] = self.best_train_session
        self.best_state['random_policy_100_episodes_mean_total_score'] = \
            self.random_policy_100_episodes_mean_total_score
        self.best_state['the_best_100_episodes_mean_total_score'] = np.mean(self.the_best_100_episodes_total_scores)
        self.test_state['best_state'] = self.best_state
        with open('PPO_state_vehicle{}_env_{}_{}_ed_{}_trainD_{}_testD_{}_seed{}.pickle'.format(
                self.vehicle_num, self.env.height, self.env.width, self.env.episode_duration,
                train_link_weight_distribution, test_link_weight_distribution, self.seed,
        ), 'wb') as file:
            pickle.dump(self.test_state, file)

    def learn(self, total_timesteps: int, test_episode_times: int,
              train_link_weight_distribution: str,
              test_link_weight_distribution: str):

        self.init_learn(train_link_weight_distribution=train_link_weight_distribution)
        self.random_policy_100_episodes_mean_total_score = 0
        self.cur_state = self.random_policy_100_episodes_mean_total_score
        self.best_state = {'test_100_episodes_mean_total_score': self.random_policy_100_episodes_mean_total_score,
                           'policy_params': self.policy.state_dict()}
        self.test_state = {}
        self.best_episode = 0

        train_session = 0
        test_session = 0
        self.best_train_session = train_session
        while self.num_timesteps < total_timesteps:
            need_test = self.collect_rollouts(link_weight_distribution=train_link_weight_distribution)
            self.train()
            train_session += 1
            print('training successful in {}th training session'.format(train_session))

            #------------------------------------------test--------------------------------------------------
            test_session += 1
            self.cur_state, episodes_grid_scores, episodes_got_scores, all_timesteps_socre = self.test(
                test_episode_times=test_episode_times,
                link_weight_distribution=test_link_weight_distribution,
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
            self.save(
                train_link_weight_distribution=train_link_weight_distribution,
                test_link_weight_distribution=test_link_weight_distribution,
            )
            # ------------------------------------------------------------------------------------------------

            if self.last_100_episodes_mean_total_score <= self.the_best_last_100_episodes_mean_total_score / 2:
                break

        self.save(
            train_link_weight_distribution=train_link_weight_distribution,
            test_link_weight_distribution=test_link_weight_distribution,
        )

    def load_params(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
        ACP_params = state['best_state']['policy_params']
        self.policy.load_state_dict(ACP_params=ACP_params)