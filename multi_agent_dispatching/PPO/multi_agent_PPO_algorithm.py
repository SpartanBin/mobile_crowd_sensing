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

from multi_agent_dispatching.common_utils import policies, multi_agent_control
from multi_agent_dispatching.PPO import buffers


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
        ).to(self.device).eval()

    def collect_rollouts(self):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        """
        assert self._last_obs is not None, "No previous observation was provided"
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

            actions, log_probs, new_obs, reward, done, self.episode_time_cost = self.make_one_step_forward_for_env(
                env=self.env,
                distributions=distributions,
                episode_time_cost=self.episode_time_cost,
            )

            if done:
                self.the_last_100_episodes_time_cost.append(self.episode_time_cost)
                self.the_shortest_100_episodes_time_cost.append(self.episode_time_cost)
                last_100_episodes_mean_time_cost = np.mean(self.the_last_100_episodes_time_cost)
                if len(self.the_last_100_episodes_time_cost) > 100:
                    if last_100_episodes_mean_time_cost < self.the_best_last_100_episodes_mean_time_cost:
                        self.the_best_last_100_episodes_mean_time_cost = last_100_episodes_mean_time_cost
                    if self.the_first_100_episodes_mean_time_cost is None:
                        self.the_first_100_episodes_mean_time_cost = last_100_episodes_mean_time_cost
                    self.the_last_100_episodes_time_cost.pop(0)
                    self.the_shortest_100_episodes_time_cost.sort()
                    self.the_shortest_100_episodes_time_cost.pop()
                print('''
                ******************************************************************************************************
                in this episode, the number of vehicle is {}, reward_type is '{}', 
                cooperative_weight is {}, negative_constant_reward is {}, 
                ------------------------------------------------------------------------------------------------------
                all reward = {}, time cost = {}, reselect_action_times = {}, 
                the_shortest_100_episodes_mean_time_cost = {}, 
                the_first_100_episodes_mean_time_cost = {}, 
                the_last_100_episodes_mean_time_cost = {}, 
                the_best_last_100_episodes_mean_time_cost = {}
                ******************************************************************************************************
                '''.format(
                    self.vehicle_num, self.reward_type, self.cooperative_weight, self.negative_constant_reward,
                    1 - self.env.left_reward, self.episode_time_cost, self.select_action_time,
                    np.mean(self.the_shortest_100_episodes_time_cost),
                    self.the_first_100_episodes_mean_time_cost,
                    last_100_episodes_mean_time_cost,
                    self.the_best_last_100_episodes_mean_time_cost
                ))
                self.episode_time_cost = 0
                new_obs = self.env.reset()
                self.select_action_time = 0
            new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
            new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)

            self.num_timesteps += 1
            timestep += 1

            self.rollout_buffer.add(self._last_obs, actions, reward, self._last_done, values, log_probs)
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

    def test(self, test_episode_times: int):
        self.policy.eval()
        env = copy.deepcopy(self.env)
        new_obs = env.reset()
        episode_time_costs = []
        for _ in range(test_episode_times):
            done = False
            episode_time_cost = 0
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
            episode_time_costs.append(episode_time_cost)
            new_obs = env.reset()
        return np.mean(episode_time_costs)

    def learn(self, total_timesteps: int, test_every_train_sessions: int,
              test_episode_times: int, lowest_train_time_cost_to_test: Union[float, int]):

        self.init_learn()
        self.cur_state = float('inf')
        self.best_state = {'episode_time_cost': float('inf'), 'policy_params': self.policy.state_dict()}

        train_session = 0
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            if self.the_best_last_100_episodes_mean_time_cost <= lowest_train_time_cost_to_test:
                self.cur_state = self.test(test_episode_times=test_episode_times)
                print('''
                *********************************************************************************
                low train time cost trigger this test: 
                current test episode_time_cost = {}, 
                best test episode_time_cost = {}
                *********************************************************************************
                '''.format(self.cur_state, self.best_state['episode_time_cost']))
                if self.cur_state < self.best_state['episode_time_cost']:
                    self.best_state['episode_time_cost'] = self.cur_state
                    self.best_state['policy_params'] = self.policy.state_dict()
            self.train()
            train_session += 1
            print('training successful in {}th training session'.format(train_session))
            if train_session % test_every_train_sessions == 0 and self.num_timesteps >= (total_timesteps / 5):
                self.cur_state = self.test(test_episode_times=test_episode_times)
                print('''
                *********************************************************************************
                training session trigger this test: 
                current test episode_time_cost = {}, 
                best test episode_time_cost = {}
                *********************************************************************************
                '''.format(self.cur_state, self.best_state['episode_time_cost']))
                if self.cur_state < self.best_state['episode_time_cost']:
                    self.best_state['episode_time_cost'] = self.cur_state
                    self.best_state['policy_params'] = self.policy.state_dict()
                elif self.best_state['episode_time_cost'] <= self.cur_state / 10:
                    with open('PPO_AC_params.pickle', 'wb') as file:
                        pickle.dump(self.best_state['policy_params'], file)
                    break