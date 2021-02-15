import sys
import os
from typing import Optional, Union, Tuple
import random

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

from multi_agent_dispatching.PPO import buffers, policies


class multi_agent_PPO():

    def __init__(
        self,
        env,
        vehicle_num: int,
        loc_dim: int,
        weight_shape: Tuple[int, int],
        share_policy: bool,
        ortho_init: bool,
        loc_feature_dim: Union[list, tuple],
        weight_feature_params: Union[list, tuple],
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
        self.learning_rate = learning_rate

        self.env = env

        self.n_steps = n_steps
        self.vehicle_num = vehicle_num
        self.loc_dim = loc_dim
        self.weight_shape = weight_shape
        self.device = torch.device(device)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.share_policy = share_policy
        self.ortho_init = ortho_init
        self.loc_feature_dim = loc_feature_dim
        self.weight_feature_params = weight_feature_params
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
            loc_dim=self.loc_dim,
            weight_shape=self.weight_shape,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.multi_agent_ACP(
            vehicle_num=self.vehicle_num,
            loc_dim=self.loc_dim,
            weight_shape=self.weight_shape,
            share_policy=self.share_policy,
            ortho_init=self.ortho_init,
            loc_feature_dim=self.loc_feature_dim,
            weight_feature_params=self.weight_feature_params,
            output_dim=self.output_dim,
            share_params=self.share_params,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
        ).to(self.device)

    def make_one_step_forward_for_env(self, env, distributions):

        reselect_agent = np.array(range(self.vehicle_num))
        ac_dict = {}
        actions = np.array([np.nan] * self.vehicle_num)
        log_probs = np.array([np.nan] * self.vehicle_num)
        new_obs, rewards, done = None, None, None

        while len(reselect_agent) > 0:

            for i in reselect_agent:
                distribution = distributions[i]
                action = distribution.get_actions()
                log_prob = distribution.log_prob(action)
                action = action.cpu().numpy()[0]
                ac_dict[i] = action
                actions[i] = action
                log_probs[i] = log_prob.cpu().numpy()[0]

            env_returns = env.step(ac_dict)
            if type(env_returns) == np.ndarray:
                reselect_agent = env_returns
            else:
                new_obs, rewards, done = env_returns
                reselect_agent = np.array([])

        return actions, log_probs, new_obs, rewards, done

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

        while timestep < self.n_steps:

            with torch.no_grad():
                # Convert to pytorch tensor
                loc_features = torch.as_tensor(self._last_obs[0]).to(self.device)
                weight_features = torch.as_tensor(self._last_obs[1]).to(self.device)
                distributions, values, _ = self.policy.forward(
                    loc_features=loc_features,
                    weight_features=weight_features,
                )

            actions, log_probs, new_obs, reward, done = self.make_one_step_forward_for_env(
                env=self.env,
                distributions=distributions,
            )

            if done:
                print(1 - self.env.left_reward)
                new_obs = self.env.reset()
            new_obs = list(new_obs)
            new_obs[0] = new_obs[0].astype(np.float32).reshape((1, -1))
            new_obs[1] = new_obs[1].astype(np.float32).reshape((1, 1,) + new_obs[1].shape)

            self.num_timesteps += 1
            timestep += 1

            self.rollout_buffer.add(self._last_obs, actions, reward, self._last_done, values, log_probs)
            self._last_obs = new_obs
            self._last_done = done

        with torch.no_grad():
            # Compute value for the last timestep
            loc_features = torch.as_tensor(self._last_obs[0]).to(self.device)
            weight_features = torch.as_tensor(self._last_obs[1]).to(self.device)
            _, values, _ = self.policy.forward(
                loc_features=loc_features,
                weight_features=weight_features,
            )

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        for i in range(self.vehicle_num):
            for param_group in self.policy.ACP[i].optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long()

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)

                values, log_prob, entropy = self.policy.forward(
                    loc_features=rollout_data.loc,
                    weight_features=rollout_data.weight.unsqueeze(dim=1),
                    actions=actions,
                )

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # flatten data
                values = torch.flatten(values)
                log_prob = torch.flatten(log_prob)
                entropy = torch.flatten(entropy)
                old_log_prob = torch.flatten(rollout_data.old_log_prob)
                advantages = torch.flatten(advantages)
                old_values = torch.flatten(rollout_data.old_values)
                returns = torch.flatten(rollout_data.returns)

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

    def learn(self, total_timesteps: int):

        self.num_timesteps = 0
        self._last_obs = list(self.env.reset())
        self._last_obs[0] = self._last_obs[0].astype(np.float32).reshape((1, -1))
        self._last_obs[1] = self._last_obs[1].astype(np.float32).reshape(
            (1, 1,) + self._last_obs[1].shape)
        self._last_done = False

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            self.train()
            print('training successful')

    def predict(self, observation: Tuple[np.ndarray, np.ndarray]):
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        loc_features = torch.as_tensor(observation[0].astype(np.float32).reshape((1, -1))).to(self.device)
        weight_features = torch.as_tensor(observation[1].astype(np.float32).reshape(
            (1, 1,) + self._last_obs[1].shape)).to(self.device)
        with torch.no_grad():
            distributions, values, _ = self.policy.forward(
                loc_features=loc_features,
                weight_features=weight_features,
            )
        return distributions