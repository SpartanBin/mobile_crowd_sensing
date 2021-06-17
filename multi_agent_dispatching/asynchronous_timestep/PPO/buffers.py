from typing import Optional, NamedTuple
import copy

import numpy as np


class RolloutBufferSamples(NamedTuple):
    vehicle_states: np.ndarray
    node_weight: np.ndarray
    grid_cover: np.ndarray
    p: np.ndarray
    actions: np.ndarray
    old_values: np.ndarray
    old_log_prob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer():
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    Buffer is used to save all agents' transitions, so all agents share the same rewards and states.
    """

    def __init__(
        self,
        vehicle_num: int,
        weight_shape: int,
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        '''
        :param weight_shape: Observation weight shape
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
        :param gamma: Discount factor
        '''
        self.vehicle_num = vehicle_num
        self.weight_shape = weight_shape
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.vehicle_states = []
        self.node_weight = []
        self.grid_cover = []
        self.p = []
        self.actions = []
        self.dones = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.values = []
        self.log_probs = []

        for vehicle_id in range(self.vehicle_num):
            self.vehicle_states.append([])
            self.node_weight.append([])
            self.grid_cover.append([])
            self.p.append([])
            self.actions.append([])
            self.dones.append([])
            self.rewards.append([])
            self.advantages.append([])
            self.returns.append([])
            self.values.append([])
            self.log_probs.append([])

    def reset(self) -> None:

        for vehicle_id in range(self.vehicle_num):
            self.vehicle_states[vehicle_id] = []
            self.node_weight[vehicle_id] = []
            self.grid_cover[vehicle_id] = []
            self.p[vehicle_id] = []
            self.actions[vehicle_id] = []
            self.dones[vehicle_id] = []
            self.rewards[vehicle_id] = []
            self.advantages[vehicle_id] = []
            self.returns[vehicle_id] = []
            self.values[vehicle_id] = []
            self.log_probs[vehicle_id] = []

        self.vehicle_states_output = None
        self.node_weight_output = None
        self.grid_cover_output = None
        self.p_output = None
        self.actions_output = None
        self.values_output = None
        self.log_probs_output = None
        self.advantages_output = None
        self.returns_output = None

    def add(
        self, obs: list, actions, reward, done, values, log_probs,
    ) -> None:
        """
        :param obs: Observation, [0] is loc, [1] is weight
        :param actions: Action
        :param reward:
        :param done: End of episode signal.
        :param values: estimated value of the current state
            following the current policy.
        :param log_probs: log probability of the action
            following the current policy.
        """
        vehicle_states, node_weight, grid_cover, p, need_move = obs
        for vehicle_id in need_move:
            self.vehicle_states[vehicle_id].append(vehicle_states[:, [vehicle_id] + list(range(vehicle_id)) + list(
                range(vehicle_id + 1, self.vehicle_num)), :].copy())
            self.node_weight[vehicle_id].append(node_weight.copy())
            self.grid_cover[vehicle_id].append(grid_cover.copy())
            self.p[vehicle_id].append(p.copy())
            self.actions[vehicle_id].append(copy.deepcopy(actions[vehicle_id]))
            self.rewards[vehicle_id].append(copy.deepcopy(reward))
            self.values[vehicle_id].append(values[vehicle_id].copy())
            self.log_probs[vehicle_id].append(log_probs[vehicle_id].copy())
        if done:
            for vehicle_id in range(self.vehicle_num):
                episode_length = len(self.vehicle_states[vehicle_id]) - len(self.dones[vehicle_id])
                self.dones[vehicle_id] += [False] * (episode_length - 1) + [True]

    def compute_returns_and_advantage(self, last_values: np.ndarray, done: bool) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_values: shape = (1, self.vehicle_num)
        :param done:

        """
        last_values = last_values.copy()
        done = np.array([done] * self.vehicle_num)

        for vehicle_id in range(self.vehicle_num):
            last_gae_lam = 0
            buffer_size = len(self.vehicle_states[vehicle_id])

            self.vehicle_states[vehicle_id] = np.array(self.vehicle_states[vehicle_id]).squeeze()
            self.node_weight[vehicle_id] = np.array(self.node_weight[vehicle_id]).squeeze()
            self.grid_cover[vehicle_id] = np.array(self.grid_cover[vehicle_id]).squeeze()
            self.p[vehicle_id] = np.array(self.p[vehicle_id]).squeeze()
            self.actions[vehicle_id] = np.array(self.actions[vehicle_id]).squeeze()
            episode_length = len(self.vehicle_states[vehicle_id]) - len(self.dones[vehicle_id])
            self.dones[vehicle_id] += [False] * episode_length
            self.dones[vehicle_id] = np.array(self.dones[vehicle_id]).squeeze()
            self.rewards[vehicle_id] = np.array(self.rewards[vehicle_id]).squeeze()
            self.values[vehicle_id] = np.array(self.values[vehicle_id]).squeeze()
            self.log_probs[vehicle_id] = np.array(self.log_probs[vehicle_id]).squeeze()

            for step in reversed(range(buffer_size)):
                if step == buffer_size - 1:
                    next_non_terminal = 1.0 - done[vehicle_id]
                    next_values = last_values[vehicle_id]
                else:
                    next_non_terminal = 1.0 - self.dones[vehicle_id][step + 1]
                    next_values = self.values[vehicle_id][step + 1]
                delta = self.rewards[vehicle_id][step] + self.gamma * next_values * next_non_terminal - \
                        self.values[vehicle_id][step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[vehicle_id].append(last_gae_lam)
            self.advantages[vehicle_id] = np.array(self.advantages[vehicle_id]).squeeze()
            self.returns[vehicle_id] = self.advantages[vehicle_id] + self.values[vehicle_id]

    def copy_or_not(self, array: np.ndarray, copy: bool = True) -> np.ndarray:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return array.copy()
        return array

    def concat(self):
        self.vehicle_states_output = np.vstack(self.vehicle_states)
        self.node_weight_output = np.vstack(self.node_weight)
        self.grid_cover_output = np.vstack(self.grid_cover)
        self.p_output = np.vstack(self.p)
        self.actions_output = np.concatenate(self.actions, axis=0)
        self.values_output = np.concatenate(self.values, axis=0)
        self.log_probs_output = np.concatenate(self.log_probs, axis=0)
        self.advantages_output = np.concatenate(self.advantages, axis=0)
        self.returns_output = np.concatenate(self.returns, axis=0)

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.vehicle_states_output[batch_inds],
            self.node_weight_output[batch_inds],
            self.grid_cover_output[batch_inds],
            self.p_output[batch_inds],
            self.actions_output[batch_inds],
            self.values_output[batch_inds],
            self.log_probs_output[batch_inds],
            self.advantages_output[batch_inds],
            self.returns_output[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.copy_or_not, data)))

    def get(self, batch_size_proportion: Optional[int] = None):

        indices = np.random.permutation(len(self.vehicle_states_output))

        batch_size = int(len(self.vehicle_states_output) * batch_size_proportion + 1)

        start_idx = 0
        while start_idx < len(self.vehicle_states_output):
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size