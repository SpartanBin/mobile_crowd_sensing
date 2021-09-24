from typing import Optional, NamedTuple

import numpy as np


class RolloutBufferSamples(NamedTuple):
    loc: np.ndarray
    weight: np.ndarray
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
        buffer_size: int,
        vehicle_num: int,
        weight_shape: int,
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        '''
        :param buffer_size: Max number of element in the buffer
        :param weight_shape: Observation weight shape
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
        :param gamma: Discount factor
        '''
        self.buffer_size = buffer_size
        self.vehicle_num = vehicle_num
        self.weight_shape = weight_shape
        self.pos = 0
        self.full = False
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.loc, self.weight, self.actions, self.rewards, self.advantages = None, None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.loc = np.zeros((self.buffer_size, self.vehicle_num * 2), dtype=np.float32)
        self.weight = np.zeros((self.buffer_size, self.weight_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.vehicle_num), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self, obs: list, actions: np.ndarray, reward: np.ndarray,
            done: bool, value: np.ndarray, log_prob: np.ndarray,
    ) -> None:
        """
        :param obs: Observation, [0] is loc, [1] is weight
        :param actions: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        self.loc[self.pos] = obs[0].copy()
        self.weight[self.pos] = obs[1].copy()
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = reward.copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.copy()
        self.log_probs[self.pos] = log_prob.copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

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
        # convert to numpy
        last_values = last_values.copy()
        done = np.array([done] * self.vehicle_num)

        last_gae_lam = np.zeros(self.vehicle_num)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

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

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.loc[batch_inds],
            self.weight[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.copy_or_not, data)))

    def get(self, batch_size: Optional[int] = None):
        assert self.full, 'Must fill the container if you want to sample from container'
        indices = np.random.permutation(self.buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size