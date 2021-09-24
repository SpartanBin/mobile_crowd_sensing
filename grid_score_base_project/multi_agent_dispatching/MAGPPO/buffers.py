import numpy as np
import torch
from torch_geometric.data import Data


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
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # reset
        self.thgm_data = [None] * (self.buffer_size * self.vehicle_num)
        self.rewards = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False
        self.state_features_ = None
        self.actions_ = None
        self.values_ = None
        self.log_probs_ = None
        self.advantages_ = None
        self.returns_ = None

    def reset(self) -> None:

        self.thgm_data = [None] * (self.buffer_size * self.vehicle_num)
        self.rewards = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.vehicle_num), dtype=torch.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

        self.state_features_ = None
        self.actions_ = None
        self.values_ = None
        self.log_probs_ = None
        self.advantages_ = None
        self.returns_ = None

    def add(
        self, obs: list, edge_index: torch.Tensor, edge_weight: torch.Tensor, actions: torch.Tensor,
            reward: torch.Tensor, done: bool, value: torch.Tensor, log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param edge_index:
        :param edge_weight:
        :param actions: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        edge_loc_features, the_same_features, loc_ID = obs
        for v_id in range(self.vehicle_num):
            x = torch.cat((edge_loc_features[v_id], the_same_features), dim=1)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                actions=torch.tensor([actions[v_id]]),
                pos_i=torch.tensor([self.pos * self.vehicle_num + v_id]),
            )
            self.thgm_data[self.pos * self.vehicle_num + v_id] = data

        self.rewards[self.pos] = torch.tensor(reward, dtype=torch.float32)
        self.dones[self.pos] = done
        self.values[self.pos] = value.clone()
        self.log_probs[self.pos] = log_prob.clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: bool) -> None:
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
        done = torch.tensor([done] * self.vehicle_num, dtype=torch.float32)
        last_values = last_values.to('cpu')

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

        self.values_ = self.values.flatten()
        self.log_probs_ = self.log_probs.flatten()
        self.advantages_ = self.advantages.flatten()
        self.returns_ = self.returns.flatten()