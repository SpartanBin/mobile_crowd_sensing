"""Policies: abstract base class and concrete implementations."""

import copy
from functools import partial
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class MlpExtractor(nn.Module):

    def __init__(self, loc_feature_dim: Union[tuple, list], weight_feature_params: Union[tuple, list],
                 output_dim: Union[tuple, list], share_params: bool):
        '''
        :param loc_feature_dim: type of item in iteration must be int object
        :param weight_feature_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(MlpExtractor, self).__init__()

        policy_loc_net = []
        value_loc_net = []
        for i in range(len(loc_feature_dim)):
            if i > 0:
                policy_loc_net.append(nn.Linear(loc_feature_dim[i - 1], loc_feature_dim[i]))
                policy_loc_net.append(nn.Tanh())
                value_loc_net.append(nn.Linear(loc_feature_dim[i - 1], loc_feature_dim[i]))
                value_loc_net.append(nn.Tanh())
        self.policy_loc_net = nn.Sequential(*policy_loc_net)
        self.value_loc_net = nn.Sequential(*value_loc_net)

        policy_weight_net = []
        value_weight_net = []
        for layer_params in weight_feature_params:
            policy_weight_net.append(nn.Conv2d(**layer_params))
            policy_weight_net.append(nn.BatchNorm2d(layer_params['out_channels']))
            policy_weight_net.append(nn.ELU())
            value_weight_net.append(nn.Conv2d(**layer_params))
            value_weight_net.append(nn.BatchNorm2d(layer_params['out_channels']))
            value_weight_net.append(nn.ELU())
        self.policy_weight_net = nn.Sequential(*policy_weight_net)
        self.value_weight_net = nn.Sequential(*value_weight_net)

        policy_output_net = []
        value_output_net = []
        for i in range(len(output_dim)):
            if i > 0:
                policy_output_net.append(nn.Linear(output_dim[i - 1], output_dim[i]))
                policy_output_net.append(nn.Tanh())
                value_output_net.append(nn.Linear(output_dim[i - 1], output_dim[i]))
                value_output_net.append(nn.Tanh())
        self.policy_output_net = nn.Sequential(*policy_output_net)
        self.value_output_net = nn.Sequential(*value_output_net)

        self.share_params = share_params
        if self.share_params:
            del self.value_loc_net
            del self.value_weight_net
            del self.value_output_net

    def forward(self, loc_features: torch.Tensor, weight_features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        policy_loc_output = self.policy_loc_net(loc_features)
        policy_weight_output = self.policy_weight_net(weight_features)
        policy_output = torch.cat(
            (policy_loc_output, policy_weight_output.view((policy_weight_output.size()[0], -1))), dim=1)
        policy_output = self.policy_output_net(policy_output)
        value_output = None
        if not self.share_params:
            value_loc_output = self.value_loc_net(loc_features)
            value_weight_output = self.value_weight_net(weight_features)
            value_output = torch.cat(
                (value_loc_output, value_weight_output.view((value_weight_output.size()[0], -1))), dim=1)
            value_output = self.policy_output_net(value_output)
        return policy_output, value_output


class CategoricalDistribution():
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :return:
        """
        return self.sample()

    def actions_from_params(self, action_logits: torch.Tensor) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions()

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        ortho_init: bool,
        loc_feature_dim: Union[tuple, list],
        weight_feature_params: Union[tuple, list],
        output_dim: Union[tuple, list],
        share_params: bool,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param ortho_init:
        :param loc_feature_dim: type of item in iteration must be int object,
        official param = (input_feature_dim, 64, 64)
        :param weight_feature_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(ActorCriticPolicy, self).__init__()

        self.ortho_init = ortho_init

        self.mlp_extractor = MlpExtractor(
            loc_feature_dim=loc_feature_dim,
            weight_feature_params=weight_feature_params,
            output_dim=output_dim,
            share_params=share_params
        )
        self.value_net = nn.Linear(output_dim[-1], 1)
        # Action distribution
        self.action_distribution = CategoricalDistribution(action_dim=action_dim)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=output_dim[-1])

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def output_action_distribution(self, latent_pi: torch.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_distribution.proba_distribution(action_logits=mean_actions)

    def forward(
            self,
            loc_features: torch.Tensor,
            weight_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param loc_features:
        :param weight_features:
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.mlp_extractor(loc_features, weight_features)
        distribution = self.output_action_distribution(latent_pi)
        if latent_vf is None:
            latent_vf = latent_pi
        value = self.value_net(latent_vf)
        if actions is None:
            # actions = distribution.get_actions(deterministic=deterministic)
            # log_prob = distribution.log_prob(actions)
            return distribution, value, None
        else:
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return value, log_prob, entropy


class multi_agent_ACP(nn.Module):

    def __init__(
            self,
            vehicle_num: int,
            loc_dim: int,
            weight_shape: tuple,
            share_policy: bool,
            ortho_init: bool,
            loc_feature_dim: Union[tuple, list],
            weight_feature_params: Union[tuple, list],
            output_dim: Union[tuple, list],
            share_params: bool,
            action_dim: int,
            learning_rate: Union[int, float]):
        super(multi_agent_ACP, self).__init__()
        '''
        :param loc_feature_dim: without input dim
        :param weight_feature_params: with input layer params
        :param output_dim: without input dim
        :return: 
        '''

        self.vehicle_num = vehicle_num
        self.loc_dim = loc_dim
        self.weight_shape = weight_shape
        self.share_policy = share_policy

        loc_feature_dim = ((vehicle_num + 1) * loc_dim,) + loc_feature_dim
        conv_output_shape = copy.deepcopy(weight_shape)
        for params in weight_feature_params:
            if 'padding' not in params:
                params['padding'] = (0, 0)
            if 'dilation' not in params:
                params['dilation'] = (1, 1)
            if 'stride' not in params:
                params['stride'] = (1, 1)
            conv_output_shape = self.cal_conv_output_shape(
                input_shape=conv_output_shape,
                kernel_size=params['kernel_size'],
                padding=params['padding'],
                dilation=params['dilation'],
                stride=params['stride'],
            )
        output_dim = (loc_feature_dim[-1] + conv_output_shape[0] * conv_output_shape[1] * 1 *
                      params['out_channels'],) + output_dim

        self.ACP = {}
        for i in range(vehicle_num):
            if i == 0:
                self.ACP[i] = ActorCriticPolicy(
                    ortho_init=ortho_init,
                    loc_feature_dim=loc_feature_dim,
                    weight_feature_params=weight_feature_params,
                    output_dim=output_dim,
                    share_params=share_params,
                    action_dim=action_dim,
                    learning_rate=learning_rate,
                )
            else:
                if share_policy:
                    self.ACP[i] = self.ACP[0]
                else:
                    self.ACP[i] = ActorCriticPolicy(
                        ortho_init=ortho_init,
                        loc_feature_dim=loc_feature_dim,
                        weight_feature_params=weight_feature_params,
                        output_dim=output_dim,
                        share_params=share_params,
                        action_dim=action_dim,
                        learning_rate=learning_rate,
                    )

    @staticmethod
    def cal_conv_output_shape(input_shape, kernel_size, padding=(0, 0), dilation=(1, 1), stride=(1, 1)):
        row = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        col = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        return (row, col)

    def optimize(self, loss: torch.Tensor, max_grad_norm):

        if self.share_policy:
            self.ACP[0].optimizer.zero_grad()
        else:
            for i in range(self.vehicle_num):
                self.ACP[i].optimizer.zero_grad()
        loss.backward()
        if self.share_policy:
            torch.nn.utils.clip_grad_norm_(self.ACP[0].parameters(), max_grad_norm)
            self.ACP[0].optimizer.step()
        else:
            for i in range(self.vehicle_num):
                torch.nn.utils.clip_grad_norm_(self.ACP[i].parameters(), max_grad_norm)
                self.ACP[i].optimizer.step()

    def forward(
            self,
            loc_features: torch.Tensor,
            weight_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param loc_features:
        :param weight_features:
        :return: action, value and log probability of the action
        """
        values = []
        if actions is None:
            distributions = []
            for i in range(self.vehicle_num):
                input_loc_features = loc_features.clone()
                vehicle_loc = input_loc_features[:, self.loc_dim * i: self.loc_dim * (i + 1)].clone()
                input_loc_features = torch.cat((vehicle_loc, input_loc_features), dim=1)
                distribution, value, _ = self.ACP[i](input_loc_features, weight_features, None)
                values.append(value)
                distributions.append(distribution)
            return distributions, values, None
        else:
            log_probs = []
            entropys = []
            for i in range(self.vehicle_num):
                input_loc_features = loc_features.clone()
                vehicle_loc = input_loc_features[:, self.loc_dim * i: self.loc_dim * (i + 1)].clone()
                input_loc_features = torch.cat((vehicle_loc, input_loc_features), dim=1)
                value, log_prob, entropy = self.ACP[i](input_loc_features, weight_features, actions[:, i])
                values.append(value)
                log_probs.append(log_prob)
                entropys.append(entropy)
            return values, log_probs, entropys


if __name__ == '__main__':

    vehicle_num = 50
    loc_dim = 4
    weight_shape = (20, 20)
    share_policy = True
    ortho_init = True
    loc_feature_dim = (64,)
    weight_feature_params = ({
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': (3, 3),
        'stride': (2, 2),
        'padding': (1, 1),
        'dilation': (1, 1),
    }, )
    output_dim = (32, )
    share_params = False
    action_dim = 4
    learning_rate = 0.0001

    maacp = multi_agent_ACP(
        vehicle_num=vehicle_num,
        loc_dim=loc_dim,
        weight_shape=weight_shape,
        share_policy=share_policy,
        ortho_init=ortho_init,
        loc_feature_dim=loc_feature_dim,
        weight_feature_params=weight_feature_params,
        output_dim=output_dim,
        share_params=share_params,
        action_dim=action_dim,
        learning_rate=learning_rate
    )

    # test decision making
    bacth_size = 1
    loc_features = torch.randn((bacth_size, vehicle_num * loc_dim))
    weight_features = torch.randn((bacth_size, 1) + weight_shape)
    actions = None
    distributions, values, _ = maacp(
        loc_features=loc_features,
        weight_features=weight_features,
        actions=actions,
    )
    print(distributions)
    print(distributions[0].get_actions())
    print(distributions[1].get_actions())
    print(values)

    # test action probs
    bacth_size = 100
    loc_features = torch.randn((bacth_size, vehicle_num * loc_dim))
    weight_features = torch.randn((bacth_size, 1) + weight_shape)
    actions = torch.tensor(
        np.random.randint(low=0, high=4, size=(bacth_size, vehicle_num))
    )
    print(actions)
    values, log_probs, entropys = maacp(
        loc_features=loc_features,
        weight_features=weight_features,
        actions=actions,
    )
    print(values)
    print(log_probs)
    print(entropys)