import sys
import os
from functools import partial
from typing import Tuple, Union
import copy

import numpy as np
from torch.distributions import Categorical

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from grid_score_base_project.multi_agent_dispatching.MACPPO.model import *


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

    def all_probs(self):
        return self.distribution.probs

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
        conv_params: list,
        add_BN: bool,
        output_dim: list,
        share_params: bool,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param ortho_init:
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(ActorCriticPolicy, self).__init__()

        self.ortho_init = ortho_init

        # self.extractor = Conv1dExtractor(
        #     conv_params=conv_params,
        #     add_BN=add_BN,
        #     output_dim=output_dim,
        #     share_params=share_params
        # )
        self.extractor = Extractor(
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            share_params=share_params,
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
                self.extractor: np.sqrt(2),
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
            input_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None,
    ):
        """
        Forward pass in all the networks (actor and critic)

        :param input_features:
        :param actions:
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.extractor(input_features)
        distribution = self.output_action_distribution(latent_pi)
        if latent_vf is None:
            latent_vf = latent_pi
        value = self.value_net(latent_vf)
        if actions is None:
            return distribution, value, None
        else:
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return value, log_prob, entropy


class multi_agent_ACP():

    def __init__(
            self,
            vehicle_num: int,
            weight_shape: int,
            ortho_init: bool,
            conv_params: list,
            add_BN: bool,
            output_dim: list,
            share_params: bool,
            action_dim: int,
            learning_rate: Union[int, float]):
        '''
        :param conv_params: with input layer params
        :param output_dim: without input dim
        :return:
        '''

        self.vehicle_num = vehicle_num

        conv_output_shape = weight_shape
        for params in conv_params:
            if 'padding' not in params:
                if type(conv_output_shape) == tuple:
                    params['padding'] = (0, 0)
                else:
                    params['padding'] = 0
            if 'dilation' not in params:
                if type(conv_output_shape) == tuple:
                    params['dilation'] = (1, 1)
                else:
                    params['dilation'] = 1
            if 'stride' not in params:
                if type(conv_output_shape) == tuple:
                    params['stride'] = (1, 1)
                else:
                    params['stride'] = 1
            conv_output_shape = self.cal_conv_output_shape(
                input_shape=conv_output_shape,
                kernel_size=params['kernel_size'],
                padding=params['padding'],
                dilation=params['dilation'],
                stride=params['stride'],
            )
        params = conv_params[-1]
        if type(conv_output_shape) == tuple:
            output_dim = [conv_output_shape[0] * conv_output_shape[1] *
                          params['out_channels']] + output_dim
        else:
            output_dim = [conv_output_shape * params['out_channels']] + output_dim

        self.ACP = ActorCriticPolicy(
            ortho_init=ortho_init,
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            share_params=share_params,
            action_dim=action_dim,
            learning_rate=learning_rate,
        )

        self.device = torch.device('cpu')

    @staticmethod
    def cal_conv_output_shape(input_shape, kernel_size, padding, dilation, stride):
        if type(input_shape) == tuple:
            row = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            col = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            return (row, col)
        else:
            return int((input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def to(self, param):
        self.ACP.to(param)
        if param == 'cuda':
            self.device = torch.device('cuda')
        elif param == torch.device('cuda'):
            self.device = torch.device('cuda')
        elif param == 'cpu':
            self.device = torch.device('cpu')
        elif param == torch.device('cpu'):
            self.device = torch.device('cpu')
        return self

    def train(self):
        self.ACP.train()
        return self

    def eval(self):
        self.ACP.eval()
        return self

    def state_dict(self):
        ACP_params = copy.deepcopy(self.ACP.state_dict())
        for key in ACP_params.keys():
            ACP_params[key] = ACP_params[key].to('cpu')
        return ACP_params

    def load_state_dict(self, ACP_params):
        self.ACP.load_state_dict(ACP_params)

    def optimize(self, loss: torch.Tensor, max_grad_norm):
        self.ACP.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.ACP.parameters(), max_grad_norm)
        self.ACP.optimizer.step()

    def forward(
            self,
            edge_loc_features: Union[np.ndarray, None] = None,
            the_same_features: Union[np.ndarray, None] = None,
            state_features: Union[np.ndarray, None] = None,
            actions: Union[np.ndarray, None] = None,
            loc_ID: Union[np.ndarray, None] = None,
            state_features_IDs: Union[np.ndarray, None] = None,
    ):
        """
        Forward pass in all the networks (actor and critic)

        :param edge_loc_features:
        :param the_same_features:
        :return: action, value and log probability of the action
        """
        values = []
        distributions = []
        if actions is None:

            ############## for ID_allID_DNN_structure ##############
            bacth_size = the_same_features.shape[0]
            the_same_features = the_same_features[:, -3:].reshape((bacth_size, -1))  # :, -1
            the_same_features = np.hstack((loc_ID, the_same_features))
            ############## ID_allID_DNN_structure end ##############


            for i in range(self.vehicle_num):

                ############## for Conv1dExtractor ##############
                # edge_loc_feature = edge_loc_features[:, i]
                # input_features = np.concatenate((edge_loc_feature, the_same_features), axis=1)
                ############## Conv1dExtractor end ##############

                ############## for ID_allID_DNN_structure ##############
                individual_loc = loc_ID[:, i * 2: (i + 1) * 2]
                input_features = np.hstack((individual_loc, the_same_features))
                ############## ID_allID_DNN_structure end ##############

                input_features = torch.from_numpy(input_features).to(self.device)
                distribution, value, _ = self.ACP(input_features, None)
                distributions.append(copy.deepcopy(distribution))
                values.append(value.clone())
            values = torch.cat(values, dim=1)
            return distributions, values, None
        else:

            ############## for Conv1dExtractor ##############
            # input_features = torch.from_numpy(state_features).to(self.device)
            ############## Conv1dExtractor end ##############

            ############## for ID_allID_DNN_structure ##############
            input_features = torch.from_numpy(state_features_IDs).to(self.device)
            ############## ID_allID_DNN_structure end ##############

            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).long()
            values, log_probs, entropy = self.ACP(input_features, actions)
            log_probs = log_probs.view(log_probs.shape + (1,))
            entropy = entropy.view(entropy.shape + (1,))
            return values, log_probs, entropy


if __name__ == '__main__':

    vehicle_num = 50
    weight_shape = 20 * 20
    conv_params = [{
        'in_channels': 8,
        'out_channels': 1,
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'dilation': 1,
    }]

    maacp = multi_agent_ACP(
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        ortho_init=True,
        conv_params=conv_params,
        add_BN=True,
        output_dim=[32],
        share_params=False,
        action_dim=4,
        learning_rate=0.00001
    )

    # test decision making
    bacth_size = 1
    edge_loc_features = np.random.rand(bacth_size, vehicle_num, 1, weight_shape).astype(np.float32)
    the_same_features = np.random.rand(bacth_size, 7, weight_shape).astype(np.float32)
    distributions, values, _ = maacp.forward(
        edge_loc_features=edge_loc_features,
        the_same_features=the_same_features,
        actions=None,
    )
    print(distributions)
    print(distributions[0].get_actions())
    print(distributions[1].get_actions())
    print(distributions[1].all_probs())
    print(values.shape)

    # test action probs
    bacth_size = 100
    edge_loc_features = np.random.rand(bacth_size, 1, weight_shape).astype(np.float32)
    the_same_features = np.random.rand(bacth_size, 7, weight_shape).astype(np.float32)
    state_features = np.concatenate((edge_loc_features, the_same_features), axis=1)
    actions = np.random.randint(low=0, high=4, size=(bacth_size,))
    print(actions)
    values, log_probs, entropy = maacp.forward(
        state_features=state_features,
        actions=actions,
    )
    print(values)
    print(log_probs)
    print(entropy)