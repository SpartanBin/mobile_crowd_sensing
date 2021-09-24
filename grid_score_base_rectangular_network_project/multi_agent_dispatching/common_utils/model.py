import torch
from torch import nn


class Conv1dExtractor(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(Conv1dExtractor, self).__init__()

        policy_conv_net = []
        value_conv_net = []
        if add_BN:
            policy_conv_net.append(nn.BatchNorm1d(conv_params[0]['in_channels']))
            value_conv_net.append(nn.BatchNorm1d(conv_params[0]['in_channels']))
        for layer_params in conv_params:
            policy_conv_net.append(nn.Conv1d(**layer_params))
            if add_BN:
                policy_conv_net.append(nn.BatchNorm1d(layer_params['out_channels']))
            policy_conv_net.append(nn.ELU())
            value_conv_net.append(nn.Conv1d(**layer_params))
            if add_BN:
                value_conv_net.append(nn.BatchNorm1d(layer_params['out_channels']))
            value_conv_net.append(nn.ELU())
        self.policy_conv_net = nn.Sequential(*policy_conv_net)
        self.value_conv_net = nn.Sequential(*value_conv_net)

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
            del self.value_conv_net
            del self.value_output_net

    def forward(self, features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        policy_output = self.policy_conv_net(features).flatten(start_dim=1, end_dim=-1)
        policy_output = self.policy_output_net(policy_output)
        value_output = None
        if not self.share_params:
            value_output = self.value_conv_net(features).flatten(start_dim=1, end_dim=-1)
            value_output = self.value_output_net(value_output)
        return policy_output, value_output


class CNN_splitting_CNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(CNN_splitting_CNN, self).__init__()

        individual_feature_conv = []
        individual_feature_conv.append(nn.BatchNorm1d(1))
        individual_feature_conv.append(nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            individual_feature_conv.append(nn.BatchNorm1d(32))
        individual_feature_conv.append(nn.ELU())
        individual_feature_conv.append(nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            individual_feature_conv.append(nn.BatchNorm1d(64))
        individual_feature_conv.append(nn.ELU())
        self.individual_feature_conv = nn.Sequential(*individual_feature_conv)

        common_feature_conv = []
        common_feature_conv.append(nn.BatchNorm1d(2))
        common_feature_conv.append(nn.Conv1d(
            in_channels=2,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(32))
        common_feature_conv.append(nn.ELU())
        common_feature_conv.append(nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(64))
        common_feature_conv.append(nn.ELU())
        self.common_feature_conv = nn.Sequential(*common_feature_conv)

        output_conv = []
        output_conv.append(nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            output_conv.append(nn.BatchNorm1d(128))
        output_conv.append(nn.ELU())
        self.output_conv = nn.Sequential(*output_conv)

        final_output = []
        final_output.append(nn.Linear(output_dim[0], 64))
        final_output.append(nn.Tanh())
        final_output.append(nn.Linear(64, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = features[:, 0, :].unsqueeze(dim=1)
        common_f = features[:, 1:, :]

        i_output = self.individual_feature_conv(individual_f)
        c_output = self.common_feature_conv(common_f)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.output_conv(output).flatten(start_dim=1, end_dim=-1)
        output = self.final_output(output)

        return output, None


class CNN_DNN_splitting_CNN_DNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(CNN_DNN_splitting_CNN_DNN, self).__init__()

        individual_feature_conv = []
        individual_feature_conv.append(nn.BatchNorm1d(1))
        individual_feature_conv.append(nn.Conv1d(
            in_channels=1,
            out_channels=44,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            individual_feature_conv.append(nn.BatchNorm1d(44))
        individual_feature_conv.append(nn.ELU())
        individual_feature_conv.append(nn.Conv1d(
            in_channels=44,
            out_channels=88,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            individual_feature_conv.append(nn.BatchNorm1d(88))
        individual_feature_conv.append(nn.ELU())
        self.individual_feature_conv = nn.Sequential(*individual_feature_conv)

        common_feature_conv = []
        common_feature_conv.append(nn.BatchNorm1d(2))
        common_feature_conv.append(nn.Conv1d(
            in_channels=2,
            out_channels=22,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(22))
        common_feature_conv.append(nn.ELU())
        common_feature_conv.append(nn.Conv1d(
            in_channels=22,
            out_channels=44,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(44))
        common_feature_conv.append(nn.ELU())
        self.common_feature_conv = nn.Sequential(*common_feature_conv)

        individual_feature_dnn = []
        individual_feature_dnn.append(nn.Linear(8800, 44))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(44))
        individual_feature_dnn.append(nn.ELU())
        self.individual_feature_dnn = nn.Sequential(*individual_feature_dnn)

        common_feature_dnn = []
        common_feature_dnn.append(nn.Linear(4400, 22))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(22))
        common_feature_dnn.append(nn.ELU())
        self.common_feature_dnn = nn.Sequential(*common_feature_dnn)

        final_output = []
        final_output.append(nn.Linear(66, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = features[:, 0, :].unsqueeze(dim=1)
        common_f = features[:, 1:, :]

        i_output = self.individual_feature_conv(individual_f).flatten(start_dim=1, end_dim=-1)
        c_output = self.common_feature_conv(common_f).flatten(start_dim=1, end_dim=-1)
        i_output = self.individual_feature_dnn(i_output)
        c_output = self.common_feature_dnn(c_output)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.final_output(output)

        return output, None


class DNN_splitting_CNN_DNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(DNN_splitting_CNN_DNN, self).__init__()

        individual_feature_dnn = []
        individual_feature_dnn.append(nn.Linear(400, 64))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(64))
        individual_feature_dnn.append(nn.ELU())
        individual_feature_dnn.append(nn.Linear(64, 32))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(32))
        individual_feature_dnn.append(nn.ELU())
        self.individual_feature_dnn = nn.Sequential(*individual_feature_dnn)

        common_feature_conv = []
        common_feature_conv.append(nn.BatchNorm1d(2))
        common_feature_conv.append(nn.Conv1d(
            in_channels=2,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(32))
        common_feature_conv.append(nn.ELU())
        common_feature_conv.append(nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(64))
        common_feature_conv.append(nn.ELU())
        self.common_feature_conv = nn.Sequential(*common_feature_conv)

        common_feature_dnn = []
        common_feature_dnn.append(nn.Linear(output_dim[0] // 2, 32))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(32))
        common_feature_dnn.append(nn.ELU())
        self.common_feature_dnn = nn.Sequential(*common_feature_dnn)

        final_output = []
        final_output.append(nn.Linear(64, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = features[:, 0, :]
        common_f = features[:, 1:, :]

        i_output = self.individual_feature_dnn(individual_f)
        c_output = self.common_feature_conv(common_f).flatten(start_dim=1, end_dim=-1)
        c_output = self.common_feature_dnn(c_output)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.final_output(output)

        return output, None


class ID_splitting_CNN_DNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_splitting_CNN_DNN, self).__init__()

        individual_feature_dnn = []
        individual_feature_dnn.append(nn.Linear(2, 64))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(64))
        individual_feature_dnn.append(nn.ELU())
        individual_feature_dnn.append(nn.Linear(64, 32))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(32))
        individual_feature_dnn.append(nn.ELU())
        self.individual_feature_dnn = nn.Sequential(*individual_feature_dnn)

        common_feature_conv = []
        common_feature_conv.append(nn.BatchNorm1d(2))
        common_feature_conv.append(nn.Conv1d(
            in_channels=2,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(32))
        common_feature_conv.append(nn.ELU())
        common_feature_conv.append(nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(64))
        common_feature_conv.append(nn.ELU())
        self.common_feature_conv = nn.Sequential(*common_feature_conv)

        common_feature_dnn = []
        common_feature_dnn.append(nn.Linear(output_dim[0] // 2, 32))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(32))
        common_feature_dnn.append(nn.ELU())
        self.common_feature_dnn = nn.Sequential(*common_feature_dnn)

        final_output = []
        final_output.append(nn.Linear(64, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 1:, :]

        i_output = self.individual_feature_dnn(individual_f)
        c_output = self.common_feature_conv(common_f).flatten(start_dim=1, end_dim=-1)
        c_output = self.common_feature_dnn(c_output)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.final_output(output)

        return output, None


class ID_allID_splitting_CNN_DNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_splitting_CNN_DNN, self).__init__()

        individual_feature_dnn = []
        individual_feature_dnn.append(nn.Linear(6, 64))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(64))
        individual_feature_dnn.append(nn.ELU())
        individual_feature_dnn.append(nn.Linear(64, 32))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(32))
        individual_feature_dnn.append(nn.ELU())
        self.individual_feature_dnn = nn.Sequential(*individual_feature_dnn)

        common_feature_conv = []
        common_feature_conv.append(nn.BatchNorm1d(1))
        common_feature_conv.append(nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(32))
        common_feature_conv.append(nn.ELU())
        common_feature_conv.append(nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
        ))
        if add_BN:
            common_feature_conv.append(nn.BatchNorm1d(64))
        common_feature_conv.append(nn.ELU())
        self.common_feature_conv = nn.Sequential(*common_feature_conv)

        common_feature_dnn = []
        common_feature_dnn.append(nn.Linear(output_dim[0] // 2, 32))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(32))
        common_feature_dnn.append(nn.ELU())
        self.common_feature_dnn = nn.Sequential(*common_feature_dnn)

        final_output = []
        final_output.append(nn.Linear(64, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :].unsqueeze(dim=1)

        i_output = self.individual_feature_dnn(individual_f)
        c_output = self.common_feature_conv(common_f).flatten(start_dim=1, end_dim=-1)
        c_output = self.common_feature_dnn(c_output)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.final_output(output)

        return output, None


class ID_allID_splitting_DNN(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_splitting_DNN, self).__init__()

        individual_feature_dnn = []
        individual_feature_dnn.append(nn.Linear(6, 64))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(64))
        individual_feature_dnn.append(nn.ELU())
        individual_feature_dnn.append(nn.Linear(64, 32))
        if add_BN:
            individual_feature_dnn.append(nn.BatchNorm1d(32))
        individual_feature_dnn.append(nn.ELU())
        self.individual_feature_dnn = nn.Sequential(*individual_feature_dnn)

        common_feature_dnn = []
        common_feature_dnn.append(nn.Linear(400, 64))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(64))
        common_feature_dnn.append(nn.ELU())
        common_feature_dnn.append(nn.Linear(64, 32))
        if add_BN:
            common_feature_dnn.append(nn.BatchNorm1d(32))
        common_feature_dnn.append(nn.ELU())
        self.common_feature_dnn = nn.Sequential(*common_feature_dnn)

        final_output = []
        final_output.append(nn.Linear(64, 32))
        final_output.append(nn.Tanh())
        self.final_output = nn.Sequential(*final_output)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        i_output = self.individual_feature_dnn(individual_f)
        c_output = self.common_feature_dnn(common_f)
        output = torch.cat((i_output, c_output), dim=1)
        output = self.final_output(output)

        return output, None


class ID_allID_DNN_structure1(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_DNN_structure1, self).__init__()

        add_BN = False
        dnn = []
        dnn.append(nn.Linear(406, 128))
        if add_BN:
            dnn.append(nn.BatchNorm1d(128))
        dnn.append(nn.ELU())
        dnn.append(nn.Linear(128, 64))
        if add_BN:
            dnn.append(nn.BatchNorm1d(64))
        dnn.append(nn.ELU())
        dnn.append(nn.Linear(64, 32))
        dnn.append(nn.Tanh())
        self.dnn = nn.Sequential(*dnn)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        input_features = torch.hstack((individual_f, common_f))
        output = self.dnn(input_features)

        return output, None


class ID_allID_DNN_structure2_a(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_DNN_structure2_a, self).__init__()

        dim = [406, 256, 128, 64, 32]

        dnn = []
        for i in range(len(dim)):
            if i > 0:
                dnn.append(nn.Linear(dim[i - 1], dim[i]))
                if add_BN:
                    dnn.append(nn.BatchNorm1d(dim[i]))
                dnn.append(nn.ELU())
        self.dnn = nn.Sequential(*dnn)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        input_features = torch.hstack((individual_f, common_f))
        output = self.dnn(input_features)

        return output, None


class ID_allID_DNN_structure2_b(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_DNN_structure2_b, self).__init__()

        dim = [406, 128, 64, 32]

        dnn = []
        for i in range(len(dim)):
            if i > 0:
                dnn.append(nn.Linear(dim[i - 1], dim[i]))
                if add_BN:
                    dnn.append(nn.BatchNorm1d(dim[i]))
                dnn.append(nn.ELU())
        self.dnn = nn.Sequential(*dnn)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        input_features = torch.hstack((individual_f, common_f))
        output = self.dnn(input_features)

        return output, None


class ID_allID_DNN_structure3(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_DNN_structure3, self).__init__()

        dim = [406, 256, 128, 64, 32]

        add_BN = False
        dnn = []
        for i in range(len(dim)):
            if i > 0:
                dnn.append(nn.Linear(dim[i - 1], dim[i]))
                if add_BN:
                    dnn.append(nn.BatchNorm1d(dim[i]))
                dnn.append(nn.Tanh())
        self.dnn = nn.Sequential(*dnn)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        input_features = torch.hstack((individual_f, common_f))
        output = self.dnn(input_features)

        return output, None


class ID_allID_DNN_structure4(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(ID_allID_DNN_structure4, self).__init__()

        dim = [406, 256, 128, 64, 32]

        add_BN = False
        dnn = []
        for i in range(len(dim)):
            if i > 0:
                dnn.append(nn.Linear(dim[i - 1], dim[i]))
                if add_BN:
                    dnn.append(nn.BatchNorm1d(dim[i]))
                dnn.append(nn.Sigmoid())
        self.dnn = nn.Sequential(*dnn)

    def forward(self, features: torch.Tensor, loc_ID):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        individual_f = torch.from_numpy(loc_ID).to(features.dtype).to(features.device)
        common_f = features[:, 2, :]

        input_features = torch.hstack((individual_f, common_f))
        output = self.dnn(input_features)

        return output, None