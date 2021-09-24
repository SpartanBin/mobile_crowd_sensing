from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, GlobalAttention, Set2Set, dense_diff_pool


class GCN_GA_Extractor(nn.Module):

    def __init__(self, in_channels: int):
        '''
        GCN + global_attention_pooling
        '''
        super(GCN_GA_Extractor, self).__init__()

        first_in_channels = in_channels

        #### actor ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_a = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_a = BatchNorm(in_channels=out_channels)

        self.pooling_layer_a = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_channels, out_channels // 3),
                nn.BatchNorm1d(out_channels // 3),
                nn.ReLU(),
                nn.Linear(out_channels // 3, 1)
            ),
            nn=None,
        )

        self.dnn1_a = nn.Linear(out_channels, 64)
        self.dnn_bn_a = nn.BatchNorm1d(64)
        self.dnn2_a = nn.Linear(64, 32)

        ########

        #### critic ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_c = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_c = BatchNorm(in_channels=out_channels)

        self.pooling_layer_c = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_channels, out_channels // 3),
                nn.BatchNorm1d(out_channels // 3),
                nn.ReLU(),
                nn.Linear(out_channels // 3, 1)
            ),
            nn=None,
        )

        self.dnn1_c = nn.Linear(out_channels, 64)
        self.dnn_bn_c = nn.BatchNorm1d(64)
        self.dnn2_c = nn.Linear(64, 32)

        ########

    def forward(
            self,
            x: torch.Tensor,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.LongTensor,
    ):

        #### actor ####

        embedding_x = F.elu(self.bn1_a(self.graph_embedding_layer1_a(self.bn0_a(x), edge_index, edge_weight)))
        embedding_x = F.elu(self.bn2_a(self.graph_embedding_layer2_a(embedding_x, edge_index, edge_weight)))
        embedding_x = F.elu(self.bn3_a(self.graph_embedding_layer3_a(embedding_x, edge_index, edge_weight)))
        a_output = self.pooling_layer_a(embedding_x, batch)
        a_output = F.elu(self.dnn_bn_a(self.dnn1_a(a_output)))
        a_output = self.dnn2_a(a_output)

        ########

        #### critic ####

        embedding_x = F.elu(self.bn1_c(self.graph_embedding_layer1_c(self.bn0_c(x), edge_index, edge_weight)))
        embedding_x = F.elu(self.bn2_c(self.graph_embedding_layer2_c(embedding_x, edge_index, edge_weight)))
        embedding_x = F.elu(self.bn3_c(self.graph_embedding_layer3_c(embedding_x, edge_index, edge_weight)))
        c_output = self.pooling_layer_c(embedding_x, batch)
        c_output = F.elu(self.dnn_bn_c(self.dnn1_c(c_output)))
        c_output = self.dnn2_c(c_output)

        ########

        return a_output, c_output


class GCN_S2S_Extractor(nn.Module):

    def __init__(self, in_channels: int):
        '''
        GCNConv + Set2Set
        '''
        super(GCN_S2S_Extractor, self).__init__()

        first_in_channels = in_channels

        #### actor ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_a = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_a = BatchNorm(in_channels=out_channels)

        self.pooling_layer_a = Set2Set(
            in_channels=out_channels,
            processing_steps=4,
            num_layers=1,
        )

        self.dnn1_a = nn.Linear(out_channels * 2, 64)
        self.dnn_bn_a = nn.BatchNorm1d(64)
        self.dnn2_a = nn.Linear(64, 32)

        ########

        #### critic ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_c = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_c = BatchNorm(in_channels=out_channels)

        self.pooling_layer_c = Set2Set(
            in_channels=out_channels,
            processing_steps=4,
            num_layers=2,
        )

        self.dnn1_c = nn.Linear(out_channels * 2, 64)
        self.dnn_bn_c = nn.BatchNorm1d(64)
        self.dnn2_c = nn.Linear(64, 32)

        ########

    def forward(
            self,
            x: torch.Tensor,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.LongTensor,
    ):

        #### actor ####

        embedding_x = F.elu(self.bn1_a(self.graph_embedding_layer1_a(self.bn0_a(x), edge_index)))
        embedding_x = F.elu(self.bn2_a(self.graph_embedding_layer2_a(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_a(self.graph_embedding_layer3_a(embedding_x, edge_index)))
        a_output = self.pooling_layer_a(x=embedding_x, batch=batch)
        a_output = F.elu(self.dnn_bn_a(self.dnn1_a(a_output)))
        a_output = self.dnn2_a(a_output)

        ########

        #### critic ####

        embedding_x = F.elu(self.bn1_c(self.graph_embedding_layer1_c(self.bn0_c(x), edge_index)))
        embedding_x = F.elu(self.bn2_c(self.graph_embedding_layer2_c(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_c(self.graph_embedding_layer3_c(embedding_x, edge_index)))
        c_output = self.pooling_layer_c(x=embedding_x, batch=batch)
        c_output = F.elu(self.dnn_bn_c(self.dnn1_c(c_output)))
        c_output = self.dnn2_c(c_output)

        ########

        return a_output, c_output


class SAGE_S2S_Extractor(nn.Module):

    def __init__(self, in_channels: int):
        '''
        GCNConv + Set2Set
        '''
        super(SAGE_S2S_Extractor, self).__init__()

        first_in_channels = in_channels

        #### actor ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_a = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_a = BatchNorm(in_channels=out_channels)

        self.pooling_layer_a = Set2Set(
            in_channels=out_channels,
            processing_steps=4,
            num_layers=1,
        )

        self.dnn1_a = nn.Linear(out_channels * 2, 64)
        self.dnn_bn_a = nn.BatchNorm1d(64)
        self.dnn2_a = nn.Linear(64, 32)

        ########

        #### critic ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_c = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_c = BatchNorm(in_channels=out_channels)

        self.pooling_layer_c = Set2Set(
            in_channels=out_channels,
            processing_steps=4,
            num_layers=2,
        )

        self.dnn1_c = nn.Linear(out_channels * 2, 64)
        self.dnn_bn_c = nn.BatchNorm1d(64)
        self.dnn2_c = nn.Linear(64, 32)

        ########

    def forward(
            self,
            x: torch.Tensor,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.LongTensor,
    ):

        #### actor ####

        embedding_x = F.elu(self.bn1_a(self.graph_embedding_layer1_a(self.bn0_a(x), edge_index)))
        embedding_x = F.elu(self.bn2_a(self.graph_embedding_layer2_a(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_a(self.graph_embedding_layer3_a(embedding_x, edge_index)))
        a_output = self.pooling_layer_a(x=embedding_x, batch=batch)
        a_output = F.elu(self.dnn_bn_a(self.dnn1_a(a_output)))
        a_output = self.dnn2_a(a_output)

        ########

        #### critic ####

        embedding_x = F.elu(self.bn1_c(self.graph_embedding_layer1_c(self.bn0_c(x), edge_index)))
        embedding_x = F.elu(self.bn2_c(self.graph_embedding_layer2_c(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_c(self.graph_embedding_layer3_c(embedding_x, edge_index)))
        c_output = self.pooling_layer_c(x=embedding_x, batch=batch)
        c_output = F.elu(self.dnn_bn_c(self.dnn1_c(c_output)))
        c_output = self.dnn2_c(c_output)

        ########

        return a_output, c_output


class SAGE_GA_Extractor(nn.Module):

    def __init__(self, in_channels: int):
        '''
        GCNConv + Set2Set
        '''
        super(SAGE_GA_Extractor, self).__init__()

        first_in_channels = in_channels

        #### actor ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_a = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_a = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_a = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_a = BatchNorm(in_channels=out_channels)

        self.pooling_layer_a = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_channels, out_channels // 3),
                nn.BatchNorm1d(out_channels // 3),
                nn.ReLU(),
                nn.Linear(out_channels // 3, 1)
            ),
            nn=None,
        )

        self.dnn1_a = nn.Linear(out_channels, 64)
        self.dnn_bn_a = nn.BatchNorm1d(64)
        self.dnn2_a = nn.Linear(64, 32)

        ########

        #### critic ####

        in_channels = first_in_channels
        out_channels = 32
        self.bn0_c = BatchNorm(in_channels=in_channels)
        self.graph_embedding_layer1_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 64
        self.graph_embedding_layer2_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn2_c = BatchNorm(in_channels=out_channels)

        in_channels = out_channels
        out_channels = 128
        self.graph_embedding_layer3_c = SAGEConv(in_channels=in_channels, out_channels=out_channels)
        self.bn3_c = BatchNorm(in_channels=out_channels)

        self.pooling_layer_c = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(out_channels, out_channels // 3),
                nn.BatchNorm1d(out_channels // 3),
                nn.ReLU(),
                nn.Linear(out_channels // 3, 1)
            ),
            nn=None,
        )

        self.dnn1_c = nn.Linear(out_channels, 64)
        self.dnn_bn_c = nn.BatchNorm1d(64)
        self.dnn2_c = nn.Linear(64, 32)

        ########

    def forward(
            self,
            x: torch.Tensor,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.LongTensor,
    ):

        #### actor ####

        embedding_x = F.elu(self.bn1_a(self.graph_embedding_layer1_a(self.bn0_a(x), edge_index)))
        embedding_x = F.elu(self.bn2_a(self.graph_embedding_layer2_a(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_a(self.graph_embedding_layer3_a(embedding_x, edge_index)))
        a_output = self.pooling_layer_a(x=embedding_x, batch=batch)
        a_output = F.elu(self.dnn_bn_a(self.dnn1_a(a_output)))
        a_output = self.dnn2_a(a_output)

        ########

        #### critic ####

        embedding_x = F.elu(self.bn1_c(self.graph_embedding_layer1_c(self.bn0_c(x), edge_index)))
        embedding_x = F.elu(self.bn2_c(self.graph_embedding_layer2_c(embedding_x, edge_index)))
        embedding_x = F.elu(self.bn3_c(self.graph_embedding_layer3_c(embedding_x, edge_index)))
        c_output = self.pooling_layer_c(x=embedding_x, batch=batch)
        c_output = F.elu(self.dnn_bn_c(self.dnn1_c(c_output)))
        c_output = self.dnn2_c(c_output)

        ########

        return a_output, c_output


# class SAGE_DP_Extractor(nn.Module):
#
#     def __init__(self, in_channels: int):
#         '''
#         SAGE + Differentiable_Pooling
#         '''
#         super(SAGE_DP_Extractor, self).__init__()
#
#         first_in_channels = in_channels
#
#         #### actor ####
#
#         in_channels = first_in_channels
#         out_channels = 32
#         self.bn0_a = BatchNorm(in_channels=in_channels)
#         self.graph_embedding_layer1_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn1_a = BatchNorm(in_channels=out_channels)
#
#         in_channels = out_channels
#         out_channels = 64
#         self.graph_embedding_layer2_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn2_a = BatchNorm(in_channels=out_channels)
#
#         in_channels = out_channels
#         out_channels = 128
#         self.graph_embedding_layer3_a = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn3_a = BatchNorm(in_channels=out_channels)
#
#         self.dnn1_a = nn.Linear(out_channels, 64)
#         self.dnn_bn_a = nn.BatchNorm1d(64)
#         self.dnn2_a = nn.Linear(64, 32)
#
#         ########
#
#         #### critic ####
#
#         in_channels = first_in_channels
#         out_channels = 32
#         self.bn0_c = BatchNorm(in_channels=in_channels)
#         self.graph_embedding_layer1_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn1_c = BatchNorm(in_channels=out_channels)
#
#         in_channels = out_channels
#         out_channels = 64
#         self.graph_embedding_layer2_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn2_c = BatchNorm(in_channels=out_channels)
#
#         in_channels = out_channels
#         out_channels = 128
#         self.graph_embedding_layer3_c = GCNConv(in_channels=in_channels, out_channels=out_channels)
#         self.bn3_c = BatchNorm(in_channels=out_channels)
#
#         self.dnn1_c = nn.Linear(out_channels, 64)
#         self.dnn_bn_c = nn.BatchNorm1d(64)
#         self.dnn2_c = nn.Linear(64, 32)
#
#         ########
#
#     def forward(
#             self,
#             x: torch.Tensor,
#             edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
#             edge_weight: Optional[torch.Tensor],
#             batch: torch.LongTensor,
#     ):
#
#         #### actor ####
#
#         embedding_x = F.elu(self.bn1_a(self.graph_embedding_layer1_a(self.bn0_a(x), edge_index, edge_weight)))
#         embedding_x = F.elu(self.bn2_a(self.graph_embedding_layer2_a(embedding_x, edge_index, edge_weight)))
#         embedding_x = F.elu(self.bn3_a(self.graph_embedding_layer3_a(embedding_x, edge_index, edge_weight)))
#         a_output = dense_diff_pool(embedding_x, batch)
#         a_output = F.elu(self.dnn_bn_a(self.dnn1_a(a_output)))
#         a_output = self.dnn2_a(a_output)
#
#         ########
#
#         #### critic ####
#
#         embedding_x = F.elu(self.bn1_c(self.graph_embedding_layer1_c(self.bn0_c(x), edge_index, edge_weight)))
#         embedding_x = F.elu(self.bn2_c(self.graph_embedding_layer2_c(embedding_x, edge_index, edge_weight)))
#         embedding_x = F.elu(self.bn3_c(self.graph_embedding_layer3_c(embedding_x, edge_index, edge_weight)))
#         c_output = dense_diff_pool(embedding_x, batch)
#         c_output = F.elu(self.dnn_bn_c(self.dnn1_c(c_output)))
#         c_output = self.dnn2_c(c_output)
#
#         ########
#
#         return a_output, c_output