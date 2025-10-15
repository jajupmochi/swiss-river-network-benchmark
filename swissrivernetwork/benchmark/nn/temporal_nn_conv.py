"""
tempo_nn_conv



@Author: linlin
@Date: Oct 15 2025
"""
import torch
from torch import Tensor
from torch_geometric.nn import NNConv


class TemporalNNConv(NNConv):
    r"""The temporal version of the NNConv operator from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    Here we assume a static graph structure with dynamic node features.

    Shapes:
        - **input:**
          node features :math:`(B, T, |\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)`

    """


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Shapes:
            - x_j: [B, T, n_nodes, node_dim]
            - edge_attr: [n_edges, edge_dim]
        """
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)  # [n_edges, in_channels, out_channels]
        return torch.matmul(x_j.unsqueeze(3), weight).squeeze(3)  # [B, T, n_edges, out_channels]
