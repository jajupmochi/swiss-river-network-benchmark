"""
temporal_gat_conv



@Author: linlin
@Date: Oct 21 2025
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class TemporalGATConv(GATConv):
    r"""The temporal version of the GATConv operator from the `"Graph
    Attention Networks" <https://arxiv.org/abs/1710.10903>`_ paper.

    Here we assume a static graph structure with dynamic node features.

    Shapes:
        - **input:**
          node features :math:`(B, T, |\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)`

    Reference:
        https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GATConv.html
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In GATConv, node_dim is set to 0, which is not the default behavior of MessagePassing and does not match the
        # temporal case. We set it to 2 here to skip Batch and Time dimensions.
        # Notice: node_dim = -2 is not valid, since in some cases a head dimension is added after node dimension.
        self.node_dim = 2


    def forward(  # noqa: F811
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            size: Size = None,
            return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)

        Shapes:
            - x: [B, T, n_nodes, node_dim] or a tuple of such tensors
            - edge_index: [2, n_edges]
            - edge_attr: [n_edges, edge_dim]
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            # assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)  # todo: validate this for temporal case

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(*x.shape[:-1], H, C)
            else:  # todo: validate for temporal case
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features: todo: validate for temporal case
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:  # todo: validate for temporal case
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr
                )
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(
            edge_index, alpha=alpha, edge_attr=edge_attr,
            size=size
        )

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)  # [B, T, n_nodes, heads, out_channels]

        # Head pooling:
        if self.concat:
            # original: out = out.view(-1, self.heads * self.out_channels)
            out = out.view(*out.shape[:-2], self.heads * self.out_channels)  # [B, T, n_nodes, heads * out_channels]
        else:
            out = out.mean(dim=-2)  # [B, T, n_nodes, out_channels]. original: out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):  # todo: validate for temporal case
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(
            self, alpha_j: Tensor, alpha_i: OptTensor,
            edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
            dim_size: Optional[int]
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size, dim=-2)  # dim is the edge dimension. The default is 0.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    # def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
    #     """
    #     Shapes:
    #         - x_j: [B, T, n_nodes, node_dim]
    #         - edge_attr: [n_edges, edge_dim]
    #     """
    #     weight = self.nn(edge_attr)
    #     weight = weight.view(-1, self.in_channels_l, self.out_channels)  # [n_edges, in_channels, out_channels]
    #     return torch.matmul(x_j.unsqueeze(3), weight).squeeze(3)  # [B, T, n_edges, out_channels]
