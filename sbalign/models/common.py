from typing import Union, List

import torch
import torch.nn as nn


def build_linear(in_dim, out_dim, activation="linear"):
    lin_layer = nn.Linear(in_dim, out_dim)
    if activation == 'silu':
        gain = 1.
    else:
        gain = torch.nn.init.calculate_gain(activation)
    torch.nn.init.xavier_normal_(lin_layer.weight, gain=gain)
    return lin_layer


def build_mlp(in_dim: int,
              h_dim: Union[int, List],
              n_layers: int = None,
              out_dim: int = None,
              dropout_p: float = 0.2,
              activation: str = 'relu') -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """

    if isinstance(h_dim, list) and n_layers is not None:
        print("n_layers should be None if h_dim is a list. Skipping")

    if isinstance(h_dim, int):
        h_dim = [h_dim]
        if n_layers is not None:
            h_dim = h_dim * n_layers

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(build_linear(prev_size, next_size, activation))
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'selu':
            layers.append(nn.SELU())
        elif activation == 'silu':
            layers.append(nn.SiLU())
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(build_linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)
    