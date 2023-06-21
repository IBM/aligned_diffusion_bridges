import torch
import torch.nn as nn

from sbalign.models.common import build_mlp
from sbalign.utils.sb_utils import get_timestep_embedding


class DoobHScore(nn.Module):

    def __init__(self, 
                 n_layers: int = 3,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 h_dim: int = 64,
                 activation: int = 'relu',
                 dropout_p: float = 0.1,
                 use_drift_in_doobs: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        # TODO: Make this a parameter
        time_embs_dim = 32

        if use_drift_in_doobs:
            input_dim = 3 * h_dim + time_embs_dim
        else:
            input_dim = 2 * h_dim + time_embs_dim

        self.time_embedding = get_timestep_embedding("sinusoidal", time_embs_dim)

        self.use_drift_in_doobs = use_drift_in_doobs

        self.x_enc = build_mlp(in_dim=in_dim, h_dim=h_dim, n_layers=2,
                             out_dim=h_dim, dropout_p=dropout_p,
                             activation=activation)
                             
        self.mlp = build_mlp(in_dim=input_dim, h_dim=h_dim, n_layers=n_layers,
                             out_dim=out_dim, dropout_p=dropout_p,
                             activation=activation)
    
    def forward(self, x, x_T, drift_x, t):
        time_embs = self.time_embedding(t)
        if self.use_drift_in_doobs:
            inputs = torch.cat([self.x_enc(x), self.x_enc(x_T), self.x_enc(drift_x), time_embs], dim=-1)
        else:
            inputs = torch.cat([self.x_enc(x), self.x_enc(x_T), time_embs], dim=-1)
        return self.mlp(inputs)
