import torch
import torch.nn as nn

from sbalign.models.common import build_mlp
from sbalign.utils.sb_utils import get_timestep_embedding


class SDEDrift(nn.Module):

    def __init__(self,
                 timestep_emb_dim: int = 64, 
                 n_layers: int = 3,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 h_dim: int = 64,
                 activation: int = 'relu',
                 dropout_p: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.x_enc = build_mlp(in_dim=in_dim, h_dim=h_dim, n_layers=n_layers,
                             out_dim=h_dim, dropout_p=dropout_p,
                             activation=activation)
        
        self.t_enc = build_mlp(in_dim=timestep_emb_dim, h_dim=h_dim, n_layers=2,
                             out_dim=h_dim, dropout_p=dropout_p,
                             activation=activation)

        self.mlp = build_mlp(in_dim=2*h_dim, h_dim=h_dim, n_layers=n_layers,
                             out_dim=out_dim, dropout_p=dropout_p,
                             activation=activation)
        if timestep_emb_dim > 1:
            self.timestep_emb_fn = get_timestep_embedding('sinusoidal', embedding_dim=timestep_emb_dim)
        else:
            self.timestep_emb_fn = lambda x: x 
    
    def forward(self, x, t):
        t_encoded = self.t_enc(self.timestep_emb_fn(t))
        x_encoded = self.x_enc(x)
        inputs = torch.cat([x_encoded, t_encoded], dim=-1)
        return self.mlp(inputs)
