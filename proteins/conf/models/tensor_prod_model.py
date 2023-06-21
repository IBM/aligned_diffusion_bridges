import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter
from torch_cluster import radius, radius_graph


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, residual=True, dropout=0.0,
                 h_dim=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if h_dim is None:
            h_dim = edge_fdim
        
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc_net = nn.Sequential(
            nn.Linear(edge_fdim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, tp.weight_numel)
        )

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, aggr='mean'):
        edge_src, edge_dst = edge_index
        tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))

        out_nodes = out_nodes or x.shape[0]

        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)
        #assert out.shape[0] == x.shape[0]

        if self.residual:
            padded = F.pad(x, (0, out.shape[-1] - x.shape[-1]))
            out = out + padded

        return out


class ConfTensorProductModel(nn.Module):

    def __init__(self, timestep_emb_fn, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2, 
                 max_radius: float = 10.0, max_neighbors: int = 24,
                 distance_emb_dim: int = 32, max_distance: float = 30.0,
                 timestep_emb_dim: int = 32, dropout_p: float = 0.2,
                 **kwargs
        ):
            super().__init__(**kwargs)

            self.node_fdim = node_fdim
            self.edge_fdim = edge_fdim
            self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
            self.n_s, self.n_v = n_s, n_v
            self.n_conv_layers = n_conv_layers
            self.timestep_emb_fn = timestep_emb_fn

            self.max_radius = max_radius
            self.max_neighbors = max_neighbors
                                                
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e + {n_v}x1o",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
            ]

            self.node_embedding = nn.Sequential(
                nn.Linear(node_fdim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + timestep_emb_dim + distance_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )

            # Distance expansions for different graphs
            self.dist_expansion = GaussianSmearing(start=0.0, stop=max_distance, 
                                                       num_gaussians=distance_emb_dim)

            conv_layers = []
            for i in range(n_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq)-1)]
                out_irreps = irrep_seq[min(i+1, len(irrep_seq)-1)]

                parameters = {
                    "in_irreps": in_irreps,
                    "sh_irreps": self.sh_irreps,
                    "out_irreps": out_irreps,
                    "edge_fdim": 3 * n_s,
                    "h_dim": 3 * n_s,
                    "residual": False,
                    "dropout": dropout_p
                }

                layer = TensorProductConvLayer(**parameters)
                conv_layers.append(layer)

            self.conv_layers = nn.ModuleList(conv_layers)

            self.final_conv_layer = TensorProductConvLayer(
                in_irreps=conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                edge_fdim=3 * n_s,
                h_dim=2 * n_s,
                residual=False,
                dropout=dropout_p,
            )

    def forward(self, data):
        x, edge_index, edge_attr, edge_sh = self.build_graph(data, pos_to_use="current")
        src, dst = edge_index
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
            msg = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)

            x = F.pad(x, (0, msg.shape[-1] - x.shape[-1]))
            x = x + msg

        # Add final convolution layer to predict drift and doobs
        edge_attr_final = torch.cat([edge_attr, x[src, :self.n_s], x[dst, :self.n_s]], dim=-1)
        global_pred = self.final_conv_layer(x, edge_index, edge_attr_final, edge_sh)

        drift_pred = global_pred[:, :3]+ global_pred[:, 6:9]
        doobs_h_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        return drift_pred.expand(x.shape[0], 3), doobs_h_pred.expand(x.shape[0], 3), None

    def build_graph(self, data, pos_to_use: str = "current"):
        if pos_to_use == "current":
            pos = data.pos_t
            t_emb = self.timestep_emb_fn(data.t)
        elif pos_to_use == "final":
            pos = data.pos_T
            t_emb = self.timestep_emb_fn(torch.ones_like(data.t).unsqueeze(1))
        elif pos_to_use == "init":
            pos = data.pos_0

        node_attr = torch.cat([data.x, t_emb], dim=-1)

        edge_index = radius_graph(
            x=pos, r=self.max_radius, max_num_neighbors=self.max_neighbors,
            batch=data.batch
        )

        src, dst = edge_index
        edge_vec = pos[dst.long()] - pos[src.long()]

        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))
        edge_t_emb = t_emb[src.long()]
        edge_attr = torch.cat([edge_length_emb, edge_t_emb], dim=-1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, 
                                         normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh

    def run_drift(self, data):
        drift, _, _ = self(data)
        return drift
