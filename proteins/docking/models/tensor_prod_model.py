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


class DockTensorProductModel(nn.Module):

    def __init__(self, timestep_emb_fn, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2, 
                 lig_max_radius: float = 10.0, rec_max_radius: float = 1.0,
                 lig_max_neighbors: int = 24, rec_max_neighbors: int = 24, 
                 cross_max_distance: float = 10.0, distance_emb_dim: int = 32,
                 cross_dist_emb_dim: float = 32, center_max_distance: float = 30.0,
                 timestep_emb_dim: int = 32, dropout_p: float = 0.2,
                 **kwargs
        ):
            super().__init__(**kwargs)

            self.node_fdim = node_fdim
            self.edge_fdim = edge_fdim
            self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
            self.n_s, self.n_v = n_s, n_v
            self.n_conv_layers = n_conv_layers

            self.lig_max_radius = lig_max_radius
            self.rec_max_radius = rec_max_radius
            self.cross_max_distance = cross_max_distance 
            self.lig_max_neighbors = lig_max_neighbors
            self.rec_max_neighbors = rec_max_neighbors
            self.timestep_emb_fn = timestep_emb_fn
                                                
            irrep_seq = [
                f"{n_s}x0e",
                f"{n_s}x0e + {n_v}x1o",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
                f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
            ]

            self.lig_node_embedding = nn.Sequential(
                nn.Linear(node_fdim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
            self.lig_edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + timestep_emb_dim + distance_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )

            self.rec_node_embedding = nn.Sequential(
                nn.Linear(node_fdim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
            self.rec_edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + distance_emb_dim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )

            self.cross_edge_embedding = nn.Sequential(
                nn.Linear(cross_dist_emb_dim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )

            # Distance expansions for different graphs
            self.lig_dist_expansion = GaussianSmearing(start=0.0, stop=lig_max_radius, 
                                                       num_gaussians=distance_emb_dim)
            self.rec_dist_expansion = GaussianSmearing(start=0.0, stop=rec_max_radius,
                                                       num_gaussians=distance_emb_dim)
            self.cross_dist_expansion = GaussianSmearing(start=0.0, stop=cross_max_distance,
                                            num_gaussians=cross_dist_emb_dim)

            lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
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

                lig_layer = TensorProductConvLayer(**parameters)
                lig_conv_layers.append(lig_layer)
                rec_layer = TensorProductConvLayer(**parameters)
                rec_conv_layers.append(rec_layer)
                lig_to_rec_layer = TensorProductConvLayer(**parameters)
                lig_to_rec_conv_layers.append(lig_to_rec_layer)
                rec_to_lig_layer = TensorProductConvLayer(**parameters)
                rec_to_lig_conv_layers.append(rec_to_lig_layer)
                
            self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
            self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
            self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
            self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_emb_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_emb_dim + timestep_emb_dim, n_s),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(n_s, n_s)
            )
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                edge_fdim=2 * n_s,
                residual=False,
                dropout=dropout_p,
            )

    def forward(self, data, eval_at_bound: bool = False):
        # build ligand graph
        lig_x, lig_edge_index, lig_edge_attr, lig_sh = self.build_lig_graph(data, pos_to_use="current")
        lig_src, lig_dst = lig_edge_index
        lig_x = self.lig_node_embedding(lig_x)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        rec_x, rec_edge_index, rec_edge_attr, rec_sh = self.build_rec_graph(data, pos_to_use="current")
        rec_src, rec_dst = rec_edge_index
        rec_x = self.rec_node_embedding(rec_x)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_graph(data, cross_cutoff, 
                                                                    pos_to_use="current")
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for i in range(len(self.lig_conv_layers)):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_x[lig_src, :self.n_s], lig_x[lig_dst, :self.n_s]], dim=-1)
            lig_intra_msg = self.lig_conv_layers[i](lig_x, lig_edge_index, lig_edge_attr_, lig_sh)

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_x[cross_lig, :self.n_s], rec_x[cross_rec, :self.n_s]], -1)
            lig_inter_msg = self.rec_to_lig_conv_layers[i](rec_x, torch.flip(cross_edge_index, dims=[0]), rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_x.shape[0])

            if i != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_x[rec_src, :self.n_s], rec_x[rec_dst, :self.n_s]], -1)
                rec_intra_msg = self.rec_conv_layers[i](rec_x, rec_edge_index, rec_edge_attr_, rec_sh)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_x[cross_lig, :self.n_s], rec_x[cross_rec, :self.n_s]], -1)
                rec_inter_msg = self.lig_to_rec_conv_layers[i](lig_x, cross_edge_index, lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_x.shape[0])

            # padding original features
            lig_x = F.pad(lig_x, (0, lig_intra_msg.shape[-1] - lig_x.shape[-1]))

            # update features with residual updates
            lig_x = lig_x + lig_intra_msg + lig_inter_msg

            if i != len(self.lig_conv_layers) - 1:
                rec_x = F.pad(rec_x, (0, rec_intra_msg.shape[-1] - rec_x.shape[-1]))
                rec_x = rec_x + rec_intra_msg + rec_inter_msg

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_graph(data, pos_to_use="current")

        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_x[center_edge_index[1], :self.n_s]], -1)
        global_pred = self.final_conv(lig_x, center_edge_index, center_edge_attr, center_edge_sh, 
                                      out_nodes=data.num_graphs)

        drift_pred = global_pred[:, :3]+ global_pred[:, 6:9]
        doobs_h_pred = global_pred[:, 3:6] + global_pred[:, 9:]

        return drift_pred.expand(lig_x.shape[0], 3), doobs_h_pred.expand(lig_x.shape[0], 3), None

    def run_drift(self, data):
        drift, _, _ = self(data)
        return drift

    def build_lig_graph(self, data, pos_to_use: str = "current"):
        if pos_to_use == "final":
            pos = data["ligand"].pos_T
            shape = (data["ligand"].x.size(0), 1)
            t_emb = self.timestep_emb_fn(data["ligand"].x.new_ones(shape))

        elif pos_to_use == "current":
            pos = data["ligand"].pos_t
            t_emb = self.timestep_emb_fn(data["ligand"].t)

        node_attr = torch.cat([data["ligand"].x, t_emb], dim=-1)

        edge_index = radius_graph(
            x=pos, r=self.lig_max_radius,
            batch=data["ligand"].batch,
            max_num_neighbors=self.lig_max_neighbors
        )

        src, dst = edge_index
        edge_vec = pos[dst.long()] - pos[src.long()]

        edge_length_emb = self.lig_dist_expansion(edge_vec.norm(dim=-1))
        edge_t_emb = t_emb[src.long()]
        edge_attr = torch.cat([edge_length_emb, edge_t_emb], dim=-1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_graph(self, data, pos_to_use: str = "current"):
        assert torch.allclose(data["receptor"].pos_T, data["receptor"].pos_0, atol=1e-5)

        if pos_to_use == "final":
            pos = data["receptor"].pos_T
            shape = (data["receptor"].x.size(0), 1)
            t_emb = self.timestep_emb_fn(data["receptor"].x.new_ones(shape))

        elif pos_to_use == "current":
            pos = data["receptor"].pos_0
            t_emb = self.timestep_emb_fn(data["receptor"].t)

        node_attr = torch.cat([data["receptor"].x, t_emb], dim=-1)

        edge_index = radius_graph(
            x=pos, r=self.rec_max_radius,
            batch=data["receptor"].batch,
            max_num_neighbors=self.rec_max_neighbors
        )
        src, dst = edge_index
        edge_vec = pos[dst.long()] - pos[src.long()]

        edge_length_emb = self.rec_dist_expansion(edge_vec.norm(dim=-1))
        edge_t_emb = t_emb[src.long()]
        edge_attr = torch.cat([edge_length_emb, edge_t_emb], dim=-1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, 
                                         normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh
    
    def build_cross_graph(self, data, cross_distance_cutoff, pos_to_use: str = "current"):
        if pos_to_use == "final":
            lig_pos = data["ligand"].pos_T
            rec_pos = data["receptor"].pos_T
            t_emb = self.timestep_emb_fn(data["ligand"].x.new_ones(lig_pos.size(0), 1))

        elif pos_to_use == "current":
            lig_pos = data["ligand"].pos_t
            rec_pos = data["receptor"].pos_0
            t_emb = self.timestep_emb_fn(data["ligand"].t)

        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(rec_pos / cross_distance_cutoff[data['receptor'].batch],
                                lig_pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(rec_pos, lig_pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = rec_pos[dst.long()] - lig_pos[src.long()]

        edge_length_emb = self.cross_dist_expansion(edge_vec.norm(dim=-1))
        edge_t_emb = t_emb[src.long()]
        edge_attr = torch.cat([edge_length_emb, edge_t_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_graph(self, data, pos_to_use: str = "current"):
        if pos_to_use == "final":
            pos = data["ligand"].pos_T
            t_emb = self.timestep_emb_fn(data["ligand"].x.new_ones(pos.size(0), 1))

        elif pos_to_use == "current":
            pos = data["ligand"].pos_t
            t_emb = self.timestep_emb_fn(data["ligand"].t)

        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0), 
                                data['ligand'].batch.unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        src, dst = edge_index
        edge_vec = pos[src] - center_pos[dst]

        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = t_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
