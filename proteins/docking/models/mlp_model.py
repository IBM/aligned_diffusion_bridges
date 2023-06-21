import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Callable
from torch_cluster import radius, radius_graph
from torch_geometric.nn import MessagePassing

from sbalign.utils.helper import print_statistics
from sbalign.models.common import build_mlp
from sbalign.utils.sb_utils import get_timestep_embedding


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
    


class ConvLayer(MessagePassing):

    def __init__(self, 
                lig_node_fdim: int,
                lig_edge_fdim: int,
                rec_node_fdim: int,
                rec_edge_fdim: int,
                h_dim: int,
                dropout_p: float = 0.1,
                leakyrelu_slope: float = 0.01,
                activation: str = 'relu',
                beta_x: float = 1.0,
                aggr: str = 'mean',
                debug: bool = False,
                **kwargs):
        super().__init__(aggr=aggr, kwargs=kwargs)

        self.lig_node_fdim = lig_node_fdim
        self.lig_edge_fdim = lig_edge_fdim
        self.rec_node_fdim = rec_node_fdim
        self.rec_edge_fdim = rec_edge_fdim

        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self.activation = activation
        self.beta_x = beta_x
        self.aggr = aggr
        self.debug = debug

        message_in_dim = 2 * self.lig_node_fdim + self.lig_edge_fdim
        self.message_mlp = build_mlp(in_dim=message_in_dim, h_dim=self.h_dim, n_layers=1, 
                                     out_dim=self.h_dim, dropout_p=self.dropout_p, 
                                     activation=self.activation)

        self.accl_mlp = build_mlp(in_dim=self.h_dim, h_dim=2 * self.h_dim, out_dim=1, 
                                  activation=self.activation, dropout_p=self.dropout_p)

        self.node_mlp = build_mlp(in_dim=self.h_dim * 2, h_dim=2 * self.h_dim, 
                                  out_dim=self.h_dim, dropout_p=self.dropout_p, 
                                  activation=self.activation)


    def forward(self, x, edge_index, edge_attr, pos):
        src, dst = edge_index
        x_out, accl_out = self.propagate(edge_index=edge_index,
                                         x=x, edge_attr=edge_attr,
                                         pos=pos)

        return x_out, edge_index, edge_attr, accl_out

    def propagate(self, edge_index, size=None, **kwargs):
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)
        
        # ------- Computing intra and inter messages ------------

        # Message from ligand neighbors
        intra_m_ij = self.message(**msg_kwargs)

        if self.debug:
            print_statistics(tensor=intra_m_ij, prefix="Intra: ")
        
        # ------ Updating node acceleration -------
        
        # Acceleration weights based on messages from ligand neighbors
        accl_intra_wij = self.accl_mlp(intra_m_ij)
        src, dst = edge_index
        rel_vec = kwargs['pos'][src] - kwargs['pos'][dst]
        rel_dist = torch.sqrt((rel_vec ** 2).sum(dim=-1, keepdim=True))

        # TODO: Consider a new update strategy with initial acceleration
        # accl_j = kwargs["accl"][src]
        # accl_wij = (rel_vec * accl_j).sum(dim=-1, keepdims=True)
        # accl_nei = m_wij * accl_wij * accl_j / rel_dist

        # Acceleration along radial unit vectors
        accl_intra_nei = accl_intra_wij * rel_vec / rel_dist
        accl_intra_update = self.aggregate(accl_intra_nei, **aggr_kwargs)
        accl_out = kwargs["pos"] + accl_intra_update

        if self.debug:
            print_statistics(tensor=accl_intra_wij, prefix="Accl Weights: ")
            print_statistics(tensor=rel_dist, prefix="Relative Distance: ")
            print_statistics(tensor=accl_intra_nei, prefix="Weighted Acceleration Neighbors: ")
            print_statistics(tensor=accl_out, prefix="Accl Out: ")

        # ------ Updating node features --------

        intra_i = self.aggregate(intra_m_ij, **aggr_kwargs)

        x_update = self.node_mlp(torch.cat([kwargs['x'], intra_i], dim=-1))
        x_out = (1.0 - self.beta_x) * kwargs['x'] + self.beta_x * x_update

        if self.debug:
            print_statistics(tensor=x_update, prefix="x update: ")
        return x_out, accl_out

    def message(self, x_i, x_j, edge_attr):
        intra_msg = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return intra_msg


class DockBase(nn.Module):

    def __init__(self,
                 h_dim: int,
                 n_conv_layers: int = 3,
                 dropout_p: float = 0.2,
                 shared_layers: bool = False,
                 leakyrelu_slope: float = 0.01,
                 activation: str = 'relu',
                 aggr: str = "mean",
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.h_dim = h_dim
        self.n_conv_layers = n_conv_layers
        self.dropout_p = dropout_p
        self.activation = activation
        self.leakyrelu_slope = leakyrelu_slope
    
        self.shared_layers = shared_layers
        self.aggr = aggr
        self.debug = debug

        self._build_layers()

    def _build_layers(self):
        conv_layers = []
        if not self.shared_layers:
            for i in range(self.n_conv_layers):
                conv_layers.append(ConvLayer(
                    lig_node_fdim=self.h_dim, lig_edge_fdim=self.h_dim,
                    rec_node_fdim=self.h_dim, rec_edge_fdim=self.h_dim,
                    h_dim=self.h_dim, dropout_p=self.dropout_p,
                    activation=self.activation,
                    leakyrelu_slope=self.leakyrelu_slope,
                    aggr=self.aggr, debug=self.debug
                ))
        else:
            conv_layer = ConvLayer(
                    lig_node_fdim=self.h_dim, lig_edge_fdim=self.h_dim,
                    rec_node_fdim=self.h_dim, rec_edge_fdim=self.h_dim,
                    h_dim=self.h_dim, dropout_p=self.dropout_p,
                    activation=self.activation,
                    leakyrelu_slope=self.leakyrelu_slope,
                    aggr=self.aggr, debug=self.debug
                )
            
            for i in range(self.n_conv_layers):
                conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, lig_graph):

        for idx, layer in enumerate(self.conv_layers):
            lig_graph = layer(*lig_graph)
        
        lig_accl = lig_graph[-1]
        return lig_accl


class DockMLPModel(nn.Module):

    def __init__(self,
                h_dim: int,
                node_fdim: int,
                edge_fdim: int = 0,
                n_conv_layers: int = 3,
                lig_max_radius: float = 10.0,
                rec_max_radius: float = 10.0,
                lig_max_neighbors: int = 32,
                rec_max_neighbors: int = 32,
                distance_embed_dim: int = 32,
                timestep_embed_dim: int = 32, 
                dropout_p: float = 0.2,
                activation: str = 'relu',
                leakyrelu_slope: float = 0.01,
                timestep_emb_fn: Callable = None,
                shared_layers: bool = False,
                flexible_rec: bool = False,
                aggr: str = "mean",
                debug: bool = False,
                **kwargs
        ):
        super().__init__(**kwargs)

        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.lig_max_neighbors = lig_max_neighbors
        self.rec_max_neighbors = rec_max_neighbors
        self.dropout_p = dropout_p
        self.aggr = aggr
        self.debug = debug

        self.distance_embed_dim = distance_embed_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.timestep_emb_fn = timestep_emb_fn
        self.flexible_rec = flexible_rec

        self.lig_node_fdim = node_fdim + timestep_embed_dim
        self.lig_edge_fdim = edge_fdim + timestep_embed_dim + distance_embed_dim
        self.rec_node_fdim = node_fdim + timestep_embed_dim
        self.rec_edge_fdim = edge_fdim + timestep_embed_dim + distance_embed_dim
        self.h_dim = h_dim
        self.activation = activation
        self.leakyrelu_slope = leakyrelu_slope

        # Ligand
        self.lig_node_embedding = nn.Linear(self.lig_node_fdim, self.h_dim)

        self.lig_dist_expansion = GaussianSmearing(start=0.0, stop=self.lig_max_radius,
                                                   num_gaussians=self.distance_embed_dim)
        self.rec_dist_expansion = GaussianSmearing(start=0.0, stop=self.rec_max_radius,
                                                   num_gaussians=self.distance_embed_dim)

        self.lig_edge_embedding = nn.Linear(self.lig_edge_fdim, self.h_dim)

        # Receptor
        self.rec_node_embedding = nn.Linear(self.rec_node_fdim, self.h_dim)
        self.rec_edge_embedding = nn.Linear(self.rec_edge_fdim, self.h_dim)

        self.drift_model = nn.Sequential(
            nn.Linear(self.h_dim + 3 + self.timestep_embed_dim, 2*self.h_dim),
            nn.Dropout(p=dropout_p),
            nn.SiLU(),
            nn.Linear(2 * self.h_dim, 3)
        )

        self.doobs_score = nn.Sequential(
            nn.Linear(self.h_dim + 6 + self.timestep_embed_dim, 2*self.h_dim),
            nn.Dropout(p=dropout_p),
            nn.SiLU(),
            nn.Linear(2 * self.h_dim, 3)
        )

    def forward(self, data, eval_at_bound: bool = False):
        lig_x, _, _, lig_pos = self.build_lig_graph(data)
        lig_x = self.lig_node_embedding(lig_x)

        t_emb = self.timestep_emb_fn(data["ligand"].t)

        lig_inputs = torch.cat([lig_x, lig_pos, t_emb], dim=-1)
        lig_drift = self.drift_model(lig_inputs)

        lig_doobs_inputs = torch.cat([lig_inputs, lig_drift], dim=-1)
        lig_doobs = self.doobs_score(lig_doobs_inputs)

        if eval_at_bound:
            lig_x_T, _, _, lig_pos_T = self.build_lig_graph(data, final_time=True)
            lig_x_T = self.lig_node_embedding(lig_x_T)

            t_emb = self.timestep_emb_fn(data["ligand"].new_ones(lig_x.size(0), 1))

            lig_inputs_T = torch.cat([lig_x_T, lig_pos_T, t_emb], dim=-1)
            lig_drift_T = self.drift_model(lig_inputs_T)

            lig_doobs_inputs_T = torch.cat([lig_inputs_T, lig_drift_T], dim=-1)
            lig_doobs_T = self.doobs_score(lig_doobs_inputs_T)

            return lig_drift, lig_doobs, lig_doobs_T

        return lig_drift, lig_doobs, None

    def build_lig_graph(self, data, final_time: bool = False):
        # Node Features
        if not final_time:
            t_emb = self.timestep_emb_fn(data["ligand"].t)
        else:
            shape = (data["ligand"].x.size(0), 1)
            t_emb = self.timestep_emb_fn(torch.ones(shape, device=data["ligand"].x.device))

        node_attr = torch.cat([data["ligand"].x, t_emb], dim=1)

        # Dynamic ligand graph
        if not final_time:
            edge_index = radius_graph(data["ligand"].pos_t, 
                                      r=self.lig_max_radius,
                                      max_num_neighbors=self.lig_max_neighbors, 
                                      batch=data['ligand'].batch)
            src, dst = edge_index
            edge_vec = data['ligand'].pos_t[dst.long()] - data['ligand'].pos_t[src.long()]
        else:
            edge_index = radius_graph(data["ligand"].pos_T, 
                                      r=self.lig_max_radius,
                                      max_num_neighbors=self.lig_max_neighbors, 
                                      batch=data['ligand'].batch)
            src, dst = edge_index
            edge_vec = data['ligand'].pos_T[dst.long()] - data['ligand'].pos_T[src.long()]

        # Edge features
        edge_attr = None
        if "edge_attr" in data["ligand"]:
            edge_attr = data["ligand"].edge_attr
        edge_t_emb = t_emb[edge_index[0].long()]
        edge_length_emb = self.lig_dist_expansion(edge_vec.norm(dim=-1))

        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_t_emb, edge_length_emb], dim=-1)
        else:
            edge_attr = torch.cat([edge_t_emb, edge_length_emb], dim=-1)

        if not final_time:
            pos = data["ligand"].pos_t
        else:
            pos = data["ligand"].pos_T

        return node_attr, edge_index, edge_attr, pos

    def run_drift(self, data):
        lig_x, _, _, lig_pos = self.build_lig_graph(data)
        lig_x = self.lig_node_embedding(lig_x)

        t_emb = self.timestep_emb_fn(data["ligand"].t)

        lig_inputs = torch.cat([lig_x, lig_pos, t_emb], dim=-1)
        lig_drift = self.drift_model(lig_inputs)

        return lig_drift
