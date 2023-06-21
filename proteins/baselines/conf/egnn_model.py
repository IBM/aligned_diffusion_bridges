import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add
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


class EGNNLayer(nn.Module):

    def __init__(self, input_fdim, out_fdim, edge_fdim, dropout_p):
        super().__init__()
        self.input_fdim = input_fdim
        self.out_fdim = out_fdim
        self.edge_fdim = edge_fdim
        self.dropout_p = dropout_p

        self._build_components()

    def _build_components(self):
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * self.input_fdim, self.input_fdim),
            nn.ReLU(),
            nn.Linear(self.input_fdim, self.out_fdim),
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * self.input_fdim + 1 + self.edge_fdim, self.input_fdim),
            nn.ReLU(),
            nn.Linear(self.input_fdim, self.input_fdim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(self.input_fdim, self.input_fdim),
            nn.ReLU(),
            nn.Linear(self.input_fdim, 3) 
        )

    def forward(self, x, edge_index, pos, pos_init, edge_attr=None, aggr='mean'):

        edge_src, edge_dst = edge_index
        d_ij_sq = ((pos[edge_src] - pos[edge_dst])**2).sum(dim=1, keepdims=True)
        d_ij_init = torch.sqrt(((pos_init[edge_src] - pos_init[edge_dst])**2).sum(dim=1, keepdims=True))

        m_ij = self.message_mlp(torch.cat([x[edge_src], x[edge_dst], d_ij_sq, edge_attr], dim=1))
        m_i = scatter(src=m_ij, index=edge_dst, dim=0, dim_size=x.size(0), reduce=aggr)
        h = self.node_mlp(torch.cat([x, m_i], dim=1))

        pos_ij_update = (pos_init[edge_src] - pos_init[edge_dst]) * self.coord_mlp(m_ij) / d_ij_init
        pos_i_update = scatter_add(pos_ij_update, index=edge_dst, dim=0, dim_size=x.size(0))

        return h, pos_i_update
    

class EGNNModel(nn.Module):

    def __init__(self, 
            node_fdim: int, 
            edge_fdim: int, 
            h_dim: int,
            n_conv_layers: int = 2,
            max_radius: float = 10.0, 
            max_neighbors: int = 24,
            distance_emb_dim: int = 32, 
            max_distance: float = 30.0,
            dropout_p: float = 0.2,**kwargs
        ):  
            super().__init__(**kwargs)

            self.node_fdim = node_fdim
            self.edge_fdim = edge_fdim
            self.h_dim = h_dim
            self.max_radius = max_radius
            self.max_neighbors = max_neighbors

            self.n_conv_layers = n_conv_layers

            dim_seq = [
                 h_dim, h_dim * 2, h_dim * 4, h_dim * 2, h_dim
            ]

            self.node_embedding = nn.Sequential(
                nn.Linear(node_fdim, self.h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(self.h_dim, self.h_dim)
            )
            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_fdim + distance_emb_dim, self.h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(self.h_dim, self.h_dim)
            )

            self.dist_expansion = GaussianSmearing(start=0.0, stop=max_distance, 
                                                       num_gaussians=distance_emb_dim)

            conv_layers = []
            for i in range(n_conv_layers):
                input_fdim = dim_seq[min(i, len(dim_seq)-1)] 
                out_fdim = dim_seq[min(i+1, len(dim_seq)-1)]

                conv_layer = EGNNLayer(
                     input_fdim=input_fdim,
                     out_fdim=out_fdim, 
                     edge_fdim=self.h_dim,
                     dropout_p=dropout_p
                 )
                
                conv_layers.append(conv_layer)

            self.conv_layers = nn.ModuleList(conv_layers)
            

    def forward(self, data):
        x, edge_index, edge_attr, pos_init = self.build_graph(data)
        pos = pos_init

        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for i in range(self.n_conv_layers):
              x, pos = self.conv_layers[i](x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, pos_init=pos_init)

        return pos
    
    def build_graph(self, data):
        node_attr = data.x
        pos = data.pos_0

        edge_index = radius_graph(
            x=pos, r=self.max_radius, max_num_neighbors=self.max_neighbors,
            batch=data.batch
        )

        src, dst = edge_index
        edge_vec = pos[dst.long()] - pos[src.long()]

        edge_attr = self.dist_expansion(edge_vec.norm(dim=-1))
        return node_attr, edge_index, edge_attr, pos



def build_model_from_args(args):

    model = EGNNModel(
         node_fdim=args.node_fdim,
         edge_fdim=args.edge_fdim,
         h_dim=args.h_dim,
         n_conv_layers=args.n_conv_layers,
         max_radius=args.max_radius,
         max_distance=args.max_distance,
         max_neighbors=args.max_neighbors,
         distance_emb_dim=args.distance_emb_dim,
         dropout_p=args.dropout_p
    )

    return model    
