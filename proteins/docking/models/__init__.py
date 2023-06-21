from proteins.docking.models.mlp_model import DockMLPModel
from proteins.docking.models.tensor_prod_model import DockTensorProductModel
from sbalign.utils.sb_utils import get_timestep_embedding


def build_model_from_args(args):

    timestep_embed_fn = get_timestep_embedding(embedding_type=args.timestep_embed_type,
                                               embedding_dim=args.timestep_embed_dim)

    if args.model_name == "mlp":

        model = DockMLPModel(h_dim=args.h_dim, node_fdim=args.node_fdim,
                        edge_fdim=args.edge_fdim, 
                        n_conv_layers=args.n_conv_layers, 
                        lig_max_radius=args.lig_max_radius,
                        rec_max_radius=args.rec_max_radius,
                        lig_max_neighbors=args.lig_max_neighbors,
                        rec_max_neighbors=args.rec_max_neighbors,
                        distance_embed_dim=args.distance_embed_dim,
                        timestep_embed_dim=args.timestep_embed_dim,
                        timestep_emb_fn=timestep_embed_fn,
                        shared_layers=args.shared_layers,
                        flexible_rec=False, aggr=args.aggr,
                        activation=args.activation,
                        leakyrelu_slope=args.leakyrelu_slope,
                        dropout_p=args.dropout_p, debug=False)

    elif args.model_name == "tensor-prod":

        model = DockTensorProductModel(
                timestep_emb_fn=timestep_embed_fn,
                node_fdim=args.node_fdim,
                edge_fdim=args.edge_fdim,
                sh_lmax=args.sh_lmax, n_s=args.n_s, n_v=args.n_v,
                n_conv_layers=args.n_conv_layers,
                lig_max_radius=args.lig_max_radius,
                rec_max_radius=args.rec_max_radius,
                lig_max_neighbors=args.lig_max_neighbors,
                rec_max_neighbors=args.rec_max_neighbors,
                distance_emb_dim=args.distance_embed_dim,
                timestep_emb_dim=args.timestep_embed_dim,
                dropout_p=args.dropout_p,
                cross_max_distance=args.cross_max_distance,
                cross_dist_emb_dim=args.cross_dist_emb_dim,
                center_max_distance=args.center_max_distance
        )

    else:
        raise ValueError(f"Model of type {args.model_name} is not supported.")
                
    return model
