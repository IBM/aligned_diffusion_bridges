from proteins.conf.models.tensor_prod_model import ConfTensorProductModel
from sbalign.utils.sb_utils import get_timestep_embedding


def build_model_from_args(args):

    timestep_embed_fn = get_timestep_embedding(embedding_type=args.timestep_embed_type,
                                               embedding_dim=args.timestep_embed_dim)

    if args.model_name == "tensor-prod":

        model = ConfTensorProductModel(
                timestep_emb_fn=timestep_embed_fn,
                node_fdim=args.node_fdim,
                edge_fdim=args.edge_fdim,
                sh_lmax=args.sh_lmax, n_s=args.n_s, n_v=args.n_v,
                n_conv_layers=args.n_conv_layers,
                max_radius=args.max_radius,
                max_neighbors=args.max_neighbors,
                distance_emb_dim=args.distance_embed_dim,
                timestep_emb_dim=args.timestep_embed_dim,
                dropout_p=args.dropout_p,
                max_distance=args.max_distance,
        )

    else:
        raise ValueError(f"Model of type {args.model_name} is not supported.")
                
    return model
