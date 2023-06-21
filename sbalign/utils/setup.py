import os
import argparse
from argparse import FileType
import yaml
import wandb

from sbalign.utils.definitions import IS_CLUSTER, EXP_DIR, CLUSTER_EXP_DIR, WANDB_ENTITY


# ------------ Wandb --------------------------

def wandb_setup(args):
    DIR = CLUSTER_EXP_DIR if IS_CLUSTER else EXP_DIR
    print(f"Supplied experiment directory: {DIR}", flush=True)

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    run_id = wandb.util.generate_id()
    
    if args.group_name is not None:
        args.run_name = args.group_name + f"-{run_id}"
    else:
        args.run_name = run_id
    
    print("Setting up wandb...", flush=True)
    wandb.init(
        id=run_id,
        project="sbalign",
        entity=args.wandb_entity,
        group=args.group_name,
        name=run_id,
        config=vars(args),
        dir=DIR,
        mode=args.wandb_mode,
        job_type=args.job_type
    )

# ------ Parsing ---------------

def int_or_float(s):
    # Very hacky for now
    if float(s) > 1: # Fractions
        return int(s)
    else: # Numbers
        return float(s)

# -------------------- Synthetic Data Args ---------------------

def parse_train_args(cmd_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to save data to.")
    parser.add_argument("--log_dir", help="Directory to save local logs to.")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")

    parser.add_argument("--task", default="synthetic")

   # wandb
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--job_type", default="dev-train")
    parser.add_argument("--online", default=False)
    
    # Data 
    parser.add_argument("--dataset", default="moon", type=str)
    parser.add_argument("--n_samples", default=10000, type=int)
    parser.add_argument("--train_bs", default=32, type=int, help="Batchsize used in training")
    parser.add_argument("--val_bs", default=8, type=int, help="Batch size used in testing")
    parser.add_argument("--transform", default=None)
    parser.add_argument("--split_fracs", nargs="+", type=int_or_float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--num_workers", type=int, default=1)

    # NN (Right now only for MLP)
    parser.add_argument("--in_dim", default=2, type=int, help="Input size")
    parser.add_argument("--out_dim", default=2, type=int, help="Output size")
    parser.add_argument("--timestep_emb_dim", default=32, type=int, help="Timestep embedding size")
    parser.add_argument("--h_dim", default=32, type=int, help="Hidden size")
    parser.add_argument("--n_layers", default=3, type=int, help='Number of MLP layers')
    parser.add_argument("--activation", default='relu', type=str, help="NN activation")
    parser.add_argument("--dropout_p", default=0.1, type=float, help="Dropout probability")
    
    # Training
    parser.add_argument("--run_name", default=None, type=str, help="Name of training run.")
    parser.add_argument("--n_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument("--use_grad_noise", action='store_true', help="Whether to use gradient noise during training")
    parser.add_argument("--drift_weight", default=1.0, type=float, help="Weight on the Loss term from drift matching")
    parser.add_argument("--reg_weight", default=1.0, type=float, help="Weight for Doobs-score regularizer")
    parser.add_argument("--reg_weight_T", default=1.0, type=float, help="Weight for Doobs-score regularizer at T")
    parser.add_argument("--reg_weight_t", default=1.0, type=float, help="Weight for Doobs-score regularizer at t")
    parser.add_argument("--diffusivity_schedule", default="constant", type=str, help="Choose how diffusivity varies in time")
    parser.add_argument("--max_diffusivity", default=1.0, type=float, help="Maximum value of diffusivity")
    parser.add_argument("--use_drift_in_doobs", default=False, type=bool, 
                        help="Whether to use the drift as input to the parametrization of Doobs score")

    # Optimizer & Scheduler
    parser.add_argument("--optim_name", default='adamw', type=str)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate used for training")
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--grad_clip_value", default=10.0, type=float, help="Gradient clipping max value")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_mode", type=str, default='min')  
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--ema_decay_rate", type=float, default=0.999)

    # Logging
    parser.add_argument("--log_every", default=1000, type=int, help="Logging frequency")
    parser.add_argument("--eval_every", default=10000, type=int, help="Evaluation frequency during training.")
    parser.add_argument("--inference_every", default=0, type=int)

    # Val Inference
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--inference_metric", default="val_rmsd", type=str)
    parser.add_argument("--inference_goal", default="min", type=str)

    parser.add_argument("--early_stop_metric", default="val_loss", type=str)
    parser.add_argument("--early_stop_goal", default="min", type=str)

    args = parser.parse_args(args=cmd_args)
    return args


def parse_inference_args():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    return args


# -------------------- Docking Specific Args ---------------------

def parse_docking_train_args(cmd_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to save data to.")
    parser.add_argument("--log_dir", help="Directory to save local logs to.")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")

   # wandb
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--job_type", default="dev-train")

    parser.add_argument("--task", default="docking")

    # Data
    parser.add_argument("--dataset", default="db5", type=str)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--lig_max_radius", default=10.0, type=int)
    parser.add_argument("--rec_max_radius", default=10.0, type=int)
    parser.add_argument("--lig_max_neighbors", default=24, type=int)
    parser.add_argument("--rec_max_neighbors", default=24, type=int)
    parser.add_argument("--resolution", default="c_alpha", type=str)
    parser.add_argument("--center_complex", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_bs", default=8, type=int)
    parser.add_argument("--val_bs", default=2, type=int)

    # Docking Model
    parser.add_argument("--node_fdim", default=145, type=int)
    parser.add_argument("--edge_fdim", default=0, type=int)
    parser.add_argument("--h_dim", default=32, type=int, help="Hidden size")
    parser.add_argument("--distance_embed_dim", default=32, type=int, help="Distance Embed dimension")
    parser.add_argument("--timestep_emb_dim", default=32, type=int, help="Timestep embedding size")
    parser.add_argument("--shared_layers", action='store_true')
    parser.add_argument("--n_conv_layers", default=3, type=int, help='Number of MLP layers')
    parser.add_argument("--activation", default='relu', type=str, help="NN activation")
    parser.add_argument("--dropout_p", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--timestep_embed_type", default="sinusoidal", type=str)
    parser.add_argument("--leakyrelu_slope", type=float, default=0.01)
    
    # Training
    parser.add_argument("--n_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument("--use_grad_noise", action='store_true', help="Whether to use gradient noise during training")
    parser.add_argument("--drift_weight", default=1.0, type=float, help="Weight on the Loss term from drift matching")
    parser.add_argument("--reg_weight_T", default=1.0, type=float, help="Weight for Doobs-score regularizer at T")
    parser.add_argument("--reg_weight_t", default=1.0, type=float, help="Weight for Doobs-score regularizer at t")
    parser.add_argument("--diffusivity_schedule", default="constant", type=str, help="Choose how diffusivity varies in time")
    parser.add_argument("--max_diffusivity", default=1.0, type=float, help="Maximum value of diffusivity")
    
    # Optimizer & Scheduler
    parser.add_argument("--optim_name", default='adam', type=str)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate used for training")
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--grad_clip_value", default=10.0, type=float, help="Gradient clipping max value")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_mode", type=str, default='min')  
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--ema_decay_rate", type=float, default=0.999)

    # Logging
    parser.add_argument("--log_every", default=1000, type=int, help="Logging frequency")
    parser.add_argument("--eval_every", default=10000, type=int, help="Evaluation frequency during training.")
    parser.add_argument("--inference_every", default=0, type=int)

    # Val Inference
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--inference_metric", default="val_rmsd", type=str)
    parser.add_argument("--inference_goal", default="min", type=str)
    parser.add_argument("--samples_per_complex", default=10, type=int)

    parser.add_argument("--early_stop_metric", default="val_loss", type=str)
    parser.add_argument("--early_stop_goal", default="min", type=str)

    args = parser.parse_args(args=cmd_args)
    return args


# -------- Conf training specific args ----------------

def parse_conf_train_args(cmd_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to save data to.")
    parser.add_argument("--log_dir", help="Directory to save local logs to.")
    parser.add_argument("--config", type=FileType(mode='r'), 
                        help="Config file to load args from. args will be overwritten")
    
    parser.add_argument("--task", default="conf")

   # wandb
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY)
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--job_type", default="dev-train")

    # Data
    parser.add_argument("--dataset", default="db5", type=str)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--max_radius", default=10.0, type=int)
    parser.add_argument("--max_neighbors", default=24, type=int)
    parser.add_argument("--resolution", default="c_alpha", type=str)
    parser.add_argument("--center_conformations", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_bs", default=8, type=int)
    parser.add_argument("--val_bs", default=2, type=int)

    # Conformation Model
    parser.add_argument("--node_fdim", default=145, type=int)
    parser.add_argument("--edge_fdim", default=0, type=int)
    parser.add_argument("--h_dim", default=32, type=int, help="Hidden size")
    parser.add_argument("--distance_embed_dim", default=32, type=int, help="Distance Embed dimension")
    parser.add_argument("--timestep_emb_dim", default=32, type=int, help="Timestep embedding size")
    parser.add_argument("--shared_layers", action='store_true')
    parser.add_argument("--center_max_radius", type=float, default=30.0)
    parser.add_argument("--n_conv_layers", default=3, type=int, help='Number of MLP layers')
    parser.add_argument("--activation", default='relu', type=str, help="NN activation")
    parser.add_argument("--dropout_p", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--timestep_embed_type", default="sinusoidal", type=str)
    parser.add_argument("--leakyrelu_slope", type=float, default=0.01)
    
    # Training
    parser.add_argument("--n_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument("--use_grad_noise", action='store_true', help="Whether to use gradient noise during training")
    parser.add_argument("--drift_weight", default=1.0, type=float, help="Weight on the Loss term from drift matching")
    parser.add_argument("--reg_weight_T", default=1.0, type=float, help="Weight for Doobs-score regularizer at T")
    parser.add_argument("--reg_weight_t", default=1.0, type=float, help="Weight for Doobs-score regularizer at t")
    parser.add_argument("--diffusivity_schedule", default="constant", type=str, help="Choose how diffusivity varies in time")
    parser.add_argument("--max_diffusivity", default=1.0, type=float, help="Maximum value of diffusivity")
    
    # Optimizer & Scheduler
    parser.add_argument("--optim_name", default='adam', type=str)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate used for training")
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--grad_clip_value", default=10.0, type=float, help="Gradient clipping max value")
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler_mode", type=str, default='min')  
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--ema_decay_rate", type=float, default=0.999)

    # Logging
    parser.add_argument("--log_every", default=1000, type=int, help="Logging frequency")
    parser.add_argument("--eval_every", default=10000, type=int, help="Evaluation frequency during training.")
    parser.add_argument("--inference_every", default=0, type=int)

    # Val Inference
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--inference_metric", default="val_rmsd", type=str)
    parser.add_argument("--inference_goal", default="min", type=str)
    parser.add_argument("--samples_per_protein", default=10, type=int)

    parser.add_argument("--early_stop_metric", default="val_loss", type=str)
    parser.add_argument("--early_stop_goal", default="min", type=str)

    args = parser.parse_args(args=cmd_args)
    return args


def update_args_from_config(args):
    if args.config is not None:

        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        args_dict = args.__dict__

        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    args_dict[key].append(v)
            else:
                args_dict[key] = value

    args.config = None
    return args