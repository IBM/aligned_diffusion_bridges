import traceback
import argparse
from argparse import FileType

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np

from sbalign.data import ListDataset
from sbalign.utils.ops import to_numpy
from sbalign.training.epoch_fns import ProgressMonitor
from sbalign.utils.definitions import DEVICE, WANDB_ENTITY


def train_epoch(
    model, loader, 
    optimizer, loss_fn,
    grad_clip_value: float = None, 
    ema_weights=None):

    model.train()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        optimizer.zero_grad()

        try:
            data = data.to(DEVICE)
            pos_pred = model(data)

            loss, loss_dict = loss_fn(
                pos_pred=pos_pred,
                pos_true=data.pos_T
            )
                                    
            monitor.add(loss_dict)

            loss.backward()

            if grad_clip_value is not None:
                grad_clip_value = 10.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            
            if ema_weights is not None:
                ema_weights.update(model.parameters())
            
        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                traceback.print_exc()
                continue

    return monitor.summarize()


def test_epoch(model, loader, loss_fn):
    model.eval()
    monitor = ProgressMonitor()

    for idx, data in enumerate(loader):
        try:
            with torch.no_grad():
                data = data.to(DEVICE)
                pos_pred = model(data)

                loss, loss_dict = loss_fn(
                    pos_pred=pos_pred,
                    pos_true=data.pos_T
                )
                                        
                monitor.add(loss_dict)

        except Exception as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                print(e)
                continue

    return monitor.summarize()


def inference_epoch(model, orig_dataset, num_inference_proteins):
    model.eval()
    model.to(DEVICE)

    def rmsd_fn(y_pred, y_true):
        se = (y_pred - y_true)**2
        mse = se.sum(axis=1).mean()
        return np.sqrt(mse)

    dataset = ListDataset(
        processed_dir=orig_dataset.full_processed_dir, 
        id_list=orig_dataset.conf_pairs_split[:num_inference_proteins]
    )

    monitor = ProgressMonitor()
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for data in loader:
        data = data.to(DEVICE)
        pos_pred = model(data)

        rmsd = rmsd_fn(to_numpy(pos_pred), to_numpy(data.pos_T))
        init_rmsd = rmsd_fn(to_numpy(data.pos_0), to_numpy(data.pos_T))
        rmsd_dict = {'rmsd': np.round(rmsd, 4), 'init_rmsd': np.round(init_rmsd, 4)}
        monitor.add(rmsd_dict)

    return monitor.summarize()
    

def parse_train_args(cmd_args=None):

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
    parser.add_argument("--max_radius", type=float, default=30.0)
    parser.add_argument("--max_neighbors", type=int, default=40)
    parser.add_argument("--max_distance", type=float, default=30.0)
    parser.add_argument("--n_conv_layers", default=3, type=int, help='Number of MLP layers')
    parser.add_argument("--dropout_p", default=0.1, type=float, help="Dropout probability")
    
    # Training
    parser.add_argument("--n_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument("--use_grad_noise", action='store_true', help="Whether to use gradient noise during training")
    
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
    parser.add_argument("--inference_metric", default="val_rmsd", type=str)
    parser.add_argument("--inference_goal", default="min", type=str)

    parser.add_argument("--early_stop_metric", default="val_loss", type=str)
    parser.add_argument("--early_stop_goal", default="min", type=str)

    args = parser.parse_args(args=cmd_args)
    return args


def loss_fn_from_args(args):
    
    def loss_fn(pos_pred: torch.Tensor, pos_true: torch.Tensor):
        loss = nn.MSELoss()(pos_pred, pos_true)
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict
    
    return loss_fn
