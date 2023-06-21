import torch
import wandb
import numpy as np

from sbalign.utils.ops import to_numpy


def count_parameters(model, log_to_wandb: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if log_to_wandb:
        wandb.log({'n_params': n_params})
    return n_params


def is_protein_dataset(opt):
    return opt in ['']


def is_shape_dataset(opt):
    return opt in ['']

def is_cell_dataset(opt):
    return opt in ['statephate', 'statephate_inverse']

def is_toy_dataset(opt):
    return opt in ['gmm', 'checkerboard', 'moon', 'spiral', "matching_with_exception", "diagonal_matching", "diagonal_matching_inverse"]


def print_statistics(tensor, prefix="", dim='all'):
    if dim == 'all':
        stats_str = f"Min={np.round(tensor.min().item(), 4)}, Max={np.round(tensor.max().item(), 4)}, Mean={np.round(tensor.mean().item(), 4)}, Std={np.round(tensor.std().item(), 4)}"
    else:
        np.set_printoptions(precision=4)
        stats_str = f"Min={to_numpy(tensor.min(dim=dim)[0])} \
                      Max={to_numpy(tensor.max(dim=dim)[0])} \
                      Mean={to_numpy(tensor.mean(dim=dim))} \
                      Std={to_numpy(tensor.std(dim=dim))}"

    print(prefix + stats_str, flush=True)
