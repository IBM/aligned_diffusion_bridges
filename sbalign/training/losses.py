import torch
import torch.nn as nn
import numpy as np
from functools import partial

from sbalign.utils.sb_utils import beta, get_diffusivity_schedule
from sbalign.utils.definitions import DEVICE


# ----------- Schroedinger Bridges with Paired Samples -------------------

def loss_function_sbalign(
        drift_x_pred, 
        doobs_score_x_pred, 
        doobs_score_xT_pred,
        data, g, 
        steps_num, 
        drift_weight: float = 1.0,
        reg_weight_T: float = 1.0,
        reg_weight_t: float = 1.0,
        t_max: float=1.0
    ):
    
    assert data.t.max().item() <= t_max

    # t_diff = (1.0 - data.t) + 1e-8
    t_diff = (beta(g, 1, steps_num) - beta(g, data.t, steps_num)).to(DEVICE)

    x_diff = (data.pos_T - data.pos_t)

    bb_drift_true = (x_diff) / t_diff
    bb_drift_pred = drift_x_pred + doobs_score_x_pred

    criterion = nn.MSELoss()

    dt = 1/steps_num
    bb_loss = criterion(bb_drift_pred, bb_drift_true) * dt

    if doobs_score_xT_pred is not None:
        reg_loss_T = (doobs_score_xT_pred ** 2).sum(dim=-1).mean()
    else:
        reg_loss_T = torch.tensor(0.0, requires_grad=True)
    reg_loss_t = (doobs_score_x_pred ** 2).sum(dim=-1).mean()

    loss = drift_weight * bb_loss + reg_weight_T * reg_loss_T + reg_weight_t * reg_loss_t
    
    loss_dict = {
        "loss": loss.item(), 
        "bb_loss": bb_loss.item(),
        "reg_loss_T": reg_loss_T.item(), 
        "reg_loss_t": reg_loss_t.item()
    }

    for key, value in loss_dict.items():
        loss_dict[key] = np.round(value, 4)

    return loss, loss_dict


def loss_function_docking(
        drift_x_pred, 
        doobs_score_x_pred, 
        doobs_score_xT_pred,
        data, g, 
        steps_num, 
        drift_weight: float = 1.0, 
        reg_weight_T: float = 1.0, 
        reg_weight_t: float = 1.0, 
        t_max: float=1.0
    ):

    assert data["ligand"].t.max().item() <= t_max

    t_diff = (beta(g, 1, steps_num) - beta(g, data["ligand"].t, steps_num)).to(DEVICE)
    x_diff = (data["ligand"].pos_T - data["ligand"].pos_t)

    bb_drift_true = (x_diff) / t_diff
    bb_drift_pred = drift_x_pred + doobs_score_x_pred

    lambda_t = 1 / (bb_drift_true ** 2).sum(dim=1, keepdim=True)
    criterion = nn.MSELoss(reduction='none')
    bb_loss = criterion(bb_drift_pred, bb_drift_true)
    bb_loss = bb_loss.sum(dim=-1, keepdim=True) * lambda_t
    bb_loss = bb_loss.mean()

    if doobs_score_xT_pred is not None:
        reg_loss_T = (doobs_score_xT_pred ** 2).sum(dim=-1).mean()
    else:
        reg_loss_T = torch.tensor(0.0, requires_grad=True)
    reg_loss_t = (doobs_score_x_pred ** 2).sum(dim=-1).mean()

    loss = drift_weight * bb_loss + reg_weight_T * reg_loss_T + reg_weight_t * reg_loss_t
    
    loss_dict = {
        "loss": loss.item(), 
        "bb_loss": bb_loss.item(),
        "reg_loss_T": reg_loss_T.item(), 
        "reg_loss_t": reg_loss_t.item()
    }

    for key, value in loss_dict.items():
        loss_dict[key] = np.round(value, 4)

    return loss, loss_dict


def loss_function_conf(
        drift_x_pred, 
        doobs_score_x_pred, 
        doobs_score_xT_pred,
        data, g, 
        steps_num, 
        drift_weight: float = 1.0, 
        reg_weight_T: float = 1.0, 
        reg_weight_t: float = 1.0, 
        t_max: float=1.0, 
        apply_mean: bool = True
    ):
    
    mean_dims = (0, 1) if apply_mean else 1

    assert data.t.max().item() <= t_max

    beta_t_diff = (beta(g, 1, steps_num) - beta(g, data.t, steps_num)).to(DEVICE)
    x_diff = (data.pos_T - data.pos_t)

    bb_drift_true = (x_diff) / beta_t_diff
    bb_drift_pred = drift_x_pred + doobs_score_x_pred

    bb_loss = ((bb_drift_pred - bb_drift_true) ** 2).mean(mean_dims)

    if doobs_score_xT_pred is not None:
        reg_loss_T = (doobs_score_xT_pred ** 2).mean(mean_dims)
    else:
        reg_loss_T = torch.tensor(0.0, requires_grad=True)
    reg_loss_t = (doobs_score_x_pred ** 2).mean(mean_dims)

    loss = drift_weight * bb_loss + reg_weight_T * reg_loss_T + reg_weight_t * reg_loss_t
    
    loss_dict = {
        "loss": loss.item(), 
        "bb_loss": bb_loss.item(),
        "reg_loss_T": reg_loss_T.item(), 
        "reg_loss_t": reg_loss_t.item()
    }

    for key, value in loss_dict.items():
        loss_dict[key] = np.round(value, 4)

    return loss, loss_dict


def loss_fn_from_args(args):

    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity)

    if args.task == "synthetic":
        loss_fn_base = loss_function_sbalign
    elif args.task == "docking":
        loss_fn_base = loss_function_docking
    elif args.task == "conf":
        loss_fn_base = loss_function_conf

    loss_fn = partial(
        loss_fn_base,
        drift_weight=args.drift_weight, 
        reg_weight_T=args.reg_weight_T,
        reg_weight_t=args.reg_weight_t, g=g, 
        steps_num=args.inference_steps
    )

    return loss_fn
