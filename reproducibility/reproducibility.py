__author__ = "Matteo Pariset"

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
sns.set_context(context='talk', font_scale=.9)
palette = ['#1A254B', '#114083', '#A7BED3', '#F2545B', '#A4243B']

cmap = LinearSegmentedColormap.from_list('cmap', palette, N=18)

colors = ['#1A254B', '#114083', '#A7BED3', '#FFFFFF', '#F2545B', '#A4243B']
from matplotlib.colors import LinearSegmentedColormap
bcmap = LinearSegmentedColormap.from_list('bcmap', colors, N=100)

import os
import torch
import numpy as np
import pandas as pd
import datetime
from omegaconf import OmegaConf
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import joblib

from sbalign.models.base import AlignedSB

from sbalign.utils.helper import is_toy_dataset, is_cell_dataset
from sbalign.utils.setup import wandb_setup, parse_train_args, update_args_from_config
from sbalign.utils.definitions import DEVICE
from sbalign.training.diffusivity import get_diffusivity_schedule
from sbalign.data.datasets import SyntheticDataset, SavedDataset
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.utils.ops import to_numpy
from sbalign.utils.sampling import sampling


def load_config(experiment_name):
    return OmegaConf.load(f"../reproducibility/configs/{experiment_name}.yaml")


def load_model(filename, args):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    print(f"Args: {args}", flush=True)

    # Model
    model = AlignedSB(
        timestep_emb_dim=args.timestep_emb_dim,
        n_layers=args.n_layers, 
        in_dim=args.in_dim, out_dim=args.out_dim,
        h_dim=args.h_dim,
        activation=args.activation, 
        dropout_p=args.dropout_p, use_drift_in_doobs=args.use_drift_in_doobs)

    model.to(DEVICE)

    model.load_state_dict(torch.load(filename))

    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity)

    return model, g


class AlignExperiment:

    def __init__(self, config, model, diffusivity):
        self.config = config
        self.model = model
        self.diffusivity = diffusivity
    
    def run(cmd_args):
        # TODO: Debug. Deplace
        import sys
        sys.path.append("../scripts/")
        import train

        list_args = list(filter(lambda x: len(x) > 0, cmd_args.split(" ")))

        # Append automatic values of data_dir and log_di
        config = vars(parse_train_args(list_args))
        list_args += ["--data_dir="+config["dataset"], "--log_dir="+os.path.join("../reproducibility/", config["dataset"], "model"), "--run_name="+"".join(np.random.choice(list("qwertyuiopasdfghjklzxcvbnm"), size=(9,)))]
        config = OmegaConf.create(vars(parse_train_args(list_args)))

        model = train.main(list_args)

        g = get_diffusivity_schedule(config.diffusivity_schedule, config.max_diffusivity)

        # Default save
        time_tag = datetime.datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

        # TODO: Debug. Re-enable auto saving
        return AlignExperiment(config, model, g) #.save(time_tag)
        

    def save(self, tag):
        OmegaConf.save(self.config, f"../reproducibility/configs/{tag}.yaml", resolve=True)

        return self

    def get_model_path(config):
        return os.path.join("../reproducibility/", config.log_dir, config.run_name, "best_ema_model.pt")

    def load(tag):
        experiment_conf = load_config(tag)
        model_filename = AlignExperiment.get_model_path(experiment_conf)

        return AlignExperiment(experiment_conf, *load_model(model_filename, experiment_conf))

    def sample(self, input_data=None, samples_num=500, mode=None, apply_score=False, trials_num=1, inference_steps=None, t_max=1.):
        if inference_steps is None:
            inference_steps = self.config.inference_steps
        
        marginals = self.get_marginals(mode, samples_num)

        t_schedule = torch.from_numpy(get_t_schedule(inference_steps, t_max=t_max)).float()

        if input_data is None:
            input_data = marginals['initial'].to(DEVICE)
    
        trajs = np.stack([sampling(input_data, self.model, self.diffusivity, inference_steps, t_schedule, return_traj=True, apply_score=apply_score) for trial in range(trials_num)], axis=0).mean(axis=0)

        return trajs

    def get_marginals(self, mode=None, samples_num=500):
        if is_toy_dataset(self.config.dataset):
            if mode is not None:
                raise ValueError(f"Cannot choose marginals mode ({mode}) for toy datasets")
            return SyntheticDataset(root=os.path.join("../reproducibility/", self.config.log_dir), problem=self.config.dataset, split_fracs=(samples_num, 0, 0), n_samples=samples_num).data
        elif is_cell_dataset(self.config.dataset):
            if mode is None or mode not in ['train', 'val', 'test']:
                raise ValueError(f"Must specify mode (train, val, test) to use to extract marginals")
            return SavedDataset(self.config.dataset, n_samples=samples_num, mode=mode, root_dir="../reproducibility/").data
        else:
            raise ValueError(f"Unable to obtain marginals for dataset: {self.config.dataset}")

    def export_drift(self, save_dir, save_name):
        drift_config = {
            'timestep_emb_dim': self.config.timestep_emb_dim,
            'n_layers': self.config.n_layers, 
            'in_dim': self.config.in_dim,
            'out_dim': self.config.out_dim,
            'h_dim': self.config.h_dim,
            'activation': self.config.activation,
            'dropout_p': self.config.dropout_p,
        }

        with open(os.path.join(save_dir, f"{save_name}_config.json"), 'w') as config_file:
            json.dump(drift_config, config_file)
        torch.save(self.model.sde_drift.state_dict(), os.path.join(save_dir, f"{save_name}_checkpoint.pt"))

        return self


##############################
# Sampling                   #
##############################

def expand_t(x, t):
    return torch.ones(x.shape[0]).to(DEVICE) * t

def compute_dx(g, drift_vec, x, t, dt, brownian_motion=None):
    dx = np.square(g(t)) * drift_vec * dt 

    if brownian_motion is None:  
        brownian_motion =  g(t) * torch.randn_like(x) * torch.sqrt(dt)

    dx = dx + brownian_motion

    return dx
def sampling_double(pos_0, model_direct, model_inverse, g, inference_steps, t_schedule, apply_score=False, return_traj: bool=False):
    pos = pos_0.clone()

    model_direct.eval()
    model_inverse.eval()

    trajectory = np.zeros((inference_steps+1, *pos_0.shape))
    trajectory[0] = pos.cpu()

    dt = t_schedule[1] - t_schedule[0]

    if apply_score:
        assert False, "Must pass x_T as parameter of the function"

    with torch.no_grad():
        for t_idx in range(1, inference_steps+1):
            t_before, t_after = t_schedule[t_idx-1], t_schedule[min(t_idx, inference_steps)]

            drift_inverse = -model_inverse.run_drift(pos, expand_t(pos, 1-t_before))
            drift_direct = model_direct.run_drift(pos, expand_t(pos, t_before))

            corrected_drift = (1-t_before) * drift_direct + t_before * drift_inverse

            pos = pos + compute_dx(g, corrected_drift, pos, t_before, dt)

            trajectory[t_idx] = pos.cpu()

    if return_traj:
        return trajectory
    else:
        return return_traj[-1]
    

def sample_double(self, other, input_data=None, samples_num=500, mode=None, apply_score=False, trials_num=1, inference_steps=None, t_max=1.):
    if inference_steps is None:
        inference_steps = self.config.inference_steps
    
    marginals = self.get_marginals(mode, samples_num)

    t_schedule = torch.from_numpy(get_t_schedule(inference_steps, t_max=t_max)).float()

    if input_data is None:
        input_data = marginals['initial'].to(DEVICE)

    trajs = np.stack([sampling_double(input_data, self.model, other.model, self.diffusivity, inference_steps, t_schedule, return_traj=True, apply_score=apply_score) for trial in range(trials_num)], axis=0).mean(axis=0)

    return trajs

def sampling_inverse(pos_0, model_inverse, g, inference_steps, t_schedule, apply_score=False, return_traj: bool=False):
    pos = pos_0.clone()

    model_inverse.eval()

    trajectory = np.zeros((inference_steps+1, *pos_0.shape))
    trajectory[0] = pos.cpu()

    dt = t_schedule[1] - t_schedule[0]

    if apply_score:
        assert False, "Must pass x_T as parameter of the function"

    with torch.no_grad():
        for t_idx in range(1, inference_steps+1):
            t_before = 1 - t_schedule[t_idx]

            brownian_motion = g(t_before) * torch.randn_like(pos) * torch.sqrt(dt)

            drift_inverse = -model_inverse.run_drift(pos, expand_t(pos, t_before))

            pos = pos + compute_dx(g, drift_inverse, pos, t_before, dt, brownian_motion=brownian_motion)

            trajectory[t_idx] = pos.cpu()

    if return_traj:
        return trajectory
    else:
        return return_traj[-1]

def sample_inverse(self, input_data=None, samples_num=500, mode=None, apply_score=False, trials_num=1, inference_steps=None, t_max=1.):
    if inference_steps is None:
        inference_steps = self.config.inference_steps
    
    marginals = self.get_marginals(mode, samples_num)

    t_schedule = torch.from_numpy(get_t_schedule(inference_steps, t_max=t_max)).float()

    if input_data is None:
        input_data = marginals['initial'].to(DEVICE)

    trajs = np.stack([sampling_inverse(input_data, self.model, self.diffusivity, inference_steps, t_schedule, return_traj=True, apply_score=apply_score) for trial in range(trials_num)], axis=0).mean(axis=0)

    return trajs


def mix_trajs(traj_direct, traj_inverse):
    timesteps = np.linspace(0, 1., traj_direct.shape[0], endpoint=True)
    timesteps = np.expand_dims(timesteps, axis=(1, 2))

    return (1-timesteps) * traj_direct + timesteps * traj_inverse


def t0_points_path(dataset_name, mode):
    return os.path.abspath(os.path.join("../reproducibility/", dataset_name, "data", f"{dataset_name}_embs_initial_{mode}.npy"))

def t1_points_path(dataset_name, mode):
    return os.path.abspath(os.path.join("../reproducibility/", dataset_name, "data", f"{dataset_name}_embs_final_{mode}.npy"))


##############################
# Visualizations             #
##############################

def plot_marginals(marginals, t0_color=bcmap(.2), t1_color=bcmap(.8), projection=lambda x: x, **kwargs):
    kwargs['alpha'] = kwargs.get('alpha', .4)
    plt.scatter(*projection(marginals['initial']).T, color=t0_color, label=r"$t_0$", **kwargs)
    plt.scatter(*projection(marginals['final']).T, color=t1_color, label=r"$t_1$", **kwargs)
    plt.axis("equal")
    plt.legend();

def plot_multiple_marginals(xs, skip_step=1, projection=lambda x: x, labels=None, save="", **kwargs):
    kwargs["s"] = kwargs.get("s", .5)
    last_idx = xs.shape[0]//skip_step-1
    for idx, time_frame in enumerate(xs[::skip_step]):
        timeframe_kwargs = kwargs.copy()
        if labels is not None and idx == 0:
            timeframe_kwargs['label'] = labels[0]
        elif labels is not None and idx == last_idx:
            timeframe_kwargs['label'] = labels[1]
        
        if save != "":
            plt.clf()
        
        plt.scatter(*projection(time_frame).T, color=bcmap(idx/(xs.shape[0]//max(skip_step-1, 1))), **timeframe_kwargs)

        if save != "":
            plt.savefig(save + str(time_frame) + ".png", dpi=400)
    plt.legend()

def plot_matchings(t0_points, t1_points, projection=lambda x: x, **kwargs):
    kwargs["color"] = kwargs.get("color", "gray")
    kwargs["alpha"] = kwargs.get("alpha", .7)
    kwargs["lw"] = kwargs.get("lw", .2)
    extended_coords = np.concatenate([projection(t0_points), projection(t1_points)], axis=1)
    plt.plot(extended_coords[:,::2].T, extended_coords[:,1::2].T, zorder=0, **kwargs);

def plot_predictions(preds, projection=lambda x: x):
    plt.scatter(*projection(preds[-1]).T, s=7.5, zorder=1, edgecolors="black", facecolors="black", label="predictions")

def in_experiment_dir(experiment_name, *args):
    return os.path.join(f"../reproducibility/{experiment_name}/", *args)

def export_fig(fig_name, force=False, extension="pdf"):
    fig_path = f"../figures/{fig_name}.{extension}"
    if not os.path.exists(fig_path) or force:
        plt.savefig(fig_path, bbox_inches="tight")