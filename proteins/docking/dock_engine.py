import torch
import yaml
import numpy as np
from argparse import Namespace
import copy
from typing import Callable

from proteins.docking.models import build_model_from_args
from proteins.docking.metrics import compute_complex_rmsd, compute_interface_rmsd, aligned_rmsd, to_numpy

from sbalign.utils.sb_utils import get_t_schedule, get_diffusivity_schedule
from sbalign.utils.ops import axis_angle_to_matrix
from sbalign.utils.definitions import DEVICE


class DockingEngine:

    def __init__(self,
                 samples_per_complex: int,
                 inference_steps: int,
                 rot_vec: torch.Tensor,
                 tr_vec: torch.Tensor,
                 model_file: str = None,
                 config_file: str = None,
                 model: torch.nn.Module = None,
                 g_fn: Callable = None,
            ):
        self.samples_per_complex = samples_per_complex
        
        if model is None:
            with open(config_file) as f:
                model_args = Namespace(**yaml.full_load(f))

            model = build_model_from_args(model_args)
            model_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(model_dict)
        
        self.model = model.to(DEVICE)
        self.model.eval()
        self.inference_steps = inference_steps

        self.rot_vec = rot_vec
        self.tr_vec = tr_vec
        
        t_schedule = get_t_schedule(inference_steps=inference_steps)
        self.t_schedule = torch.from_numpy(t_schedule)
        self.dt = self.t_schedule[1] - self.t_schedule[0]

        if g_fn is None:
            g_fn = get_diffusivity_schedule(model_args.diffusivity_schedule,
                                            g_max=model_args.max_diffusivity)
        self.g_fn = g_fn
        
    def generate_sample(self, data):
        # Initialization (zeroing out the final position just to make sure)
        data["ligand"].pos_T = None
        data["ligand"].pos_t = data["ligand"].pos_0
        data["ligand"].pos_orig = data["ligand"].pos_0.clone()

        trajectory = []

        with torch.no_grad():
            for t_idx in range(self.inference_steps):
                t = self.t_schedule[t_idx]

                data["ligand"].t = t * torch.ones(data["ligand"].num_nodes).to(DEVICE)
                data["receptor"].t = t * torch.ones(data["receptor"].num_nodes).to(DEVICE)

                g_t = torch.tensor(self.g_fn(t)).float().to(data["ligand"].x.device)
                ligand_drift = self.model.run_drift(data)
                diffusion = g_t * torch.randn_like(data["ligand"].pos_t) * torch.sqrt(self.dt)

                dpos = torch.square(g_t) * ligand_drift * self.dt + diffusion
                pos_t = data["ligand"].pos_t  + dpos
                data["ligand"].pos_t = pos_t
                trajectory.append(pos_t)

        trajectory = torch.stack(trajectory, dim=0)
        return trajectory[-1], trajectory
    
    def setup_inference(self, data):
        rot_mat = axis_angle_to_matrix(self.rot_vec).squeeze(0)
        tr_vec = self.tr_vec

        pos_mean = torch.mean(data["ligand"].pos_T, dim=0, keepdims=True)
        data["ligand"].pos_0 = (rot_mat @ (data["ligand"].pos_T - pos_mean).T).T + pos_mean + tr_vec
        return data.to(DEVICE)
    
    def dock(self, data):
        data = self.setup_inference(data)

        ligand_poses = []
        trajectories  = []

        metrics = {'rmsd': [], 'i_rmsd': [], 'rmsd_unbound_bound': []}

        for sample_id in range(self.samples_per_complex):
            data_copy = copy.deepcopy(data)
            ligand_pose, trajectory = self.generate_sample(data_copy)
            trajectories.append(trajectory)

            ligand_poses.append(ligand_pose)
          
            complex_rmsd, _ = compute_complex_rmsd(ligand_pred=ligand_pose, 
                                                ligand_true=data["ligand"].pos_T,
                                                receptor_pred=data["receptor"].pos_T,
                                                receptor_true=data["receptor"].pos_T)

            interface_rmsd, _ = compute_interface_rmsd(ligand_pred=ligand_pose, 
                                                       ligand_true=data["ligand"].pos_T,
                                                       receptor_pred=data["receptor"].pos_T,
                                                       receptor_true=data["receptor"].pos_T)
            
            pos_0 = np.concatenate([to_numpy(data["ligand"].pos_0), to_numpy(data["receptor"].pos_T)], axis=0)
            pos_T = np.concatenate([to_numpy(data["ligand"].pos_T), to_numpy(data["receptor"].pos_T)], axis=0)
            rmsd_base = aligned_rmsd(pos_0, pos_T)

            metrics['rmsd'].append(complex_rmsd)
            metrics['i_rmsd'].append(interface_rmsd)
            metrics["rmsd_unbound_bound"].append(rmsd_base)

        trajectories = to_numpy(torch.stack(trajectories, dim=0).mean(dim=0))
        
        for metric, metric_list in metrics.items():
            metrics[metric] = np.round(np.mean(metric_list), 4)

        return trajectories, metrics
