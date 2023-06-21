import os
import torch
import copy
import yaml
import argparse
import numpy as np

from Bio.PDB import PDBIO
from torch_geometric.loader import DataLoader

from proteins.docking.utils import parse_pdb
from proteins.docking.models import build_model_from_args
from proteins.docking.dock_engine import DockingEngine
from proteins.docking.dataset import RigidProteinDocking, BrownianBridgeTransform

from sbalign.utils.sb_utils import get_diffusivity_schedule


# --------------- PDB File Preparation ----------------------

def load_residues_from_pdb(struct):
    valid_residues = {}
    count = 0

    for chain_idx, chain in enumerate(struct):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                # Add water molecule to invalid residues
                continue

            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                valid_residues[
                    (chain_idx,) + residue.get_id() + (residue.get_resname(),)
                ] = residue
                count += 1

    residue_ids = valid_residues.keys()
    residue_ids = sorted(residue_ids, key=lambda x: x[:-1], reverse=False)
    
    updated_residues = {residue_id: valid_residues[residue_id] for residue_id in residue_ids}
    return updated_residues  


def generate_prediction_file(pdb_file, ca_coords_0, ca_coords_T, pdb_out_file):
    io = PDBIO()

    struct = parse_pdb(pdb_file=pdb_file, backend="biopython")
    struct_copy = copy.deepcopy(struct)

    residues = load_residues_from_pdb(struct=struct_copy)
    assert len(residues) == len(ca_coords_0), f"{len(residues)}, {len(ca_coords_0)}"
    
    for idx, (residue_id, residue) in enumerate(residues.items()):

        diff = ca_coords_T[idx] - ca_coords_0[idx]

        for atom in residue:
            old_coord = atom.get_coord()
            new_coord = old_coord + diff
            atom.set_coord(new_coord)

    io.set_structure(struct_copy)
    print(f"Saving file to {pdb_out_file}")
    print()
    io.save(pdb_out_file)


# ----------- Inference setup related -------------

def parse_inference_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/", 
                        help="Directory to load processed samples from.")
    parser.add_argument("--inference_dir")
    parser.add_argument("--inference_out_dir")
    parser.add_argument("--log_dir", default="./logs", help="Log directory")
    parser.add_argument("--run_name", help="Name of the run to load model from")
    parser.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt",
                        help="Checkpoint to use for the trained model.")
    parser.add_argument("--traj_dir", default="./trajectory_outputs/", type=str,
                        help="Directory where inference results will be saved.")

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--samples_per_complex", type=int, default=5)
    parser.add_argument("--inference_steps", type=int, default=100)

    args = parser.parse_args()
    return args


def prepare_inference_setup(args):
    with open(f'{args.log_dir}/{args.run_name}/config_train.yml') as f:
        model_args = argparse.Namespace(**yaml.full_load(f))

    R_tr_dict = torch.load(f"{args.log_dir}/{args.run_name}/R_tr.pt")
    rot_vec = R_tr_dict['rot_vec']
    tr_vec = R_tr_dict['tr_vec']

    dataset = model_args.dataset
    args.dataset = dataset

    # Loading the model
    model = build_model_from_args(model_args)
    model_dict = torch.load(f"{args.log_dir}/{args.run_name}/best_inference_epoch_model.pt",
                             map_location='cpu')
    model.load_state_dict(model_dict)

    g_fn = get_diffusivity_schedule(schedule=model_args.diffusivity_schedule, 
                                    g_max=model_args.max_diffusivity)

    if model_args.transform is None:
        transform = BrownianBridgeTransform(g=g_fn, rot_vec=rot_vec, tr_vec=tr_vec)

    dataset = RigidProteinDocking(
                        root=args.data_dir, transform=transform,
                        dataset=model_args.dataset, split_mode="test",
                        resolution=model_args.resolution, 
                        num_workers=model_args.num_workers,
                        samples_per_complex=1,
                        progress_every=100
                    )

    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    engine = DockingEngine(samples_per_complex=args.samples_per_complex,
                           inference_steps=args.inference_steps, model=model,
                           g_fn=g_fn)    

    return model, loader, engine


def run_docking(args):

    model, loader, engine = prepare_inference_setup(args)
    
    metrics = {'rmsd': [], 'i_rmsd': []}
    traj_dict = {}

    for data in loader:
        complex_name = data['complex_id'][0]
        print(f"Docking complex {complex_name}")

        trajectory, complex_metrics = engine.dock(data)
        for metric, metric_val in complex_metrics.items():
            if metric in metrics:
                metrics[metric].append(metric_val)

        if 'complex_id' in data:
            if len(data['complex_id']) > 0:
                traj_dict[data['complex_id'][0]] = trajectory
    
    return traj_dict, metrics


if __name__ == "__main__":
    args = parse_inference_args()

    print("Generating docked examples for the test dataset...")
    traj_dict, metrics_dict = run_docking(args)
    
    log_metrics = ['rmsd', 'i_rmsd']
    for metric in log_metrics:
        metrics = metrics_dict[metric]
        median_metric = np.median(metrics)
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)

        print(f"{metric}: Mean={mean_metric} Std={std_metric} Median={median_metric}")
    print(flush=True)

    # Save PDB Files to output directory

    inference_files_dir = f"{args.inference_dir}/db5/test_inputs/" 
    inference_out_dir = os.path.join(args.inference_out_dir, "db5", "sbalign_results")
    traj_dir = f"{args.traj_dir}/{args.dataset}/{args.run_name}/trajectories"

    os.makedirs(inference_out_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    
    for complex_id in traj_dict:
        trajectory = traj_dict[complex_id]

        ca_coords_0 = trajectory[0]
        ca_coords_T = trajectory[-1]

        pdb_file = os.path.join(inference_files_dir, complex_id + '_l_u_rigid' + '.pdb')
        pdb_out_file = f"{inference_out_dir}/{complex_id}_l_b_SBALIGN.pdb"

        print(f"Performing inference on {pdb_file}")
        try:
            generate_prediction_file(pdb_file=pdb_file, ca_coords_0=ca_coords_0, 
                                ca_coords_T=ca_coords_T, pdb_out_file=pdb_out_file)
        except Exception as e:
            print(e)
            continue

        traj_file = f"{traj_dir}/{complex_id}.npy"
        print(f"Saving trajectory to {traj_file}")
        np.save(traj_file, trajectory)
