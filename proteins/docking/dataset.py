import os
import json
import copy
import random
import traceback
from multiprocessing.pool import Pool
from itertools import product

import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


from proteins.docking.utils import parse_pdb, prepare_complex_graph, prepare_residue_dict
from sbalign.utils.sb_utils import (
    sample_from_brownian_bridge, 
    get_diffusivity_schedule
)
from sbalign.utils.ops import axis_angle_to_matrix


class RigidProteinDocking(Dataset):

    def __init__(self, 
            root, 
            transform=None, 
            dataset: str = "db5", 
            split_mode: str = "train",
            resolution: str = "c_alpha", 
            esm_embeddings_path: str = None, 
            max_complexes: int = None,
            samples_per_complex: int = 5, 
            center_complex: bool = False,
            num_workers: int = 1,
            progress_every: int = 1000, 
            backend: str = "biopandas"
        ):

        super().__init__(root=root, transform=transform)

        # Dataset and setup args
        self.raw_data_dir = os.path.join(self.raw_dir, dataset)
        self.processed_data_dir = os.path.join(self.processed_dir, dataset)
        self.num_workers = num_workers
        self.progress_every = progress_every
        self.split_mode = split_mode
        self.backend = backend

        # Ligand/Receptor associated args
        self.resolution = resolution
        self.max_complexes = max_complexes
        self.center_complex = center_complex
        self.samples_per_complex = samples_per_complex

        proceseed_arg_str = f"resolution={resolution}"

        if esm_embeddings_path is not None:
            proceseed_arg_str += f"_esmEmbeddings"

        self.full_processed_dir = os.path.join(
            self.processed_data_dir, proceseed_arg_str
        )

        if split_mode in ["train", "val", "test"]:
            with open(f"{self.raw_data_dir}/splits.json", "r") as f:
                splits = json.load(f)
                self.complexes_split = splits[split_mode]

            self.complexes_split = [complex for complex in self.complexes_split
                                    if os.path.exists(f"{self.full_processed_dir}/{complex}.pt")]

            self.load_ids()

    def load_ids(self):
        complex_ids = self.complexes_split
        sample_ids_per_complex = list(range(self.samples_per_complex))
        # We can generate multiple samples per complex -> artificial dataset size increase
        self.ids = list(product(complex_ids, sample_ids_per_complex))

        print(f"Number of {self.split_mode} complexes: {len(self.complexes_split)}", flush=True)
        random.shuffle(self.ids)

    def len(self):
        return len(self.ids)

    def get(self, idx):
        complex_id, sample_id = self.ids[idx]

        if not os.path.exists(f"{self.full_processed_dir}/{complex_id}.pt"):
            return None
        
        complex = torch.load(f"{self.full_processed_dir}/{complex_id}.pt")

        if "pos_orig" not in complex:
            complex['ligand'].pos_orig = complex['ligand'].pos_T.clone()
            complex['receptor'].pos_orig = complex['receptor'].pos_T.clone()
        
        return copy.deepcopy(complex)

    def preprocess_complexes(self):
        os.makedirs(self.full_processed_dir, exist_ok=True)
        
        # Loading all complex ids
        with open(f"{self.raw_data_dir}/complexes.txt", "r") as f:
            complex_ids_all = f.readlines()
            complex_ids_all = [complex_id.strip() for complex_id in complex_ids_all]

        if self.max_complexes is not None:
            complex_ids_all = complex_ids_all[: self.max_complexes]

        print(f"Preprocessing {len(complex_ids_all)} complexes.", flush=True)
        print(f"Loading from: {self.raw_data_dir}/complexes", flush=True)
        print(f"Saving to: {self.full_processed_dir}", flush=True)
        print(flush=True)

        failures = []

        if self.num_workers > 1:
            for i in range(len(complex_ids_all) // self.progress_every + 1):
                complex_ids_batch = complex_ids_all[
                    self.progress_every * i : self.progress_every * (i + 1)
                ]

                p = Pool(self.num_workers, maxtasksperchild=1)
                map_fn = p.imap_unordered
                for (complex, complex_id) in map_fn(self.preprocess_complex, complex_ids_batch):
                    if complex is not None:
                        print(f"Saving {complex_id} to {self.full_processed_dir}/{complex_id}.pt", flush=True)
                        torch.save(complex, f"{self.full_processed_dir}/{complex_id}.pt")
                        print(flush=True)
                    else:
                        failures.append(complex_id)
                        print(flush=True)
                p.__exit__(None, None, None)

        else:
            for (complex, complex_id) in map(self.preprocess_complex, complex_ids_all):
                if complex is not None:
                    print(f"Saving {complex_id} to {self.full_processed_dir}/{complex_id}.pt", flush=True)
                    torch.save(complex, f"{self.full_processed_dir}/{complex_id}.pt")
                    print(flush=True)
                else:
                    failures.append(complex_id)
                    print(flush=True)

        print("Finished preprocessing complexes", flush=True)
        print(f"Failures: {failures}", flush=True)

    def preprocess_complex(self, complex_id):
        try:
            structures = self._load_structures(complex_id=complex_id)
            residue_info = {}

            groupby = ['chain_id', 'residue_number', 'insertion', 'residue_name'] \
                if self.backend == "biopandas" else None

            for key, struct_info in structures.items():
                residue_info[key] = prepare_residue_dict(protein_info=struct_info,
                                                         groupby=groupby, 
                                                         backend=self.backend)

            base_complex = HeteroData()
            for key in structures:
                prepare_complex_graph(complex_data=base_complex, prot_key=key,
                                      residue_dict=residue_info[key],
                                      resolution=self.resolution)
            
            for key in structures:
                attr_str_key = pretty_print_pyg(base_complex, key)
                print(f"{complex_id}: Prepared {key} graph - {attr_str_key}", flush=True)

            # Receptor is fixed
            base_complex['receptor'].pos_0 = base_complex['receptor'].pos_T.clone()

            if self.center_complex:
                rec_center = torch.mean(complex["receptor"].pos_T, dim=0, keepdim=True)
                complex.orig_center = rec_center
                
                complex["ligand"].pos_T -= rec_center
                complex["receptor"].pos_T -= rec_center
            
            return base_complex, complex_id

        except Exception as e:
            print(f"Failed to process {complex_id} because of {e}", flush=True)
            traceback.print_exc()
            return None, complex_id

    def _load_structures(self, complex_id: str):
        proteins = ["ligand", "receptor"]
        file_suffixes = {"ligand": "l_b.pdb", "receptor": "r_b.pdb"}
        structures = {}

        for protein in proteins:
            protein_file = f"{self.raw_data_dir}/complexes/{complex_id}_{file_suffixes[protein]}"
            structures[protein] = parse_pdb(pdb_file=protein_file, backend=self.backend)
        
        return structures
    

# ------------- Transforms ---------------------


class BrownianBridgeTransform(BaseTransform):

    def __init__(self, 
            g, rot_vec = None, tr_vec = None,
            max_tr: float = 5.0):
        
        self.g = g
        self.max_tr = max_tr

        if rot_vec is None:
            self.rot_vec, self.tr_vec = \
                self.sample_rigid_body_transform(max_tr=max_tr)
        else:
            self.rot_vec = rot_vec
            self.tr_vec = tr_vec

    def __call__(self, data):
        if data is None:
            return None

        if isinstance(data, list):
            values = np.linspace(start=0., stop=1., num=len(data)+1)
            t_intervals = [(values[i], values[i+1]) for i in range(len(values)-1)]

            t = [np.random.uniform(t_start, t_stop) for t_start, t_stop in t_intervals]
            data_list = [self.apply_transform(data[idx], t[idx]) for idx in range(len(t))]
            return data_list

        t = np.random.uniform()
        return self.apply_transform(data=data, t=t)
    
    def sample_rotation(self):
        rot_vec = Rotation.random(num=1).as_rotvec()
        return torch.from_numpy(rot_vec).float()

    def sample_translation(self, max_tr=5.0):
        tr = np.randn(1, 3)
        tr = tr / np.sqrt( np.sum(tr * tr))
        length = np.random.uniform(low=0, high=max_tr)
        tr = tr * length
        return np.from_numpy(tr).float()
    
    def sample_rigid_body_transform(self, max_tr=5.0):
        rot_vec = self.sample_rotation()
        tr_vec = self.sample_translation(max_tr=max_tr)
        return rot_vec, tr_vec

    def apply_transform(self, data, t):
        data["ligand"].t = t * torch.ones(data["ligand"].num_nodes)
        data["receptor"].t = t * torch.ones(data["receptor"].num_nodes)

        rot_mat = axis_angle_to_matrix(self.rot_vec).squeeze(0)
        tr_vec = self.tr_vec
        
        # Applying random rotation translation to obtain unbound ligand
        # Rotation is applied about the center of mass
        pos_mean = torch.mean(data["ligand"].pos_orig, dim=0, keepdims=True)
        data["ligand"].pos_0 = (rot_mat @ (data["ligand"].pos_orig - pos_mean).T).T + pos_mean + tr_vec

        data["ligand"].pos_t = sample_from_brownian_bridge(g=self.g, t=t, 
                                                           x_0=data["ligand"].pos_0, x_T=data["ligand"].pos_T, 
                                                           t_min=0.0, t_max=1.0)                                               
        return data



# -------------------- Convenience Functions ---------------

def build_data_loader(args):

    print("Sampling random rotation and translation", flush=True)
    
    # Sample axis-angle rotation vector
    rot_vec = np.random.randn(1, 3)
    rot_vec = rot_vec / np.linalg.norm(rot_vec)
    rot_vec = args.rot_norm * rot_vec
    rot_vec = torch.from_numpy(rot_vec).float()

    # Sample translation
    tr_vec = np.random.randn(1, 3)
    tr_vec = tr_vec / np.sqrt( np.sum(tr_vec * tr_vec))
    length = np.random.uniform(low=args.tr_min, high=args.tr_max)
    tr_vec = tr_vec * length
    tr_vec = torch.from_numpy(tr_vec).float()

    if args.transform is None:
        g_fn = get_diffusivity_schedule(schedule=args.diffusivity_schedule, 
                                        g_max=args.max_diffusivity)
        transform = BrownianBridgeTransform(g=g_fn, rot_vec=rot_vec, tr_vec=tr_vec)

    train_dataset = RigidProteinDocking(
                        root=args.data_dir, transform=transform,
                        dataset=args.dataset, split_mode="train",
                        resolution=args.resolution, 
                        num_workers=args.num_workers,
                        samples_per_complex=args.samples_per_complex_train,
                        progress_every=100,
                        center_complex=args.center_complex,
                    )

    val_dataset = RigidProteinDocking(
                        root=args.data_dir, transform=transform,
                        dataset=args.dataset, split_mode="val",
                        resolution=args.resolution, 
                        num_workers=args.num_workers,
                        progress_every=100,
                        samples_per_complex=1,
                        center_complex=args.center_complex
                    )  

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_bs, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_bs, shuffle=False)

    return train_loader, val_loader, {"rot_vec": rot_vec, "tr_vec": tr_vec}


def pretty_print_pyg(data, key):
    attr_str = f"x={data[key].x.shape}, pos_T={data[key].pos_T.shape}"
    if 'edge_index' in data[key]:
        edge_index = data[key].edge_index
        if edge_index is not None:
            attr_str += f", edge_index={data[key].edge_index.shape}"
    return attr_str
