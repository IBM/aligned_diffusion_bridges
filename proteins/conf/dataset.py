import os
import json
import copy
import random
from multiprocessing.pool import Pool
from itertools import product

import numpy as np
import torch
import torch.utils.data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from proteins.conf.utils import (
    parse_biopandas,
    get_residue_dict,
    align_residue_dicts,
    prepare_init_final_graph
)
from sbalign.utils.sb_utils import (
    sample_from_brownian_bridge, 
    get_diffusivity_schedule
)


class ProteinConfDataset(Dataset):
    
    def __init__(self,
            root: str,
            transform: BaseTransform = None,
            dataset: str = "d3pm",
            split_mode: str = "train",
            resolution: str = "c_alpha", 
            num_workers: int = 1,
            progress_every: int = 1000,
            max_protein_pairs: int = None,
            center_conformations: bool = False,
            samples_per_protein: int = None,
        ):
        
        super().__init__(root=root, transform=transform)

        self.raw_data_dir = os.path.join(self.raw_dir, dataset)
        self.processed_data_dir = os.path.join(self.processed_dir, dataset)

        self.dataset = dataset
        self.split_mode = split_mode
        self.resolution = resolution
        self.num_workers = num_workers
        self.progress_every = progress_every
        self.max_protein_pairs = max_protein_pairs
        self.center_conformations = center_conformations
        self.samples_per_protein = samples_per_protein

        processed_arg_str = f"resolution={resolution}"
        if center_conformations:
            processed_arg_str += f"_centered_conf"

        self.full_processed_dir = os.path.join(
            self.processed_data_dir, processed_arg_str
        )

        if split_mode in ["train", "val", "test"]:
            with open(f"{self.raw_data_dir}/splits.json", "r") as f:
                splits = json.load(f)
                self.conf_pairs_split = splits[split_mode]

            self.conf_pairs_split = [conf_pair_id for conf_pair_id in self.conf_pairs_split
                                    if os.path.exists(f"{self.full_processed_dir}/{conf_pair_id}.pt")]

            self.load_ids()

    def load_ids(self):
        if self.samples_per_protein is not None:
            self.sample_ids = list(range(self.samples_per_protein))
            self.ids = list(product(self.conf_pairs_split, self.sample_ids))

        else:
            self.ids = self.conf_pairs_split
        print(f"Number of {self.split_mode} conformation pairs: {len(self.ids)}", flush=True)
        random.shuffle(self.ids)
    
    def len(self):
        return len(self.ids)
    
    def get(self, idx):
        if self.samples_per_protein is not None:
            conf_pair_id, _ = self.ids[idx]
        else:
            conf_pair_id = self.ids[idx]

        if not os.path.exists(f"{self.full_processed_dir}/{conf_pair_id}.pt"):
            return None
        
        conf_pair_out = torch.load(f"{self.full_processed_dir}/{conf_pair_id}.pt")
        conf_pair_out.conf_id = conf_pair_id
        return conf_pair_out.clone()
    
    def preprocess_conformation_pairs(self):
        os.makedirs(self.full_processed_dir, exist_ok=True)

        # Loading all conformation pair ids
        with open(f"{self.raw_data_dir}/conf_pairs.txt", "r") as f:
            conf_pairs = f.readlines()
            conf_pairs = [conf_pair.strip().split(",") for conf_pair in conf_pairs]

        if self.max_protein_pairs is not None:
            conf_pairs = conf_pairs[: self.max_protein_pairs]

        print(f"Preprocessing {len(conf_pairs)} conformation pairs.", flush=True)
        print(f"Loading from: {self.raw_data_dir}/conformations/", flush=True)
        print(f"Saving to: {self.full_processed_dir}", flush=True)
        print(flush=True)

        failures = []
        if self.num_workers > 1:
            for i in range(len(conf_pairs) // self.progress_every + 1):
                conf_pairs_batch = conf_pairs[
                    self.progress_every * i : self.progress_every * (i + 1)
                ]

                p = Pool(self.num_workers, maxtasksperchild=1)
                map_fn = p.imap_unordered
                for (conf_pair_data, conf_pair_id) in map_fn(self.preprocess_conformation_pair, conf_pairs_batch):
                    if conf_pair_data is not None:
                        conf_pair_file = f"{self.full_processed_dir}/{conf_pair_id}.pt"

                        print(f"Saving {conf_pair_id} to {conf_pair_file}", flush=True)
                        torch.save(conf_pair_data, f"{conf_pair_file}")
                        print(flush=True)
                    else:
                        failures.append(conf_pair_id)
                        print(flush=True)
                p.__exit__(None, None, None)

        else:
            for (conf_pair_data, conf_pair_id) in map(self.preprocess_conformation_pair, conf_pairs):
                if conf_pair_data is not None:
                    conf_pair_file = f"{self.full_processed_dir}/{conf_pair_id}.pt"

                    print(f"Saving {conf_pair_id} to {conf_pair_file}", flush=True)

                    torch.save(conf_pair_data, f"{conf_pair_file}")
                    print(flush=True)
                else:
                    failures.append(conf_pair_id)
                    print(flush=True)

        print("Finished preprocessing conformation pairs.", flush=True)
        print(f"Failures: {failures}", flush=True)

    def preprocess_conformation_pair(self, id_chains):
        init_id_chain, final_id_chain = id_chains
        
        struct_init, struct_final = self._load_structures(init_id_chain, final_id_chain)
        residues_init = get_residue_dict(df=struct_init)
        residues_final = get_residue_dict(df=struct_final)

        residues_init, residues_final = align_residue_dicts(residues_init, residues_final)

        conf_pair_data = prepare_init_final_graph(
                residues_init=residues_init,
                residues_final=residues_final,
                resolution=self.resolution,
                centering=self.center_conformations
            )
        conf_pair_id = f"{init_id_chain}_{final_id_chain}"

        return conf_pair_data, conf_pair_id

    def _load_structures(self, init_id_chain, final_id_chain):
        init_id, init_chain = init_id_chain.split("_")
        final_id, final_chain = final_id_chain.split("_")

        init_file = f"{self.raw_data_dir}/conformations/{init_id}.cif"
        struct_init = parse_biopandas(filename=init_file, chain_id=init_chain)

        final_file = f"{self.raw_data_dir}/conformations/{final_id}.cif"
        struct_final = parse_biopandas(filename=final_file, chain_id=final_chain)
        return struct_init, struct_final


# -------- Transforms -----------------


class BrownianBridgeTransform(BaseTransform):
    
    def __init__(self, g):
        self.g = g

    def __call__(self, data):
        if data is None:
            return None
        t = np.random.uniform()
        return self.apply_transform(data=data, t=t)
    
    def apply_transform(self, data, t):
        data.t = t * torch.ones(data.num_nodes)
        data.pos_t = sample_from_brownian_bridge(g=self.g, t=t, x_0=data.pos_0, x_T=data.pos_T)

        return data


# ----------- Convenience Functions -------------

def pretty_print_pyg(data):
    attr_str = f"x={data.x.shape}, pos_0={data.pos_0.shape}, pos_T={data.pos_T.shape}"
    if 'edge_index' in data:
        edge_index = data.edge_index
        if edge_index is not None:
            attr_str += f", edge_index={data.edge_index.shape}"
    return attr_str


def construct_transform(args):
    if args.transform is None:
        return None

    if args.transform == "brownian_bridge":
        g_fn = get_diffusivity_schedule(schedule=args.diffusivity_schedule, 
                                        g_max=args.max_diffusivity)
        transform = BrownianBridgeTransform(g=g_fn)
        return transform
    else:
        raise ValueError(f" Transform of type {args.transform} is not supported.")


def build_data_loader(args):

    transform = construct_transform(args)

    train_dataset = ProteinConfDataset(root=args.data_dir, transform=transform,
                                       split_mode="train", resolution=args.resolution,
                                       dataset=args.dataset,
                                       num_workers=args.num_workers, 
                                       progress_every=None,
                                       center_conformations=args.center_conformations,
                                       samples_per_protein=args.samples_per_protein_train,
                                    )
    
    val_dataset = ProteinConfDataset(root=args.data_dir, transform=transform,
                                     split_mode="val", resolution=args.resolution,
                                     dataset=args.dataset,
                                     num_workers=args.num_workers, 
                                     progress_every=None,
                                     center_conformations=args.center_conformations,
                                     samples_per_protein=1
                                    ) 

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_bs, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_bs, shuffle=False)

    return train_loader, val_loader
