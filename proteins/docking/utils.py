import os
import warnings
import copy
from typing import Dict

import torch
import numpy as np
import scipy.spatial as spa
from scipy.special import softmax
from Bio.PDB import PDBParser
from torch_geometric.data import HeteroData
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb

from sbalign.utils.ops import rbf_basis, onek_encoding_unk, index_with_unk

ProtParser = PDBParser()

# --------- Definitions and constants --------

KD_SCALE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
    "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
    "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5, "unk": 0.0
}

AMINO_ACIDS = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
    "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
    "THR", "VAL", "TRP", "TYR", "unk",
]

VOLUMES = {
    "GLY": 60.1, "ALA": 88.6, "SER": 89.0, "CYS": 108.5, "ASP": 111.1, "PRO": 112.7, 
    "ASN": 114.1, "THR": 116.1, "GLU": 138.4, "VAL": 140.0, "GLN": 143.8, "HIS": 153.2, 
    "MET": 162.9, "ILE": 166.7, "LEU": 166.7, "LYS": 168.6, "ARG": 173.4, "PHE": 189.9, 
    "TYR": 193.6, "TRP": 227.8, "unk": 0,
}

CHARGES = {
    "ARG": 1, "LYS": 1, "ASP": -1, "GLU": -1, "HIS": 0.1, "ALA": 0, "CYS": 0,
    "PHE": 0, "GLY": 0, "ILE": 0, "LEU": 0, "MET": 0, "ASN": 0, "PRO": 0,
    "GLN": 0, "SER": 0, "THR": 0, "VAL": 0, "TRP": 0, "TYR": 0, "unk": 0,
}

POLARITY = {
    "ARG": 1, "ASN": 1, "ASP": 1, "GLN": 1, "GLU": 1, "HIS": 1, "LYS": 1,
    "SER": 1, "THR": 1, "TYR": 1, "ALA": 0, "CYS": 0, "GLY": 0, "ILE": 0,
    "LEU": 0, "MET": 0, "PHE": 0, "PRO": 0, "TRP": 0, "VAL": 0, "unk": 0,
}

ACCEPTOR = {
    "ASP": 1, "GLU": 1, "ASN": 1, "GLN": 1, "HIS": 1, "SER": 1, "THR": 1,
    "TYR": 1, "ARG": 0, "LYS": 0, "TRP": 0, "ALA": 0, "CYS": 0, "GLY": 0,
    "ILE": 0, "LEU": 0, "MET": 0, "PHE": 0, "PRO": 0, "VAL": 0, "unk": 0,
}

DONOR = {
    "ARG": 1, "LYS": 1, "TRP": 1, "ASN": 1, "GLN": 1, "HIS": 1, "SER": 1,
    "THR": 1, "TYR": 1, "ASP": 0, "GLU": 0, "ALA": 0, "CYS": 0, "GLY": 0,
    "ILE": 0, "LEU": 0, "MET": 0, "PHE": 0, "PRO": 0, "VAL": 0, "unk": 0,
}


# --------- Parsing and Loading --------

def parse_pdb(pdb_file, backend='biopython'):
    pdb_file = os.path.abspath(pdb_file)

    if backend == "biopandas":
        ppdb = PandasPdb().read_pdb(pdb_file)
        df = ppdb.df['ATOM']
        df['residue_number'] = df['residue_number'].astype(int)
        return df

    elif backend == "biopython":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PDBConstructionWarning)
            structure = ProtParser.get_structure(id="random", file=pdb_file)
        prot = structure[0]
        return prot
    

def prepare_residue_dict(protein_info, groupby=None, backend='biopython'):
    if backend == "biopython":
        residue_dict, _ = extract_valid_chains_and_residues(
            copy.deepcopy(protein_info)
        )

        return residue_dict
        
    elif backend == "biopandas":
        info_grouped = protein_info.groupby(groupby)
        residue_dict = {
            key: value for key, value in info_grouped
        }
    
        return residue_dict


def extract_valid_chains_and_residues(prot):
    valid_chains = {}
    valid_residues = {}
    invalid_residues_in_chains = {}

    for chain_idx, chain in enumerate(prot):
        invalid_residues = []
        count = 0

        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                # Add water molecule to invalid residues
                invalid_residues.append(
                    (chain_idx,) + residue.get_id() + (residue.get_resname(),)
                )
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
            else:
                invalid_residues.append(
                    (chain_idx,) + residue.get_id() + (residue.get_resname(),)
                )

        if count:
            valid_chains[chain_idx] = chain
            invalid_residues_in_chains[chain_idx] = invalid_residues
            # TODO: Should we delete the invalid residues from the chain?

    invalid_residues = [
        invalid_residue
        for invalid_residues_in_chain in invalid_residues_in_chains.values()
        for invalid_residue in invalid_residues_in_chain
    ]
    valid_chains_updated = delete_residues_from_chains(
        chains=valid_chains, residue_ids=invalid_residues
    )
    return valid_residues, valid_chains_updated


def delete_residues_from_chains(chains, residue_ids, verbose: bool = False):
    if verbose:
        print("Before deletion, Residue Count:", flush=True)
        for chain_id, chain in chains.items():
            num_residues = len(list(chain.get_residues()))
            print(f"Chain {chain_id}={num_residues}", flush=True)

    # Delete residues from the chain
    for residue_id in residue_ids:
        chain_id = residue_id[0]
        chain = chains[chain_id]
        chain.detach_child(residue_id[1:-1])

    # Create an updated copy of the chains
    chains_updated = {
        chain_id: copy.deepcopy(chain) for chain_id, chain in chains.items()
    }

    if verbose:
        print("After deletion, Residue Count:", flush=True)
        for chain_id, chain in chains_updated.items():
            num_residues = len(list(chain.get_residues()))
            print(f"Chain {chain_id}={num_residues}", flush=True)

    return chains_updated


# ---------- Featurizing --------------

def compute_residue_feats(residues):
    residue_feats = []

    for residue_id, residue_info in residues.items():
        res_name = residue_id[1]
        residue_feat = onek_encoding_unk(res_name, AMINO_ACIDS) + [
            index_with_unk(KD_SCALE, res_name),
            index_with_unk(VOLUMES, res_name),
            index_with_unk(CHARGES, res_name),
            index_with_unk(POLARITY, res_name),
            index_with_unk(ACCEPTOR, res_name),
            index_with_unk(DONOR, res_name),
        ]
        residue_feats.append(residue_feat)

    residue_feats = torch.tensor(residue_feats)
    hydrophobicity = residue_feats[:, len(AMINO_ACIDS)]
    volumes = residue_feats[:, len(AMINO_ACIDS) + 1] / 100.0
    charges = residue_feats[:, len(AMINO_ACIDS) + 2]
    polarity_hbonds = residue_feats[:, len(AMINO_ACIDS) + 2 :]

    # Expand components into gaussian basis
    # Taken from https://github.com/wengong-jin/abdockgen
    residue_feats = torch.cat(
        [
            residue_feats[:, : len(AMINO_ACIDS)],
            rbf_basis(hydrophobicity, -4.5, 4.5, 0.1),
            rbf_basis(volumes, 0, 2.2, 0.1),
            rbf_basis(charges, -1.0, 1.0, 0.25),
            torch.sigmoid(polarity_hbonds * 6 - 3),
        ],
        dim=-1,
    )
    return residue_feats


# ---------- Graph Preparation ---------

def gather_ca_coords(residue_info):
    try:
        ca_coord = residue_info['CA'].get_coord()

    except Exception:
        ca = residue_info[residue_info['atom_name'] == 'CA']
        ca_coord = ca[["x_coord", "y_coord", "z_coord"]].values
        if ca_coord.shape[0] > 1:
            # This is most likely due to the altloc attribute
            ca_coord = ca_coord.mean(axis=0, keepdims=True)
    
    return ca_coord.squeeze()


def prepare_complex_graph(
    complex_data: HeteroData,
    prot_key: str,
    residue_dict,
    resolution: str = "c_alpha"
):
    assert resolution == "c_alpha", "Currently only supports c_alpha graphs"

    n_residues = len(residue_dict)
    c_alpha_coords = np.asarray([
        gather_ca_coords(residue_info) for res_id, residue_info in residue_dict.items()
    ])

    node_features = compute_residue_feats(residue_dict)

    complex_data[prot_key].x = node_features
    complex_data[prot_key].pos_T = torch.from_numpy(c_alpha_coords).float()
    return
