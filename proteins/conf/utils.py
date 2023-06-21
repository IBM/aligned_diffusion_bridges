import os
import warnings
import copy
import math
from typing import Dict, Tuple, List

import torch
import numpy as np
import pandas as pd
import scipy.spatial as spa
from scipy.special import softmax
from torch_geometric.data import Data
from biopandas.mmcif import PandasMmcif
from biopandas.pdb import PandasPdb

from sbalign.utils.ops import rbf_basis, onek_encoding_unk, index_with_unk

ResidueId = Tuple[int, str]
ResidueInfo = pd.DataFrame
ResidueDict = Dict[ResidueId, ResidueInfo]


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


# -------------- Loading Structures -----------------

def parse_biopandas(filename: str, chain_id=None):
    ext = filename.split(".")[-1]
    if ext == "cif":
        return parse_cif_biopandas(filename, chain_id=chain_id)
    elif ext == "pdb":
        return parse_pdb_biopandas(filename, chain_id=chain_id)
    else:
        raise ValueError(f"Extension of type {ext} is not supported yet.")


def parse_cif_biopandas(filename: str, chain_id: str = None):
    pdb_id = os.path.basename(filename)
    ppdb = PandasMmcif().read_mmcif(path=filename)
    df_atom = ppdb._df['ATOM']
    
    if chain_id is not None:
        df_chain = df_atom[df_atom['auth_asym_id'] == chain_id]
        df_chain.loc[:, 'label_seq_id'] = df_chain.loc[:, 'label_seq_id'].astype(int)
        return df_chain
    return df_atom


def parse_pdb_biopandas(filename: str, chain_id: str = None):
    pdb_file = os.path.abspath(filename)
    ppdb = PandasPdb().read_pdb(pdb_file)
    df = ppdb.df['ATOM']
    df['residue_number'] = df['residue_number'].astype(int)
    return df


# --------------- Preparing Structures --------------

def get_residue_dict(df, groupby=["auth_seq_id", "label_comp_id"]):
    df_grouped = df.groupby(groupby)
    residue_dict = {
        key: value for key, value in df_grouped
    }
    return residue_dict


def align_residue_dicts(residues_A: ResidueDict, residues_B: ResidueDict):
    residue_list_A = list(residues_A.keys())
    residue_list_B = list(residues_B.keys())

    common_residues = set(residue_list_A).intersection(residue_list_B)
    common_residues = sorted(common_residues)

    residues_A_aligned = {
        key: residues_A[key] for key in common_residues
    }

    residues_B_aligned = {
        key: residues_B[key] for key in common_residues
    }

    return residues_A_aligned, residues_B_aligned


def gather_ca_coords(residue_df: pd.DataFrame) -> np.ndarray:
    ca = residue_df[residue_df['label_atom_id'] == 'CA']
    ca_coords = ca[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    if ca_coords.shape[0] > 1:
        # This is most likely due to the altloc attribute
        ca_coords = ca_coords.mean(axis=0, keepdims=True)
    return ca_coords.squeeze()


# ----------- Preparing Graphs --------------

def compute_residue_feats(residues: ResidueDict):
    residue_feats = []

    for residue_id, residue_info in residues.items():
        res_name = residue_id[-1]
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


def compute_residue_feats_advanced(residues: ResidueDict):
    raise NotImplementedError()


def prepare_init_final_graph(
    residues_init: ResidueDict,
    residues_final: ResidueDict,
    resolution: str = "c_alpha",
    centering: bool = False,
) -> Data:
    
    assert resolution == "c_alpha", "Currently only supports c_alpha graphs"
    num_residues = len(residues_init)
    if num_residues <= 1:
        raise ValueError("Provided protein contains only 1 residue.")
    assert len(residues_init) == len(residues_final)

    conf_data = Data()
    
    ca_coords_init = np.asarray([
        gather_ca_coords(residue_df=residue_df) for residue_id, residue_df in residues_init.items()
    ])

    ca_coords_final = np.asarray([
        gather_ca_coords(residue_df=residue_df) for residue_id, residue_df in residues_final.items()
    ])

    assert ca_coords_init.shape == ca_coords_final.shape, "Mismatch in c_alpha shapes"

    # Do the alignment between structures
    rot_mat, tr = rigid_transform_kabsch_3D(ca_coords_init.T, ca_coords_final.T)
    ca_coords_init_aligned = ( (rot_mat @ ca_coords_init.T) + tr ).T

    if centering:
        ca_coords_init_com = np.mean(ca_coords_init_aligned, axis=0, keepdims=True)
        ca_coords_init_aligned = ca_coords_init_aligned - ca_coords_init_com

        ca_coords_final_com = np.mean(ca_coords_final, axis=0, keepdims=True)
        ca_coords_final = ca_coords_final - ca_coords_final_com

    node_features = compute_residue_feats(residues_init)

    conf_data.pos_0 = torch.from_numpy(ca_coords_init_aligned).float()
    conf_data.pos_T = torch.from_numpy(ca_coords_final).float()
    conf_data.x = node_features
    return conf_data
 

def rigid_transform_kabsch_3D(A: np.ndarray, B: np.ndarray):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag(np.array([1., 1., -1.], dtype=np.float32))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t