import torch
import math
import numpy as np
import scipy.spatial as spa

from sbalign.utils.ops import to_numpy


def rmsd(y_pred, y_true):
    se = (y_pred - y_true)**2
    mse = se.sum(axis=1).mean()
    return np.sqrt(mse)


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


def aligned_rmsd(y_pred, y_true):
    R, t = rigid_transform_kabsch_3D(y_pred.T, y_true.T)
    y_pred_aligned = ( (R @ y_pred.T) + t ).T
    rmsd_value = rmsd(y_pred_aligned, y_true)
    return rmsd_value


def compute_complex_rmsd(ligand_pred, ligand_true, receptor_pred, receptor_true):
    if isinstance(ligand_pred, torch.Tensor):
        ligand_pred = to_numpy(ligand_pred)
        ligand_true = to_numpy(ligand_true)

        receptor_pred = to_numpy(receptor_pred)
        receptor_true = to_numpy(receptor_true)

    complex_pred = np.concatenate([ligand_pred, receptor_pred], axis=0)
    complex_true = np.concatenate([ligand_true, receptor_true], axis=0)

    R, t = rigid_transform_kabsch_3D(complex_pred.T, complex_true.T)
    complex_pred_aligned = ( (R @ complex_pred.T) + t ).T

    complex_rmsd = rmsd(complex_pred_aligned, complex_true)
    ligand_rmsd = rmsd(ligand_pred, ligand_true)
    receptor_rmsd = rmsd(receptor_pred, receptor_true)

    return complex_rmsd, {
        "complex_rmsd": complex_rmsd, 
        "ligand_rmsd": ligand_rmsd,
        "receptor_rmsd": receptor_rmsd
    }


def compute_interface_rmsd(ligand_pred, ligand_true, receptor_pred, receptor_true, dist: float = 8.0):
    if isinstance(ligand_pred, torch.Tensor):
        ligand_pred = to_numpy(ligand_pred)
        ligand_true = to_numpy(ligand_true)

        receptor_pred = to_numpy(receptor_pred)
        receptor_true = to_numpy(receptor_true)
    
    ligand_receptor_distance = spa.distance.cdist(ligand_true, receptor_true)
    positive_tuple = np.where(ligand_receptor_distance < dist)
    active_ligand = positive_tuple[0]
    active_receptor = positive_tuple[1]

    ligand_pocket_pred = ligand_pred[active_ligand, :]
    ligand_pocket_true = ligand_true[active_ligand, :]

    rec_pocket_pred = receptor_pred[active_receptor, :]
    rec_pocket_true = receptor_true[active_receptor, :]

    interface_rmsd, complex_dict = compute_complex_rmsd(
            ligand_pred=ligand_pocket_pred,
            ligand_true=ligand_pocket_true,
            receptor_pred=rec_pocket_pred,
            receptor_true=rec_pocket_true
        )

    irmsd_dict = {}
    for key, value in complex_dict.items():
        irmsd_dict[key.split("_")[0] + "irmsd"] = value
    return interface_rmsd, irmsd_dict
