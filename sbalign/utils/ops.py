import torch
from typing import Any, Set, Union, List

# -------- Tensor ops -----------------

def rbf_basis(inputs, basis_min: float, basis_max: float, interval: int):
    # Compute the centers of the basis
    n_basis_terms = int((basis_max - basis_min) / interval)
    mus = torch.linspace(basis_min, basis_max, n_basis_terms)
    mus = mus.view(1, -1) # (1, n_basis_terms)

    inputs_expanded = inputs.unsqueeze(dim=-1) # (n_residues, 1)
    return torch.exp( -((inputs_expanded - mus) / interval)**2) # (n_residues, n_basis_terms)


def expand_to_shape():
    pass
 

def to_numpy(tensor):
    if tensor.device != "cpu":
        tensor = tensor.cpu()

    if tensor.requires_grad_:
        return tensor.detach().numpy()
    return tensor.numpy()


def get_mask(a_batch, b_batch):
    rows, cols = len(a_batch), len(b_batch)
    mask = torch.zeros((rows, cols), device=a_batch.device)

    lig_counts = torch.unique(a_batch, return_counts=True)[1]
    rec_counts = torch.unique(b_batch, return_counts=True)[1]

    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(lig_counts, rec_counts):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n

    return mask

# -------- Featurizing ops ----------------

def onek_encoding_unk(x: Any, allowable_set: Union[List, Set]) -> List:
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def index_with_unk(a, b):
    if b in a:
        return a[b]
    else:
        return a['unk']


# ------- Geometry Ops -------------

def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))