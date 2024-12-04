# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common 3D pose methods
"""

import torch
from jaxtyping import Float
from torch import Tensor


def to4x4(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 4 4"]:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def multiply(pose_a: Float[Tensor, "*batch 3 4"], pose_b: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right.
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a.

    Returns:
        Camera pose matrix where pose_a was applied to pose_b.
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


def normalize(poses: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Normalize the XYZs of poses to fit within a unit cube ([-1, 1]). Note: This operation is not in-place.

    Args:
        poses: A collection of poses to be normalized.

    Returns;
        Normalized collection of poses.
    """
    pose_copy = torch.clone(poses)
    pose_copy[..., :3, 3] /= torch.max(torch.abs(poses[..., :3, 3]))

    return pose_copy

import torch

def inv_sqrtm(matrix, normalize_eigvals=False):
    """
    Takes the inverse matrix square root of a positive semi-definite (PSD) matrix, supporting batched inputs.

    Args:
        matrix (torch.Tensor): A PSD matrix of shape (..., d, d), where ... represents batch dimensions.
        normalize_eigvals (bool): If True, normalize the eigenvalues by the geometric mean.

    Returns:
        torch.Tensor: The inverse square root of the matrix, shape (..., d, d).
        tuple: A tuple (eigvec, eigval), where eigvec are the eigenvectors and eigval are the eigenvalues.
    """
    # Eigenvalue decomposition for batched matrices
    eigval, eigvec = torch.linalg.eigh(matrix)  # eigval: (..., d), eigvec: (..., d, d)

    # Normalize eigenvalues by the geometric mean if required
    if normalize_eigvals:
        log_eigval = torch.log(eigval)
        mean_log_eigval = log_eigval.mean(dim=-1, keepdim=True)  # Shape: (..., 1)
        eigval = torch.exp(log_eigval - mean_log_eigval)  # Shape: (..., d)

    # Compute scaling (1 / sqrt(eigenvalues)), with clamping for numerical stability
    scaling = 1.0 / torch.sqrt(eigval.clamp(min=1e-12))  # Shape: (..., d)
    scaling = scaling.unsqueeze(-2)  # Reshape to (..., 1, d) for broadcasting

    # Compute the inverse square root matrix
    sqrtm_mat = eigvec @ (scaling * eigvec.transpose(-2, -1))  # Shape: (..., d, d)

    return sqrtm_mat, (eigvec, eigval)


