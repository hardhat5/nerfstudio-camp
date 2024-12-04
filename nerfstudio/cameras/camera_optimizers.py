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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import numpy
import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import *
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3", "SE3WithFocalIntrinsics", "SCNeRF", "FocalPoseWithIntrinsics"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: float = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    focal_std_penalty: torch.Tensor = torch.tensor([0.1, 0.1, 0.01, 0.01])

    optimize_intrinsics: bool = True
    """Whether to optimize focal length."""

    principal_point_penalty: float = 1e-2
    distortion_penalty: float = 1e-1
    distortion_std_penalty: float = 0.1
    start_focal_train: int = 3000

    use_preconditioning: bool = True

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

class CameraPreconditioner(nn.Module):

    config: CameraOptimizerConfig

    def __init__(self, config: CameraOptimizerConfig):
        super().__init__()
        self.config = config
        self.precondition_num_points = 10  # The number of points used to compute the preconditioning matrix.
        self.precondition_near = 1  # The near plane depth of the point sampling frustum.
        self.precondition_far = 1000  # The far plane depth of the point sampling frustum.

        #TODO: Fix the values based on CamP code
        self.precondition_diagonal_absolute_padding = 1e-8
        self.precondition_diagonal_relative_padding_scale = 1e-1

    def _unproject_points(self, points, depths, cameras):
        """
            Unproject 2D pixel coordinates to depth-scaled 3D points.
        """
        K_inverse = torch.linalg.inv(cameras.get_intrinsics_matrices())

        # Unproject points into normalized camera coordinates
        points = torch.cat([points, torch.ones((points.shape[0],points.shape[1],1))], dim=-1)
        points_norm = (K_inverse @ points.transpose(1,2)).transpose(1,2)

        # scale by depth (norm P3 -> R3)
        points_3d = points_norm * depths.unsqueeze(-1)
        test_pixels = self._project_points(torch.zeros((len(cameras), 6)), points_3d, cameras, in_camera_frame=True)
        assert torch.allclose(test_pixels, points[..., :2], atol=0.01, rtol=0), "Projection-Unprojection works fine."


        # transform to world coordinates
        points_3d = torch.cat([points_3d, torch.ones(points_3d.shape[0],points_3d.shape[1],1)], dim=-1)
        c2ws = cameras.camera_to_worlds
        points_3d = torch.matmul(c2ws, points_3d.transpose(1,2)).transpose(1,2)

        # for testing purposes
        test_pixels = self._project_points(torch.zeros((len(cameras), 6)), points_3d, cameras)
        assert torch.allclose(test_pixels, points[..., :2], atol=0.01, rtol=0), "Coordinate transformation is buggy."

        return points_3d
        

    def _create_points_from_frustum(self, cameras):
        """
            Create points from frustum corners.
            Ensure different cameras have different sampled points.
            TODO: Set PyTorch seed somewhere.
        
        """
        
        # Create frustum corners
        img_width = cameras.width[0].item()  # assume all cameras have the same width
        img_height = cameras.height[0].item()  # assume all cameras have the same height

        # Choose random integer points for all cameras
        frustum_pixels = torch.stack([
            torch.randint(0, img_height, (len(cameras), self.precondition_num_points)),
            torch.randint(0, img_width, (len(cameras), self.precondition_num_points))
        ], dim=2).to(cameras.device)

        # choose integer depth points in the scene
        depths = torch.randint(low=self.precondition_near, high=self.precondition_far, size=(len(cameras), self.precondition_num_points)).to(cameras.device)

        # unproject to get 3D points, keep in camera space for ease of calculations later
        return self._unproject_points(frustum_pixels, depths, cameras)

    def _update_intrinsincs(self, cameras, camera_params):
        """
            Update camera intrinsics based on camera parameters.
        """
        K_corrected = cameras.get_intrinsics_matrices().to(camera_params.device)
        D_corrected = torch.zeros(2, device=cameras.device).repeat(K_corrected.shape[0], 1)

        if not self.config.optimize_intrinsics:
            pass
        elif self.config.mode == "SE3WithFocalIntrinsics":
            K_adjustment = camera_params[:, 6:10]
            D_adjustment = camera_params[:, 10:]

            K_corrected[:, 0, 0] *= torch.exp(K_adjustment[:, 0])
            K_corrected[:, 1, 1] *= torch.exp(K_adjustment[:, 1])
            K_corrected[:, 0, 2] += K_adjustment[:, 2]
            K_corrected[:, 1, 2] += K_adjustment[:, 3]

            D_corrected[:, 0] += D_adjustment[:, 0]
            D_corrected[:, 1] += D_adjustment[:, 1]
            
        elif self.config.mode == "SCNeRF":
            K_adjustment = camera_params[:, 9:13]
            D_adjustment = camera_params[:, 13:]

            K_corrected[:, 0, 0] += torch.exp(K_adjustment[:, 0])
            K_corrected[:, 1, 1] += torch.exp(K_adjustment[:, 1])
            K_corrected[:, 0, 2] += K_adjustment[:, 2]
            K_corrected[:, 1, 2] += K_adjustment[:, 3]
            D_corrected[:, 0] += D_adjustment[:, 0]
            D_corrected[:, 1] += D_adjustment[:, 1]
        elif self.config.mode == "FocalPoseWithIntrinsics":
            K_adjustment = camera_params[:, 9:12]
            D_adjustment = camera_params[:, 12:]

            K_corrected[:, 0, 0] *= torch.exp(K_adjustment[:, 0])
            K_corrected[:, 1, 1] *= torch.exp(K_adjustment[:, 0])
            K_corrected[:, 0, 2] += K_adjustment[:, 1]
            K_corrected[:, 1, 2] += K_adjustment[:, 2]
            D_corrected[:, 0] += D_adjustment[:, 0]
            D_corrected[:, 1] += D_adjustment[:, 1]
        else:
            assert_never(self.config.mode)
        
        return K_corrected, D_corrected
    
    def _update_extrinsics(self, cameras, camera_params):
        """
            Update camera extrinsics based on camera parameters.
        """
        if self.config.mode == "SO3xR3":
            pose_adjustment = camera_params[:, :6]
            pose_corrected = exp_map_SO3xR3(pose_adjustment)
        elif self.config.mode == "SE3":
            pose_adjustment = camera_params[:, :6]
            pose_corrected = exp_map_SE3(pose_adjustment)
        elif self.config.mode == "SE3WithFocalIntrinsics":
            pose_adjustment = camera_params[:, :6]
            pose_corrected = exp_map_SE3(pose_adjustment)
        elif self.config.mode == "SCNeRF":
            # slice the relevant parts of the camera adjustment
            pose_adjustment = camera_params[:, :9]

            # extract rotation matrix from 1st 6 elements
            R = get_rotation_matrix_from_6d_vector(pose_adjustment[:, 0:6])
            translation_vector = pose_adjustment[:, 6:9]
            pose_corrected = torch.zeros(
                translation_vector.shape[0],
                3,
                4,
                dtype=translation_vector.dtype,
                device=translation_vector.device,
            )
            pose_corrected[:, :3, :3] = R
            pose_corrected[:, :3, 3] = translation_vector
        elif self.config.mode == "FocalPoseWithIntrinsics":
            Ks = cameras.get_intrinsics_matrices()
            c2ws = cameras.camera_to_worlds
            pose_adjustment = camera_params[:, :9]
            K_adjustment = camera_params[:, 9:12]

            R = get_rotation_matrix_from_6d_vector(pose_adjustment[:, 0:6])
            vx = pose_adjustment[:, 6]
            vy = pose_adjustment[:, 7]
            vz = pose_adjustment[:, 8]
            f = Ks[indices, 0, 0]
            
            x = c2ws[:, 0, 3]
            y = c2ws[:, 1, 3] 
            z = c2ws[:, 2, 3]

            f_new = torch.exp(K_adjustment[:, 0]) * f
            z_new = torch.exp(vz) * z
            x_new = (vx / f_new + x / z) * z_new
            y_new = (vy / f_new + y / z) * z_new

            trans = torch.stack([x_new - x, y_new - y, z_new - z], dim=1)
            pose_corrected = torch.zeros(
                    R.shape[0],
                    3,
                    4,
                    dtype=R.dtype,
                    device=R.device,
                )

            pose_corrected[:, :3, :3] = R  # Set the rotation part
            pose_corrected[:, :3, 3] = trans  # Set the translation part
        else:
            assert_never(self.config.mode)

        c2ws = cameras.camera_to_worlds.to(pose_corrected.device)
        pose_corrected = pose_utils.multiply(c2ws, pose_corrected)  # Shape [num_cameras, 3, 4]

        # During camP initialization, pose_corrected is the same as original cameras as deltas are zero
        # assert torch.allclose(c2ws, pose_corrected, atol=0.01, rtol=0), "Pose correction is buggy."

        return pose_corrected

    def _update_camera(self, cameras, camera_params):
        """
            Update camera intrinsics and extrinsics based on camera parameters.
        """
        
        pose_corrected = self._update_extrinsics(cameras, camera_params)
        K_corrected, D_corrected = self._update_intrinsincs(cameras, camera_params)

        return pose_corrected, K_corrected, D_corrected

    def _project_points(self, camera_params, points_3d, cameras, in_camera_frame=False):
        """
            Project 3D points to 2D pixel coordinates.
        """
        pose_corrected, K_corrected, D_corrected = self._update_camera(cameras, camera_params)  # ensure camera parameters gradients
        if in_camera_frame:
            pose_corrected = torch.eye(4, device=cameras.device)[None, :3, :4].repeat(len(cameras), 1, 1)

        # Optional assertion
        # if not in_camera_frame:
        #     assert torch.allclose(cameras.camera_to_worlds, pose_corrected, atol=0.01, rtol=0), "Pose correction is buggy."
        
        # convert c2w correction to w2c correction
        pose_corrected = pose_utils.inverse(pose_corrected)

        # apply perspective projection correction to image space
        points_3d = torch.cat([points_3d, torch.ones((points_3d.shape[0],points_3d.shape[1],1),device=points_3d.device)], dim=-1)
        points_3d = torch.matmul(pose_corrected, points_3d.transpose(1,2)).transpose(1,2)
        pixels = torch.matmul(K_corrected, points_3d.transpose(1,2)).transpose(1,2)
        pixels = pixels/pixels[..., 2, None]

        # Do radial distortion correction
        # r = raybundle.directions[..., :2].norm(dim=-1, keepdim=True)
        r = pixels[..., :2].norm(dim=-1, keepdim=True)
        r2 = r**2.0
        r4 = r2**2.0
        radial = (
            1.0
            + r2 * D_corrected[:, None, 0].unsqueeze(-1)
            + r4 * D_corrected[:, None, 1].unsqueeze(-1)
        )
        pixels = pixels * torch.cat(
            (radial, radial, torch.ones((*radial.shape[:-1], 1), device=radial.device, dtype=radial.dtype)), dim=-1
        )
        pixels = pixels/pixels[..., 2, None]

        return pixels[..., :2]

    def _compute_jacobian(self, fn, camera_params, points_3d, cameras):
        """
            Compute the Jacobian of a function fn at x.
            Returns the Jacobian matrix and projected pixels.
            TODO: Re-check the implementation of this method.
        """
        points_3d = points_3d.detach().requires_grad_(True)
        nc, m, _ = points_3d.shape  # Number of cameras, number of points
        camera_params = camera_params.detach().requires_grad_(True)  # Ensure x is differentiable

        # project 3D points to 2D pixel coordinates
        projected_pixels = fn(camera_params, points_3d, cameras)  # Expected shape: (nc, m, 2)

        # assert that projected pixels are within image bounds as 4 separate conditions
        assert torch.all(projected_pixels[..., 0] >= -0.01), "Projected pixels are out of bounds."
        assert torch.all(projected_pixels[..., 0] < cameras.height[0]+0.01), "Projected pixels are out of bounds."
        assert torch.all(projected_pixels[..., 1] >= -0.01), "Projected pixels are out of bounds."
        assert torch.all(projected_pixels[..., 1] < cameras.width[0]+0.01), "Projected pixels are out of bounds."

        # Initialize the Jacobian matrix
        jacobian = torch.zeros(nc, m, 2, camera_params.shape[-1], device=camera_params.device)  # Shape: (nc, m, 2, x.shape[-1])

        camera_params = camera_params.to("cuda")
        points_3d = points_3d.to("cuda")
        cameras = cameras.to("cuda")

        def wrapped_fn(camera_params_batch, points_3d_batch, cameras_batch):
            return fn(camera_params_batch, points_3d_batch, cameras_batch).reshape(-1, 20)  # Flatten to 20 outputs per camera

        # Initialize a list to collect Jacobians
        jacobians = []

        # Loop through each batch (camera)
        for i in range(camera_params.shape[0]):
            camera_params_batch = camera_params[None, i]
            points_3d_batch = points_3d[None, i]
            cameras_batch = cameras[None, i]

            # Compute the Jacobian for this batch
            jacobian = torch.autograd.functional.jacobian(
                lambda p: wrapped_fn(p, points_3d_batch, cameras_batch).reshape(-1),  # Flatten 20 outputs for this camera
                camera_params_batch,
                create_graph=True
            )
            jacobians.append(jacobian.squeeze(1))

        # Stack Jacobians for all cameras
        jacobians = torch.stack(jacobians, dim=0)

        return jacobians, projected_pixels

    def _compute_approximate_hessian(self, camera_params, cameras):
        """
            Compute the approximate Hessian matrix.
            TODO: Finish implementing this method.
        """
        points_3d = self._create_points_from_frustum(cameras)
        j, pixels = self._compute_jacobian(self._project_points, camera_params, points_3d, cameras)

        # TODO: Zero out the gradients for pixels that are outside of camera viewpoint -> heavy distortion
        # pixels = None
        # jtj = None

        jtj = j.transpose(1,2) @ j
        return jtj

    def _precondition_with_padding(self, jtj, normalize_eigvals=False):
        """
        Applies preconditioning by adding diagonal padding and computing the inverse square root matrix.
        """
        diagonal_absolute_padding = self.precondition_diagonal_absolute_padding * torch.ones(jtj.shape[-1], device=jtj.device)
        diagonal_relative_padding = self.precondition_diagonal_relative_padding_scale * jtj.diagonal(dim1=-2, dim2=-1)  # Shape: [204, 6]

        diagonal_padding = torch.diag(torch.maximum(diagonal_absolute_padding, diagonal_relative_padding))
        
        padded_matrix = jtj + diagonal_padding
        matrix = pose_utils.inv_sqrtm(padded_matrix, normalize_eigvals=normalize_eigvals)
        return matrix

        
    def get_camera_preconditioner(self, camera_params, cameras):
        """Get the preconditioner for the camera parameters"""
        jtj = self._compute_approximate_hessian(camera_params, cameras)  
        P, _ = self._precondition_with_padding(jtj)
        return P
        


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        cameras: Optional[Cameras] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        self.cameras = cameras
        self.step = 0
        if cameras is not None:
            self.c2ws = self.cameras.camera_to_worlds
            self.Ks = torch.eye(3)
            self.Ks = self.Ks[None, ...].repeat(self.c2ws.shape[0], 1, 1)
            self.Ks[:, 0, 0] = (self.cameras.fx / self.cameras.width).squeeze()
            self.Ks[:, 1, 1] = (self.cameras.fy / self.cameras.height).squeeze()
            self.Ks[:, 0, 2] = (self.cameras.cx / self.cameras.width).squeeze()
            self.Ks[:, 1, 2] = (self.cameras.cy / self.cameras.height).squeeze()
            self.Ksinv = torch.inverse(self.Ks)
            self.Ds = torch.zeros(2, device=device).repeat(self.c2ws.shape[0], 1) # fix this
        else:
            self.c2ws = torch.eye(4, device=device)[None, :3, :4].repeat(
                num_cameras, 1, 1
            )

        # temporarily re-setting config for debugging
        # Eliminate this code later

        self.config.mode = "SO3xR3"
        psize = 6

        # self.config.mode = "SE3WithFocalIntrinsics"
        # psize = 12

        # self.config.mode = "SCNeRF"
        # p_size = 15

        # self.config.mode = "FocalPoseWithIntrinsics"
        # p_size = 14

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.cameras is None:
            raise ValueError("cameras must be provided to optimize parameters")
        elif self.config.mode in ("SO3xR3", "SE3"):
            # only optimize pose
            # forward() function expects pose_adjustment to be defined but we want it to be called camera_adjustment
            #TODO: Fix this bug
            self.camera_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
            self.config.optimize_intrinsics = False
        elif self.config.mode == "SE3WithFocalIntrinsics":
            # optimize pose, fx, fy, cx, cy
            self.camera_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 12), device=device))
            self.config.optimize_intrinsics = True # bjhamb - uncouple config.optimize_intrinsics from mode
        elif self.config.mode == "SCNeRF":
            # SCNeRF has 6D rotation vector and a 3D translation vector
            self.camera_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 15), device=device))
            self.config.optimize_intrinsics = True # bjhamb - uncouple config.optimize_intrinsics from mode

        elif self.config.mode == "FocalPoseWithIntrinsics":
            # FocalPoseWithIntrinsics has 3 elements for translation and 3 elements for rotation
            self.camera_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 14), device=device))
            self.config.optimize_intrinsics = True # bjhamb - uncouple config.optimize_intrinsics from mode

        else:
            assert_never(self.config.mode)

        # initialize preconditioner and calculate P matrix
        if self.config.use_preconditioning:
            self.preconditioner = CameraPreconditioner(self.config)
            self.P_matrix = self.preconditioner.get_camera_preconditioner(self.camera_adjustment.detach(), cameras)
            # self.P_matrix = torch.eye(psize, device=device)[None, ...].repeat(num_cameras, 1, 1)  # override with dummy P matrix for debuggingx

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        if self.Ks.device != self.camera_adjustment.device:
            self.Ks = self.Ks.to(self.camera_adjustment.device)
            self.Ksinv = self.Ksinv.to(self.camera_adjustment.device)
        K_corrected = self.Ks[indices]
        if self.Ds.device != self.camera_adjustment.device:
            self.Ds = self.Ds.to(self.camera_adjustment.device)
        D_corrected = self.Ds[indices]

        if self.config.use_preconditioning:
            # apply pre-conditioning to entire camera delta tensor
            preconditioned_camera_adjustment = torch.bmm(self.P_matrix.to(indices.device), self.camera_adjustment[..., None]).squeeze()

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            self.pose_adjustment = preconditioned_camera_adjustment*1
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            self.pose_adjustment = preconditioned_camera_adjustment*1
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3WithFocalIntrinsics":
            # slice the relevant parts of the camera adjustment
            self.pose_adjustment = preconditioned_camera_adjustment[:, :6]
            self.K_adjustment = preconditioned_camera_adjustment[:, 6:10]
            self.D_adjustment = preconditioned_camera_adjustment[:, 10:]

            # update it to only use pose part (first 6 elements)
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
            K_corrected[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 1])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 2]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 3]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]
        elif self.config.mode == "SCNeRF":
            # slice the relevant parts of the camera adjustment
            self.pose_adjustment = preconditioned_camera_adjustment[:, :9]
            self.K_adjustment = preconditioned_camera_adjustment[:, 9:13]
            self.D_adjustment = preconditioned_camera_adjustment[:, 13:]

            # extract rotation matrix from 1st 6 elements
            R = get_rotation_matrix_from_6d_vector(self.pose_adjustment[indices, 0:6])
            translation_vector = self.pose_adjustment[indices, 6:9]
            ret = torch.zeros(
                translation_vector.shape[0],
                3,
                4,
                dtype=translation_vector.dtype,
                device=translation_vector.device,
            )
            ret[:, :3, :3] = R  # Set the rotation part
            ret[:, :3, 3] = translation_vector  # Set the translation part
            outputs.append(ret)
            K_corrected[:, 0, 0] += torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] += torch.exp(self.K_adjustment[indices, 1])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 2]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 3]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]
        elif self.config.mode == "FocalPoseWithIntrinsics":
            # slice the relevant parts of the camera adjustment
            self.pose_adjustment = preconditioned_camera_adjustment[:, :9]
            self.K_adjustment = preconditioned_camera_adjustment[:, 9:12]
            self.D_adjustment = preconditioned_camera_adjustment[:, 12:]

            R = get_rotation_matrix_from_6d_vector(self.pose_adjustment[indices, 0:6])
            vx = self.pose_adjustment[indices, 6]
            vy = self.pose_adjustment[indices, 7]
            vz = self.pose_adjustment[indices, 8]
            f = self.Ks[indices, 0, 0]
            self.c2ws = self.c2ws.to(self.pose_adjustment.device) #bjhamb
            x = self.c2ws[indices, 0, 3]
            y = self.c2ws[indices, 1, 3] 
            z = self.c2ws[indices, 2, 3]

            f_new = torch.exp(self.K_adjustment[indices, 0]) * f
            z_new = torch.exp(vz) * z
            x_new = (vx / f_new + x / z) * z_new
            y_new = (vy / f_new + y / z) * z_new

            trans = torch.stack([x_new - x, y_new - y, z_new - z], dim=1)
            ret = torch.zeros(
                    R.shape[0],
                    3,
                    4,
                    dtype=R.dtype,
                    device=R.device,
                )

            ret[:, :3, :3] = R  # Set the rotation part
            ret[:, :3, 3] = trans  # Set the translation part
            outputs.append(ret)
            # assumes fx = fy
            K_corrected[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 1]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 2]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        pose_corrections = functools.reduce(pose_utils.multiply, outputs)
        return K_corrected, pose_corrections

    # def apply_to_raybundle(self, raybundle: RayBundle) -> None:
    #     """Apply the pose correction to the raybundle"""
    #     if self.config.mode != "off":
    #         correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
    #         raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
    #         raybundle.directions = torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None]).squeeze()

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        
        if self.config.mode != "off":
            K_corrected, H_corrected = self(raybundle.camera_indices.squeeze())  # type: ignore
            D_corrected = self.Ds[raybundle.camera_indices.squeeze()]
            if self.c2ws.device != K_corrected.device:
                self.c2ws = self.c2ws.to(K_corrected.device)

            if self.config.optimize_intrinsics:
            
                # Convert to 4x4 transformation matrix
                Hs = self.c2ws[raybundle.camera_indices.squeeze()]
                rows = torch.tensor([0, 0, 0, 1], dtype=Hs.dtype, device=Hs.device)[None, :].repeat(Hs.shape[0], 1, 1)

                # Get world-to-camera transformations
                Hs = torch.cat([Hs, rows], dim=1)
                Hs_inv = torch.inverse(Hs)

                # Transform directions to camera space and project
                raybundle.directions = torch.bmm(Hs_inv[:, :3, :3], raybundle.directions[..., None]).squeeze()

                # Do perspective projection and homogeneous divide
                Ks = self.Ks[raybundle.camera_indices.squeeze()]
                raybundle.directions = torch.bmm(Ks, raybundle.directions[..., None]).squeeze()
                raybundle.directions = raybundle.directions / -raybundle.directions[..., 2:3].repeat(1, 3)

                # Do radial distortion correction
                r = raybundle.directions[..., :2].norm(dim=-1, keepdim=True)  # 4096,1
                r2 = r**2.0
                r4 = r2**2.0
                radial = (
                    1.0
                    + r2 * D_corrected[raybundle.camera_indices, 0]
                    + r4 * D_corrected[raybundle.camera_indices, 1]
                )
                raybundle.directions = raybundle.directions * torch.cat(
                    (radial, radial, torch.ones((*radial.shape[:-1], 1), device=radial.device, dtype=radial.dtype)), dim=-1
                )

                # Get ray directions in camera space using corrected intrinsics matrix
                raybundle.directions = torch.bmm(torch.inverse(K_corrected), raybundle.directions[..., None]).squeeze()
                raybundle.directions = raybundle.directions / raybundle.directions.norm(dim=-1, keepdim=True)

                # Transform rays back to world space
                raybundle.directions = torch.bmm(Hs[..., :3, :3], raybundle.directions[..., None]).squeeze()

            # Apply pose correction to rays
            raybundle.directions = torch.bmm(H_corrected[:, :3, :3], raybundle.directions[..., None]).squeeze()
            raybundle.origins = raybundle.origins + H_corrected[:, :3, 3]

    def apply_to_camera(self, camera: Cameras) -> torch.Tensor:
        """Apply the pose correction to the world-to-camera matrix in a Camera object"""
        if self.config.mode == "off":
            return camera.camera_to_worlds

        if camera.metadata is None or "cam_idx" not in camera.metadata:
            # Viser cameras
            return camera.camera_to_worlds

        camera_idx = camera.metadata["cam_idx"]
        adj = self(torch.tensor([camera_idx], dtype=torch.long)).to(camera.device)  # type: ignore

        return torch.cat(
            [
                # Apply rotation to directions in world coordinates, without touching the origin.
                # Equivalent to: directions -> correction[:3,:3] @ directions
                torch.bmm(adj[..., :3, :3], camera.camera_to_worlds[..., :3, :3]),
                # Apply translation in world coordinate, independently of rotation.
                # Equivalent to: origins -> origins + correction[:3,3]
                camera.camera_to_worlds[..., :3, 3:] + adj[..., :3, 3:],
            ],
            dim=-1,
        )

    def set_step(self, step: int) -> None:
        """Set the step for the optimizer"""
        self.step = step

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            loss_dict["camera_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
                + self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )
            # option 1 : somehow get  SE3 transform here, and then get the translation and rotation from it
            # use the translation as it is, and for rotation find se3 to calculate the norm

            # option 2 : directly calculate the norm of the 6D vector
            # does it even make sense to have a penalty on the 6D vector?

            if self.config.optimize_intrinsics:
                
                if self.config.focal_std_penalty.device != self.K_adjustment.device:
                    self.config.focal_std_penalty = self.config.focal_std_penalty.to(self.K_adjustment.device)
                
                loss_dict["camera_opt_std_regularizer"] = (
                    self.K_adjustment.std(dim=0) * self.config.focal_std_penalty
                ).mean()

                loss_dict["camera_opt_principal_point_regularizer"] = (
                    self.K_adjustment.norm(dim=0)[2:].norm() * self.config.principal_point_penalty
                )

                loss_dict["camera_opt_distortion_regularizer"] = (
                    self.D_adjustment.norm(dim=0).norm() * self.config.distortion_penalty
                )

                loss_dict["camera_opt_distortion_std_regularizer"] = (
                    self.D_adjustment.std(dim=0) * self.config.distortion_std_penalty
                ).mean()

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            trans = self.pose_adjustment[:, :3].detach().norm(dim=-1)
            rot = self.pose_adjustment[:, 3:].detach().norm(dim=-1)
            metrics_dict["camera_opt_translation_max"] = trans.max()
            metrics_dict["camera_opt_translation_mean"] = trans.mean()
            metrics_dict["camera_opt_rotation_mean"] = numpy.rad2deg(rot.mean().cpu())
            metrics_dict["camera_opt_rotation_max"] = numpy.rad2deg(rot.max().cpu())

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0
