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
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
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
    distortion_std_penalty: float = .1
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
    def __init__(self):
        super().__init__()
        self.precondition_num_points = 1000  # The number of points used to compute the preconditioning matrix.
        self.precondition_near = 1  # The near plane depth of the point sampling frustum.
        self.precondition_far = 1000  # The far plane depth of the point sampling frustum.

    def _unproject_points(self, points, depths, cameras):
        """
            Unproject 2D pixel coordinates to depth-scaled 3D points.
            Note: points_3d is returned in camera coordinates.
        """
        K_inverse = torch.linalg.inv(cameras.get_intrinsics_matrices())

        # Unproject points into normalized camera coordinates
        points = torch.cat([points, torch.ones((points.shape[0],points.shape[1],1))], dim=-1)
        points = (K_inverse @ points.transpose(1,2)).transpose(1,2)

        # scale by depth (P2 -> R3)
        points_3d = points * depths.unsqueeze(-1)

        # # transform to world coordinates
        # points_3d = torch.cat([points_3d, torch.ones(points_3d.shape[0],points_3d.shape[1],1)], dim=-1)
        # c2ws = self.cameras.camera_to_worlds
        # points_3d = torch.matmul(c2ws, points_3d.transpose(1,2)).transpose(1,2)

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

    def _project_points(self, camera_params, points_3d, cameras):
        """
            Project 3D points to 2D pixel coordinates.
            Note: points_3d is already in camera coordinates.
            # TODO: This is an implementation of the SE(3)+focal length projection equation.
                    In the future, we should make this an abstact method, to be defined in child classes.
        """
        # # Transform to camera coordinates
        # points_3d = torch.cat([points_3d, torch.ones(points_3d.shape[0],points_3d.shape[1],1)], dim=-1)
        # c2ws = self.cameras.camera_to_worlds
        # points_3d = torch.matmul(c2ws, points_3d.transpose(1,2)).transpose(1,2)

        # get camera intrinsics
        # TODO: apply camera correction updates
        K = cameras.get_intrinsics_matrices()

        # project to image space
        # points_3d = torch.cat([points_3d, torch.ones(points_3d.shape[0],points_3d.shape[1],1)], dim=-1)
        pixels = torch.matmul(K, points_3d.transpose(1,2)).transpose(1,2)
        pixels = pixels/pixels[..., 2, None]

        # un-homogenize
        return pixels[..., :2]

    def _compute_jacobian(self, fn, x, *args):
        """
            Compute the Jacobian of a function fn at x.
            Returns the Jacobian matrix and projected pixels.
            TODO: Re-check the implementation of this method.
        """
        points = args[0]
        points = points.detach().requires_grad_(True)
        nc, m, _ = points.shape  # Number of cameras, number of points
        x = x.detach().requires_grad_(True)  # Ensure x is differentiable

        # project 3D points to 2D pixel coordinates
        y = fn(x, *args)  # Expected shape: (nc, m, 2)

        # Initialize the Jacobian matrix
        jacobian = torch.zeros(nc, m, 2, x.shape[-1], device=x.device)  # Shape: (nc, m, 2, x.shape[-1])

        # Compute gradients for each 2D point across cameras
        # TODO: Check if this can be parallelized
        # TODO: Fix x.grad None because x:camera_params is not being used for projection. Refer to previous function TODO
        # Temporarily commented out until issue is fixed.
        
        # for c in range(nc):  
        #     for i in range(m): 
        #         for j in range(2):
        #             if x.grad is not None:
        #                 x.grad.zero_()

        #             # Backpropagate the gradient of y[c, i, j] w.r.t. x[c]
        #             y[c, i, j].backward(retain_graph=True)
        #             jacobian[c, i, j] = x.grad[c]

        return jacobian, y

    def _compute_approximate_hessian(self, camera_params, cameras, points_3d):
        """
            Compute the approximate Hessian matrix.
            TODO: Finish implementing this method.
        """
        j, pixels = self._compute_jacobian(self._project_points, camera_params, points_3d, cameras)

        # TODO: ignore pixels outside of camera viewpoint
        pixels = None
        jtj = None

        # TODO: do xTx stuff, check for valid pixels
        return jtj

    def _calculate_jtj(self, camera_params, cameras):
        """Compute the approximate Hessian matrix"""
        points_3d = self._create_points_from_frustum(cameras)
        jtj = self._compute_approximate_hessian(camera_params, cameras, points_3d)
        return jtj
        
    def get_camera_preconditioner(self, camera_params, cameras):
        """Get the preconditioner for the camera parameters"""
        return self._calculate_jtj(camera_params, cameras)
        


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
        else:
            self.c2ws = torch.eye(4, device=device)[None, :3, :4].repeat(num_cameras, 1, 1)

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            if self.config.optimize_intrinsics:
                if self.cameras is None:
                    raise ValueError("cameras must be provided to optimize intrinsics")
                # optimize pose, fx, fy, cx, cy
                self.camera_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 12), device=device))
            else:
                # only optimize pose
                self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

        # initialize preconditioner and calculate P matrix
        if self.config.use_preconditioning:
            self.preconditioner = CameraPreconditioner()
            self.P_matrix = self.preconditioner.get_camera_preconditioner(self.camera_adjustment.detach(), cameras)
            self.P_matrix = torch.eye(12, device=device)[None, ...].repeat(num_cameras, 1, 1)  # override with dummy P matrix for debugging

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

        if self.config.use_preconditioning:
            # apply pre-conditioning to entire camera delta tensor
            preconditioned_camera_adjustment = torch.bmm(self.P_matrix.to(indices.device), self.camera_adjustment[..., None]).squeeze()

            # index relevant fields to stay consistent with the rest of the code
            # ensure gradient flow is preserved
            # TODO: Remove the self attributes, we do not need to store them. For now, just to stay consistent with the rest of the code.
            self.K_adjustment = preconditioned_camera_adjustment[:, :4]
            self.D_adjustment = preconditioned_camera_adjustment[:, 4:6]
            self.pose_adjustment = preconditioned_camera_adjustment[:, 6:]

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
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

        # next multiply in the focal corrections if provided
        # optimize them separately
        if self.Ks.device != self.pose_adjustment.device:
            self.Ks = self.Ks.to(self.pose_adjustment.device)
            self.Ksinv = self.Ksinv.to(self.pose_adjustment.device)
        K_cors = self.Ks[indices]
        if self.config.optimize_intrinsics:
            K_cors[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_cors[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 1])
            K_cors[:, 0, 2] += self.K_adjustment[indices, 2]
            K_cors[:, 1, 2] += self.K_adjustment[indices, 3]
        return K_cors, pose_corrections

    # def apply_to_raybundle(self, raybundle: RayBundle) -> None:
    #     """Apply the pose correction to the raybundle"""        
    #     if self.config.mode != "off":
    #         correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
    #         raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
    #         raybundle.directions = torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None]).squeeze()

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        
        if self.config.mode != "off":
            K_cors, H_cors = self(raybundle.camera_indices.squeeze())  # type: ignore
            if self.c2ws.device != K_cors.device:
                self.c2ws = self.c2ws.to(K_cors.device)

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
                r = raybundle.directions[..., :2].norm(dim=-1, keepdim=True)
                r2 = r**2.0
                r4 = r2**2.0
                radial = (
                    1.0
                    + r2 * self.D_adjustment[raybundle.camera_indices, 0]
                    + r4 * self.D_adjustment[raybundle.camera_indices, 1]
                )
                raybundle.directions = raybundle.directions * torch.cat(
                    (radial, radial, torch.ones((*radial.shape[:-1], 1), device=radial.device, dtype=radial.dtype)), dim=-1
                )

                # Get ray directions in camera space using corrected intrinsics matrix
                raybundle.directions = torch.bmm(torch.inverse(K_cors), raybundle.directions[..., None]).squeeze()
                raybundle.directions = raybundle.directions / raybundle.directions.norm(dim=-1, keepdim=True)

                # Transform rays back to world space
                raybundle.directions = torch.bmm(Hs[..., :3, :3], raybundle.directions[..., None]).squeeze()

            # Apply pose correction to rays
            raybundle.directions = torch.bmm(H_cors[:, :3, :3], raybundle.directions[..., None]).squeeze()
            raybundle.origins = raybundle.origins + H_cors[:, :3, 3]

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
