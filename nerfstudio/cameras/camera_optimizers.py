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

    mode: Literal["off", "SO3xR3", "SE3", "SE3WithFocalIntrinsics", "SCNeRF", "FocalPoseWithIntrinsics", "SO3xR3WithFocalIntrinsics"] = "off"
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

    position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for debugging."""

    orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for debugging."""

    focal_length_noise_std: float = 0.0
    """Noise to add to initial focal lengths. Useful for debugging."""

    principal_point_penalty: float = 1e-2
    distortion_penalty: float = 1e-1
    distortion_std_penalty: float = 0.1
    start_focal_train: int = 3000

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
            self.Ks[:, 0, 0] = (self.cameras.fx).squeeze()
            self.Ks[:, 1, 1] = (self.cameras.fy).squeeze()
            self.Ks[:, 0, 2] = (self.cameras.cx).squeeze()
            self.Ks[:, 1, 2] = (self.cameras.cy).squeeze()
            self.Ksinv = torch.inverse(self.Ks)
            self.Ds = torch.zeros(2, device=device).repeat(self.c2ws.shape[0], 1) # fix this
        else:
            self.c2ws = torch.eye(4, device=device)[None, :3, :4].repeat(
                num_cameras, 1, 1
            )

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.cameras is None:
            raise ValueError("cameras must be provided to optimize parameters")
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 6), device=device)
            )
            self.config.optimize_intrinsics = False
        elif self.config.mode in ("SE3WithFocalIntrinsics", "SO3xR3WithFocalIntrinsics"):
            self.pose_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 6), device=device)
            )
            self.config.optimize_intrinsics = True # bjhamb - uncouple config.optimize_intrinsics from mode
            # optimize fx, fy, cx, cy
            self.K_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 4), device=device)
            )
            # optimize k1, k2
            self.D_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 2), device=device)
            )
        elif self.config.mode == "SCNeRF":
            # SCNeRF has 6D rotation vector and a 3D translation vector
            self.pose_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 9), device=device)
            )
            self.config.optimize_intrinsics = True
            # optimize fx, fy, cx, cy
            self.K_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 4), device=device)
            )
            # optimize k1, k2
            self.D_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 2), device=device)
            )
        elif self.config.mode == "FocalPoseWithIntrinsics":
            # FocalPoseWithIntrinsics has 3 elements for translation and 3 elements for rotation
            self.config.optimize_intrinsics = True
            self.pose_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 6), device=device)
            )
            # optimize f, cx, cy  (assume fx = fy)
            self.K_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 3), device=device)
            )            
            self.config.focal_std_penalty = self.config.focal_std_penalty[:3] 
            # take only the first 3 elements as K_adjustment has only 3 elements
            self.D_adjustment = torch.nn.Parameter(
                torch.zeros((num_cameras, 2), device=device)
            )
    
        else:
            assert_never(self.config.mode)

        # Initialize pose noise; useful for debugging
        if self.config.position_noise_std != 0.0 or self.config.orientation_noise_std != 0.0:
            assert self.config.position_noise_std >= 0.0 and self.config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [self.config.position_noise_std] * 3 + [self.config.orientation_noise_std] * 3, device=self.device
            )
            self.pose_noise = exp_map_SE3(
                torch.normal(torch.zeros((self.num_cameras, 6), device=self.device), std_vector)
            )
        else:
            self.pose_noise = None

        # Initialize focal noise; useful for debugging
        if self.config.focal_length_noise_std != 0.0:
            std_vector = torch.tensor(
                [self.config.focal_length_noise_std] * 2, device=self.device
            )
            self.focal_noise = torch.normal(torch.zeros((self.num_cameras, 2), device=self.device), torch.log(std_vector))
        else:
            self.focal_noise = None

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

        # Move everything to the current device
        if self.config.mode != "off":
            self.pose_adjustment = self.pose_adjustment.to(indices.device)

            if self.config.optimize_intrinsics:
                self.K_adjustment = self.K_adjustment.to(indices.device)
                self.D_adjustment = self.D_adjustment.to(indices.device)

        if self.Ks.device != indices.device:
            self.Ks = self.Ks.to(indices.device)
            self.Ksinv = self.Ksinv.to(indices.device)
        K_corrected = self.Ks[indices]
        if self.Ds.device != indices.device:
            self.Ds = self.Ds.to(indices.device)
        D_corrected = self.Ds[indices]

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3WithFocalIntrinsics":
            # update it to only use pose part (first 6 elements)
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
            K_corrected[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 1])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 2]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 3]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]
        elif self.config.mode == "SO3xR3WithFocalIntrinsics":
            # update it to only use pose part (first 6 elements)
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
            K_corrected[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 1])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 2]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 3]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]
        
        elif self.config.mode == "SCNeRF":
            
            if self.c2ws.device != indices.device:
                self.c2ws = self.c2ws.to(indices.device)

            if self.pose_noise is not None:
                self.pose_noise = self.pose_noise.to(indices.device)
                Rs = self.pose_noise[indices, :3, :3] @ self.c2ws[indices, :3, :3]

            else:
                Rs = self.c2ws[indices, :3, :3]
            
            # Convert rotation matrices to 6d vectors by flattening out first two columns
            original_params = Rs.permute(0,2,1)[:, :2, :].reshape(-1, 6)

            # Update params and calculate new rotation matrix
            new_params = original_params + self.pose_adjustment[indices, 0:6]
            new_R = get_rotation_matrix_from_6d_vector(new_params)

            # Create a delta rotation matrix by inverting the original rotation matrix and multiplying the new one
            delta_R = new_R @ Rs.permute(0,2,1)

            translation_vector = self.pose_adjustment[indices, 6:9]
            ret = torch.zeros(
                translation_vector.shape[0],
                3,
                4,
                dtype=translation_vector.dtype,
                device=translation_vector.device,
            )

            ret[:, :3, :3] = delta_R  # Set the rotation part
            ret[:, :3, 3] = translation_vector  # Set the translation part
            outputs.append(ret)

            K_corrected[:, 0, 0] *= torch.exp(self.K_adjustment[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.K_adjustment[indices, 1])
            K_corrected[:, 0, 2] += self.K_adjustment[indices, 2]
            K_corrected[:, 1, 2] += self.K_adjustment[indices, 3]
            D_corrected[:, 0] += self.D_adjustment[indices, 0]
            D_corrected[:, 1] += self.D_adjustment[indices, 1]

        elif self.config.mode == "FocalPoseWithIntrinsics":

            if self.c2ws.device != indices.device:
                self.c2ws = self.c2ws.to(indices.device)

            delta_R = exp_map_SO3(self.pose_adjustment[indices, :3])

            vx = self.pose_adjustment[indices, 3]
            vy = self.pose_adjustment[indices, 4]
            vz = self.pose_adjustment[indices, 5]

            if self.focal_noise is not None:
                self.focal_noise = self.focal_noise.to(indices.device)
                f = self.Ks[indices, 0, 0] * torch.exp(self.focal_noise[indices, 0])
            else:
                f = self.Ks[indices, 0, 0]

            # Scale predictions using focal length, easier to regress
            vx, vy = vx * f, vy * f

            x = self.c2ws[indices, 0, 3]
            y = self.c2ws[indices, 1, 3] 
            z = self.c2ws[indices, 2, 3]

            f_new = torch.exp(self.K_adjustment[indices, 0]) * f
            z_new = torch.exp(vz) * z
            x_new = (vx / f_new + torch.sign(z) * x / (torch.abs(z) + 1e-8)) * z_new
            y_new = (vy / f_new + torch.sign(z) * y / (torch.abs(z) + 1e-8)) * z_new

            trans = torch.stack([x_new - x, y_new - y, z_new - z], dim=1)
            ret = torch.zeros(
                    delta_R.shape[0],
                    3,
                    4,
                    dtype=delta_R.dtype,
                    device=delta_R.device,
                )

            ret[:, :3, :3] = delta_R  # Set the rotation part
            ret[:, :3, 3] = trans  # Set the translation part
            outputs.append(ret)
            # assumes fx = fy
            
            K_corrected[:, 0, 0] = f_new
            K_corrected[:, 1, 1] = f_new
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
            outputs.append(torch.eye(4, device=indices.device)[None, :3, :4].tile(indices.shape[0], 1, 1))

        pose_corrections = functools.reduce(pose_utils.multiply, outputs)
        return K_corrected, pose_corrections, D_corrected

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        
        indices = raybundle.camera_indices.squeeze()
        K_corrected, H_corrected, D_corrected = self(indices)  # type: ignore

        # print(self.K_adjustment[0,0].item(), self.K_adjustment[0,1].item(), self.K_adjustment[0,2].item(), self.K_adjustment[0,3].item())

        # Apply noise to poses if specified
        if self.pose_noise is not None:
            outputs = [H_corrected]
            self.pose_noise = self.pose_noise.to(indices.device)
            outputs.append(self.pose_noise[indices, :, :])
            H_corrected = functools.reduce(pose_utils.multiply, outputs)

        # Apply noise to focal length if specified
        if self.focal_noise is not None and self.config.mode != "FocalPoseWithIntrinsics":
            self.focal_noise = self.focal_noise.to(indices.device)
            K_corrected[:, 0, 0] *= torch.exp(self.focal_noise[indices, 0])
            K_corrected[:, 1, 1] *= torch.exp(self.focal_noise[indices, 1])

        if self.c2ws.device != K_corrected.device:
            self.c2ws = self.c2ws.to(K_corrected.device)

        # Apply intrinsics deltas
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
        raybundle.directions = raybundle.directions / raybundle.directions[..., 2:3].repeat(1, 3)
        
        # Normalize pixels for numerical stability
        raybundle.directions[..., 0] = (raybundle.directions[..., 0] - Ks[:, 0, 2]) / Ks[:, 0, 0]
        raybundle.directions[..., 1] = (raybundle.directions[..., 1] - Ks[:, 1, 2]) / Ks[:, 1, 1]

        # Do radial distortion correction
        r = raybundle.directions[..., :2].norm(dim=-1, keepdim=True)
        r2 = r**2.0
        r4 = r2**2.0
        radial = (
            1.0
            + r2 * D_corrected[raybundle.camera_indices, 0]
            + r4 * D_corrected[raybundle.camera_indices, 1]
        )
        raybundle.directions = (raybundle.directions) * torch.cat(
            (radial, radial, torch.ones((*radial.shape[:-1], 1), device=radial.device, dtype=radial.dtype)), dim=-1
        )

        # Undo the normalization back to pixels
        raybundle.directions[:, 0] = raybundle.directions[:, 0] * Ks[:, 0, 0] + Ks[:, 0, 2]
        raybundle.directions[:, 1] = raybundle.directions[:, 1] * Ks[:, 1, 1] + Ks[:, 1, 2]

        # Get ray directions in camera space using corrected intrinsics matrix
        raybundle.directions = torch.bmm(torch.inverse(K_corrected), raybundle.directions[..., None]).squeeze()
        raybundle.directions = -raybundle.directions / raybundle.directions.norm(dim=-1, keepdim=True) # Minus sign for some reason

        # print(raybundle.directions[0,0].item(), raybundle.directions[0,1].item(), raybundle.directions[0,2].item())

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

        return 

        # """Add regularization"""
        # if self.config.mode in ["SO3xR3", "SE3", "SE3WithFocalIntrinsics"]:
        #     loss_dict["camera_opt_regularizer"] = (
        #         self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
        #         + self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
        #     )
        #     # option 1 : somehow get  SE3 transform here, and then get the translation and rotation from it
        #     # use the translation as it is, and for rotation find se3 to calculate the norm

        #     # option 2 : directly calculate the norm of the 6D vector
        #     # does it even make sense to have a penalty on the 6D vector?
        
        # # elif self.config.mode == "SCNeRF":
        # #     loss_dict["camera_opt_regularizer"] = (
        # #         self.pose_adjustment[:, 6:].norm(dim=-1).mean() * self.config.trans_l2_penalty
        # #         + self.pose_adjustment[:, :6].norm(dim=-1).mean() * self.config.rot_l2_penalty
        # #     )

        # if self.config.optimize_intrinsics:
            
        #     if self.config.focal_std_penalty.device != self.K_adjustment.device:
        #         self.config.focal_std_penalty = self.config.focal_std_penalty.to(self.K_adjustment.device)
            
        #     loss_dict["camera_opt_std_regularizer"] = (
        #         self.K_adjustment.std(dim=0) * self.config.focal_std_penalty
        #     ).mean()

        #     loss_dict["camera_opt_principal_point_regularizer"] = (
        #         self.K_adjustment.norm(dim=0)[2:].norm() * self.config.principal_point_penalty
        #     )

        #     loss_dict["camera_opt_distortion_regularizer"] = (
        #         self.D_adjustment.norm(dim=0).norm() * self.config.distortion_penalty
        #     )

        #     loss_dict["camera_opt_distortion_std_regularizer"] = (
        #         self.D_adjustment.std(dim=0) * self.config.distortion_std_penalty
        #     ).mean()

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
