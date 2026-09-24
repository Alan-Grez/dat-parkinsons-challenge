from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ImageConfig:
    image_shape_3d: tuple[int, int, int]
    view_size_2d: int
    slices_per_view: int
    clip_percentiles: tuple[float, float]
    scale_percentiles: tuple[float, float]
    normalized_clip: tuple[float, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageConfig":
        return cls(
            image_shape_3d=tuple(int(v) for v in payload["image_shape_3d"]),
            view_size_2d=int(payload["view_size_2d"]),
            slices_per_view=int(payload["slices_per_view"]),
            clip_percentiles=tuple(float(v) for v in payload["clip_percentiles"]),
            scale_percentiles=tuple(float(v) for v in payload["scale_percentiles"]),
            normalized_clip=tuple(float(v) for v in payload["normalized_clip"]),
        )


def _image_from_array(
    array: np.ndarray,
    *,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))
    image.SetDirection(tuple(float(v) for v in direction))
    return image


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    mask = np.isfinite(first) & np.isfinite(second) & ((first > 0) | (second > 0))
    if int(mask.sum()) < 10:
        return float("nan")
    a = first[mask].astype(np.float64)
    b = second[mask].astype(np.float64)
    if float(a.std()) <= 1e-8 or float(b.std()) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rescale01(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.min(array[finite]))
    high = float(np.max(array[finite]))
    result = (np.asarray(array, dtype=np.float32) - low) / max(high - low, 1e-8)
    result[~finite] = 0.0
    return result


def _rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    ax, ay, az = angles
    one = torch.ones((), device=angles.device, dtype=angles.dtype)
    zero = torch.zeros((), device=angles.device, dtype=angles.dtype)
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    rotation_x = torch.stack(
        [one, zero, zero, zero, cx, -sx, zero, sx, cx]
    ).reshape(3, 3)
    rotation_y = torch.stack(
        [cy, zero, sy, zero, one, zero, -sy, zero, cy]
    ).reshape(3, 3)
    rotation_z = torch.stack(
        [cz, -sz, zero, sz, cz, zero, zero, zero, one]
    ).reshape(3, 3)
    return rotation_z @ rotation_y @ rotation_x


def _warp(volume: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    theta = torch.cat(
        [_rotation_matrix(parameters[:3]), parameters[3:, None]], dim=1
    )[None]
    grid = F.affine_grid(theta, volume.shape, align_corners=False)
    return F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def _ncc_loss(fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    valid = (fixed > 0.01) | (moving.detach() > 0.01)
    first = fixed[valid]
    second = moving[valid]
    if first.numel() < 10:
        return moving.sum() * 0 + 1
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.sqrt(
        first.square().mean() * second.square().mean()
    ).clamp_min(1e-6)
    return -(first * second).mean() / denominator


class SubmissionPreprocessor:
    def __init__(self, template_path: Path, config: ImageConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        with np.load(template_path, allow_pickle=False) as payload:
            fixed = payload["fixed"].astype(np.float32)
            spacing = payload["spacing_xyz"].astype(np.float64)
            origin = payload["origin_xyz"].astype(np.float64)
            direction = payload["direction"].astype(np.float64)
            self.brain_mask = payload["brain_mask"].astype(bool)
            self.crop_start = payload["crop_start_zyx"].astype(int)
            self.crop_stop = payload["crop_stop_zyx"].astype(int)
        self.fixed = _image_from_array(
            fixed, spacing=spacing, origin=origin, direction=direction
        )
        self.fixed_array01 = _rescale01(fixed)
        self.fixed_tensor = (
            torch.from_numpy(self.fixed_array01)[None, None].to(device)
            if device.type == "cuda"
            else None
        )
        self.registration_stages = (
            (0.25, 28, 0.04),
            (0.50, 20, 0.025),
            (1.00, 12, 0.012),
        )
        self.early_stopping_patience = 7

    def _centered(self, moving: sitk.Image) -> sitk.Image:
        initial = sitk.CenteredTransformInitializer(
            self.fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        return sitk.Resample(
            moving, self.fixed, initial, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )

    def _register_cuda(self, moving: sitk.Image) -> tuple[np.ndarray, float, float]:
        centered = self._centered(moving)
        original = sitk.GetArrayFromImage(centered).astype(np.float32)
        moving01 = _rescale01(original)
        before = _normalized_correlation(self.fixed_array01, moving01)
        moving_tensor = torch.from_numpy(moving01)[None, None].to(self.device)
        parameters = torch.zeros(6, device=self.device, requires_grad=True)
        best_loss = math.inf
        for scale, iterations, learning_rate in self.registration_stages:
            level_size = [
                max(12, int(round(size * scale)))
                for size in self.fixed_tensor.shape[2:]
            ]
            fixed_level = F.interpolate(
                self.fixed_tensor,
                size=level_size,
                mode="trilinear",
                align_corners=False,
            )
            moving_level = F.interpolate(
                moving_tensor,
                size=level_size,
                mode="trilinear",
                align_corners=False,
            )
            optimizer = torch.optim.Adam([parameters], lr=learning_rate)
            stage_best = math.inf
            stage_parameters = parameters.detach().clone()
            stale = 0
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                warped = _warp(moving_level, parameters)
                loss = _ncc_loss(fixed_level, warped)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite CUDA registration loss.")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    parameters[:3].clamp_(-0.35, 0.35)
                    parameters[3:].clamp_(-0.35, 0.35)
                current = float(loss.detach().cpu())
                if current < stage_best - 1e-5:
                    stage_best = current
                    stage_parameters = parameters.detach().clone()
                    stale = 0
                else:
                    stale += 1
                if stale >= self.early_stopping_patience:
                    break
            with torch.no_grad():
                parameters.copy_(stage_parameters)
            best_loss = stage_best
        with torch.inference_mode():
            registered = (
                _warp(torch.from_numpy(original)[None, None].to(self.device), parameters)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            registered01 = (
                _warp(moving_tensor, parameters)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        after = _normalized_correlation(self.fixed_array01, registered01)
        return registered, before, after

    def _register_sitk(self, moving: sitk.Image) -> np.ndarray:
        initial = sitk.CenteredTransformInitializer(
            self.fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsCorrelation()
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(0.10, 20260821)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,
            gradientMagnitudeTolerance=1e-8,
        )
        method.SetOptimizerScalesFromPhysicalShift()
        method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(initial, inPlace=False)
        transform = method.Execute(
            sitk.RescaleIntensity(sitk.Cast(self.fixed, sitk.sitkFloat32), 0.0, 1.0),
            sitk.RescaleIntensity(sitk.Cast(moving, sitk.sitkFloat32), 0.0, 1.0),
        )
        registered = sitk.Resample(
            moving, self.fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        return sitk.GetArrayFromImage(registered).astype(np.float32)

    def register(self, path: Path) -> np.ndarray:
        moving = sitk.Image(sitk.DICOMOrient(sitk.ReadImage(str(path), sitk.sitkFloat32), "LPS"))
        if self.device.type == "cuda":
            try:
                registered, before, after = self._register_cuda(moving)
                unreliable = (
                    not np.isfinite(after)
                    or after < 0.35
                    or (np.isfinite(before) and after < before - 0.03)
                )
                if not unreliable:
                    return registered
            except RuntimeError:
                torch.cuda.empty_cache()
        try:
            return self._register_sitk(moving)
        except RuntimeError:
            centered = self._centered(moving)
            return sitk.GetArrayFromImage(centered).astype(np.float32)

    def _intensity01_crop(self, registered: np.ndarray) -> np.ndarray:
        positive = np.isfinite(registered) & (registered > 0)
        foreground = self.brain_mask & positive
        if int(foreground.sum()) < 64:
            foreground = positive
        values = registered[foreground]
        if values.size < 32:
            raise ValueError("Insufficient positive foreground after registration.")
        p99 = float(np.percentile(values, 99.0))
        intensity01 = np.clip(registered / max(p99, 1e-8), 0.0, 1.0)
        z0, y0, x0 = self.crop_start
        z1, y1, x1 = self.crop_stop
        return intensity01[z0:z1, y0:y1, x0:x1].astype(np.float32)

    def _robust_normalize(self, volume: np.ndarray) -> np.ndarray:
        array = np.asarray(volume, dtype=np.float32)
        finite_positive = np.isfinite(array) & (array > 0)
        values = array[finite_positive]
        if values.size < 32:
            return np.zeros_like(array, dtype=np.float32)
        clip_low, clip_high = np.percentile(values, self.config.clip_percentiles)
        scale_low, median, scale_high = np.percentile(
            values,
            [self.config.scale_percentiles[0], 50.0, self.config.scale_percentiles[1]],
        )
        normalized = (
            np.clip(array, clip_low, clip_high) - median
        ) / max(float(scale_high - scale_low), 1e-6)
        normalized = np.clip(normalized, *self.config.normalized_clip)
        normalized[~finite_positive] = 0.0
        return normalized.astype(np.float32, copy=False)

    def _triplanar(self, volume: torch.Tensor) -> torch.Tensor:
        _, depth, height, width = volume.shape
        half = self.config.slices_per_view // 2
        offsets = torch.arange(-half, half + 1)
        if len(offsets) > self.config.slices_per_view:
            offsets = offsets[:-1]

        def indices(center: int, size: int) -> torch.Tensor:
            return (offsets + center).clamp(0, size - 1).long()

        axial = volume[0, indices(depth // 2, depth), :, :]
        coronal = volume[0, :, indices(height // 2, height), :].permute(1, 0, 2)
        sagittal = volume[0, :, :, indices(width // 2, width)].permute(2, 0, 1)
        planes = []
        for plane in (axial, coronal, sagittal):
            planes.append(
                F.interpolate(
                    plane[None],
                    size=(self.config.view_size_2d, self.config.view_size_2d),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
        return torch.stack(planes, dim=0)

    def transform(self, path: Path) -> torch.Tensor:
        crop = self._intensity01_crop(self.register(path))
        normalized = torch.from_numpy(self._robust_normalize(crop))[None, None]
        resized = F.interpolate(
            normalized,
            size=self.config.image_shape_3d,
            mode="trilinear",
            align_corners=False,
        )[0]
        return self._triplanar(resized)
