from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .config import AugmentationConfig, LateralStrategy


def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 1e-6:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def _separable_gaussian(
    image: torch.Tensor,
    sigma_voxels: Sequence[float],
) -> torch.Tensor:
    dimensions = image.ndim - 2
    if dimensions not in (2, 3):
        raise ValueError("El blur fisico espera NCHW o NCDHW.")
    output = image
    for axis, sigma in enumerate(sigma_voxels):
        kernel = _gaussian_kernel1d(float(sigma), image.dtype, image.device)
        if len(kernel) == 1:
            continue
        shape = [1] * dimensions
        shape[axis] = len(kernel)
        weight = kernel.reshape(1, 1, *shape).repeat(image.shape[1], 1, *([1] * dimensions))
        padding = [0] * dimensions
        padding[axis] = len(kernel) // 2
        if dimensions == 2:
            output = F.conv2d(output, weight, padding=tuple(padding), groups=image.shape[1])
        else:
            output = F.conv3d(output, weight, padding=tuple(padding), groups=image.shape[1])
    return output


def _self_normalize_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive = (mask > 0.5) & (image > 0)
    flat = image.flatten(1)
    selected = positive.flatten(1)
    d = selected.sum(dim=1).clamp_min(1).to(dtype=image.dtype)
    norm = torch.sqrt((flat.square() * selected).sum(dim=1).clamp_min(1e-8))
    scale = torch.sqrt(d) / norm
    view_shape = (len(image),) + (1,) * (image.ndim - 1)
    return torch.where(positive, image * scale.reshape(view_shape), torch.zeros_like(image))


def _affine_2d(
    image: torch.Tensor, mask: torch.Tensor, config: AugmentationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = math.radians(random.uniform(-config.rotation_degrees, config.rotation_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    theta = image.new_tensor(
        [
            [
                [
                    cosine,
                    -sine,
                    random.uniform(-config.translation_fraction, config.translation_fraction),
                ],
                [
                    sine,
                    cosine,
                    random.uniform(-config.translation_fraction, config.translation_fraction),
                ],
            ]
        ]
    )
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return (
        F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False),
        F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False),
    )


def _affine_3d(
    image: torch.Tensor, mask: torch.Tensor, config: AugmentationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = math.radians(config.rotation_degrees)
    angles = [random.uniform(-maximum, maximum) for _ in range(3)]
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])
    rx = image.new_tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = image.new_tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = image.new_tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    theta = image.new_zeros((1, 3, 4))
    theta[0, :, :3] = rz @ ry @ rx
    theta[0, :, 3] = image.new_tensor(
        [
            random.uniform(-config.translation_fraction, config.translation_fraction)
            for _ in range(3)
        ]
    )
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return (
        F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False),
        F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False),
    )


class PhysicalUnrealisticAugment:
    """Strong DaT-SPECT augmentation parameterized in physical millimetres."""

    def __init__(
        self,
        config: AugmentationConfig,
        *,
        spacing_mm: float,
        lateral_strategy: LateralStrategy,
    ) -> None:
        self.config = config
        self.spacing_mm = float(spacing_mm)
        self.lateral_strategy = lateral_strategy

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if image.ndim not in (3, 4):
            raise ValueError("Se esperaba CHW o CDHW.")
        batch = image[None].clone()
        mask_batch = mask[None].to(dtype=image.dtype)
        if (
            self.lateral_strategy == "random_flip"
            and random.random() < self.config.left_right_flip_probability
        ):
            batch = torch.flip(batch, dims=(-1,))
            mask_batch = torch.flip(mask_batch, dims=(-1,))
        if self.config.rotation_degrees > 0 or self.config.translation_fraction > 0:
            if image.ndim == 3:
                batch, mask_batch = _affine_2d(batch, mask_batch, self.config)
            else:
                batch, mask_batch = _affine_3d(batch, mask_batch, self.config)
        dimensions = image.ndim - 1
        if random.random() < self.config.probability:
            base = random.uniform(*self.config.blur_base_fwhm_mm)
            fwhm_mm = self.config.magnitude * base
            sigma = fwhm_mm / 2.354820045 / self.spacing_mm
            batch = _separable_gaussian(batch, [sigma] * dimensions)
        if random.random() < self.config.probability:
            noise = torch.randn_like(batch)
            noise_fwhm = random.uniform(*self.config.correlated_noise_fwhm_mm)
            noise_sigma = noise_fwhm / 2.354820045 / self.spacing_mm
            noise = _separable_gaussian(noise, [noise_sigma] * dimensions)
            selected = mask_batch > 0.5
            noise_values = noise[selected]
            image_values = batch[selected]
            if noise_values.numel() > 1 and image_values.numel() > 1:
                noise = noise / noise_values.std(unbiased=False).clamp_min(1e-6)
                fraction = self.config.magnitude * random.uniform(
                    *self.config.correlated_noise_sd_fraction
                )
                batch = batch + noise * fraction * image_values.std(unbiased=False)
        batch = batch.clamp_min(0.0) * (mask_batch > 0.5)
        batch = _self_normalize_tensor(batch, mask_batch)
        return batch[0]
