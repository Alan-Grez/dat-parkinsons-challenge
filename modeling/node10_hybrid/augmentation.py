from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modeling.dat_spect_v2.augmentation import _separable_gaussian


def self_normalize_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Project positive masked voxels onto the sqrt(d) L2 sphere."""

    positive = (mask > 0.5) & (image > 0)
    d = positive.sum().clamp_min(1).to(dtype=image.dtype)
    norm = torch.sqrt((image.square() * positive).sum()).clamp_min(1e-8)
    return torch.where(positive, image * torch.sqrt(d) / norm, torch.zeros_like(image))


@dataclass(frozen=True)
class PairedAugmentationConfig:
    spacing_mm: float = 2.0
    probability: float = 0.90
    magnitude: float = 2.5
    blur_base_fwhm_mm: tuple[float, float] = (1.0, 6.0)
    correlated_noise_fwhm_mm: tuple[float, float] = (4.0, 12.0)
    correlated_noise_sd_fraction: tuple[float, float] = (0.02, 0.08)
    rotation_degrees: float = 2.0
    translation_fraction: float = 0.025
    intensity_gain_range: tuple[float, float] = (0.90, 1.10)


class PairedPhysicalAugment:
    """Synchronized 3D augmentation for intensity and pattern branches.

    Geometry and the spatial noise field are shared, so the branches remain
    registered. The self-normalized branch is reprojected after perturbation.
    Left/right flips are deliberately excluded because the auxiliary target
    includes affected-side laterality.
    """

    def __init__(self, config: PairedAugmentationConfig) -> None:
        self.config = config

    def _geometry(
        self,
        intensity: torch.Tensor,
        pattern: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum = math.radians(self.config.rotation_degrees)
        angles = [random.uniform(-maximum, maximum) for _ in range(3)]
        cx, sx = math.cos(angles[0]), math.sin(angles[0])
        cy, sy = math.cos(angles[1]), math.sin(angles[1])
        cz, sz = math.cos(angles[2]), math.sin(angles[2])
        rx = intensity.new_tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = intensity.new_tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = intensity.new_tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        theta = intensity.new_zeros((1, 3, 4))
        theta[0, :, :3] = rz @ ry @ rx
        theta[0, :, 3] = intensity.new_tensor(
            [
                random.uniform(-self.config.translation_fraction, self.config.translation_fraction)
                for _ in range(3)
            ]
        )
        shape = (1, *intensity.shape)
        grid = F.affine_grid(theta, shape, align_corners=False)

        def sample(value: torch.Tensor, mode: str) -> torch.Tensor:
            return F.grid_sample(
                value[None],
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )[0]

        return sample(intensity, "bilinear"), sample(pattern, "bilinear"), sample(mask, "nearest")

    def __call__(
        self,
        intensity: torch.Tensor,
        pattern: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intensity.ndim != 4 or pattern.shape != intensity.shape or mask.shape != intensity.shape:
            raise ValueError("El augmentation dual espera tres tensores CDHW alineados.")
        intensity, pattern, mask = self._geometry(intensity, pattern, mask)
        if random.random() < self.config.probability:
            base_fwhm = random.uniform(*self.config.blur_base_fwhm_mm)
            sigma = self.config.magnitude * base_fwhm / 2.354820045 / self.config.spacing_mm
            intensity = _separable_gaussian(intensity[None], [sigma] * 3)[0]
            pattern = _separable_gaussian(pattern[None], [sigma] * 3)[0]
        if random.random() < self.config.probability:
            noise = torch.randn_like(intensity)[None]
            noise_fwhm = random.uniform(*self.config.correlated_noise_fwhm_mm)
            sigma = noise_fwhm / 2.354820045 / self.config.spacing_mm
            noise = _separable_gaussian(noise, [sigma] * 3)[0]
            selected = mask > 0.5
            if selected.sum() > 1:
                noise = noise / noise[selected].std(unbiased=False).clamp_min(1e-6)
                fraction = self.config.magnitude * random.uniform(
                    *self.config.correlated_noise_sd_fraction
                )
                raw_scale = intensity[selected].std(unbiased=False).clamp_min(1e-6)
                pattern_scale = pattern[selected].std(unbiased=False).clamp_min(1e-6)
                intensity = intensity + noise * fraction * raw_scale
                pattern = pattern + noise * fraction * pattern_scale
        gain = random.uniform(*self.config.intensity_gain_range)
        selected = mask > 0.5
        intensity = intensity.clamp_min(0) * selected * gain
        pattern = pattern.clamp_min(0) * selected
        return intensity, self_normalize_tensor(pattern, mask)
