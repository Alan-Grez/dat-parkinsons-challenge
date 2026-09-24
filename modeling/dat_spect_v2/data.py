from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augmentation import PhysicalUnrealisticAugment
from .config import DataConfig, ModelConfig


def volume_to_triplanar(
    volume: torch.Tensor,
    slices_per_view: int,
    output_size: int,
) -> torch.Tensor:
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError("Se esperaba 1xDxHxW.")
    _, depth, height, width = volume.shape
    half = slices_per_view // 2
    offsets = torch.arange(-half, half + 1)
    if len(offsets) > slices_per_view:
        offsets = offsets[:-1]

    def indices(center: int, size: int) -> torch.Tensor:
        return (offsets + center).clamp(0, size - 1).long()

    axial = volume[0, indices(depth // 2, depth)]
    coronal = volume[0, :, indices(height // 2, height), :].permute(1, 0, 2)
    sagittal = volume[0, :, :, indices(width // 2, width)].permute(2, 0, 1)
    planes = []
    for plane in (axial, coronal, sagittal):
        planes.append(
            F.interpolate(
                plane[None],
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(planes, dim=0)


class DaTSpectDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        model_config: ModelConfig,
        data_config: DataConfig,
        radiomics: np.ndarray | None = None,
        sbr: np.ndarray | None = None,
        sbr_valid: np.ndarray | None = None,
        auxiliary: np.ndarray | None = None,
        augmenter: PhysicalUnrealisticAugment | None = None,
        paired_views: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.model_config = model_config
        self.data_config = data_config
        self.radiomics = radiomics
        self.sbr = sbr
        self.sbr_valid = sbr_valid
        self.auxiliary = auxiliary
        self.augmenter = augmenter
        self.paired_views = paired_views
        for matrix in (radiomics, sbr, sbr_valid, auxiliary):
            if matrix is not None and len(matrix) != len(self.frame):
                raise ValueError("Las matrices tabulares no estan alineadas.")

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, volume: torch.Tensor, slab: torch.Tensor) -> torch.Tensor:
        if self.model_config.architecture == "slab2d":
            return slab
        if self.model_config.architecture == "3d":
            return volume
        return volume_to_triplanar(
            volume,
            self.model_config.slices_per_view,
            self.data_config.view_size_2d,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        suffix = "canonical" if self.model_config.lateral_strategy == "canonical" else "native"
        with np.load(Path(row["node7_cache_path"]), allow_pickle=False) as payload:
            volume = torch.from_numpy(payload[f"volume_{suffix}"].astype(np.float32))[None]
            slab = torch.from_numpy(payload[f"slab_{suffix}"].astype(np.float32))[None]
            target = torch.from_numpy(payload[f"target_mask_{suffix}"].astype(np.float32))[None]
            cached_auxiliary = torch.from_numpy(payload[f"auxiliary_{suffix}"].astype(np.float32))
        if self.model_config.architecture == "slab2d":
            mask = target.amax(dim=1)
            first_source = self.augmenter(slab, mask) if self.augmenter else slab
            first = first_source
        else:
            first_source = self.augmenter(volume, target) if self.augmenter else volume
            first = self._image(first_source, slab)
        result: dict[str, Any] = {
            "uid": str(row["uid"]),
            "image": first,
            "label": torch.tensor(float(row["is_pathologic"]), dtype=torch.float32),
            "auxiliary": (
                torch.from_numpy(self.auxiliary[index])
                if self.auxiliary is not None
                else cached_auxiliary
            ),
        }
        if self.paired_views:
            if self.model_config.architecture == "slab2d":
                second = self.augmenter(slab, mask) if self.augmenter else slab.clone()
            else:
                second_volume = self.augmenter(volume, target) if self.augmenter else volume.clone()
                second = self._image(second_volume, slab)
            result["image_view2"] = second
        if self.radiomics is not None:
            result["radiomics"] = torch.from_numpy(self.radiomics[index])
        if self.sbr is not None:
            result["sbr"] = torch.from_numpy(self.sbr[index])
        if self.sbr_valid is not None:
            result["sbr_valid"] = torch.tensor(float(self.sbr_valid[index]), dtype=torch.float32)
        return result
