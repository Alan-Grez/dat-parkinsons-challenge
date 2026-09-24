from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig, stable_hash
from .features import robust_normalize_volume
from .io import atomic_csv, atomic_json, atomic_npz


def _resize_volume(array: np.ndarray, shape: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=shape,
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    return resized[0, 0].cpu().numpy()


def prepare_image_cache(
    crops_dir: Path,
    cohort: pd.DataFrame,
    cache_root: Path,
    *,
    config: DataConfig,
    upstream_hash: str,
) -> pd.DataFrame:
    cache_contract = {
        "upstream_hash": upstream_hash,
        "shape": config.image_shape_3d,
        "normalization": {
            "clip_percentiles": config.clip_percentiles,
            "scale_percentiles": config.scale_percentiles,
            "normalized_clip": config.normalized_clip,
        },
        "uid_hash": stable_hash(sorted(cohort["uid"].astype(str).tolist())),
    }
    cache_hash = stable_hash(cache_contract)
    cache_dir = cache_root / cache_hash[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "image_cache_manifest.csv"
    config_path = cache_dir / "image_cache_config.json"
    if config_path.exists():
        import json

        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("cache_hash") != cache_hash:
            raise RuntimeError("El cache de imagen existente pertenece a otra configuracion.")
    existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    valid_existing: dict[str, dict[str, Any]] = {}
    for record in existing.to_dict("records"):
        path = Path(str(record.get("cache_path", "")))
        if path.exists():
            valid_existing[str(record["uid"])] = record
    records = list(valid_existing.values())
    processed = set(valid_existing)
    pending = cohort.loc[~cohort["uid"].astype(str).isin(processed)]
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        uid = str(row.uid)
        source = crops_dir / f"{uid}.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            volume = robust_normalize_volume(payload["volume_intensity01"], config)
            target = np.asarray(payload["target_mask"], dtype=np.uint8)
        volume = _resize_volume(volume, config.image_shape_3d, "trilinear")
        target = _resize_volume(target, config.image_shape_3d, "nearest") >= 0.5
        destination = cache_dir / f"{uid}.npz"
        atomic_npz(
            destination,
            volume=volume.astype(config.cache_dtype),
            target_mask=target.astype(np.uint8),
        )
        records.append(
            {
                "uid": uid,
                "cache_path": str(destination.resolve()),
                "source_size": int(source.stat().st_size),
                "source_mtime_ns": int(source.stat().st_mtime_ns),
            }
        )
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame(records).sort_values("uid"), manifest_path)
            print(f"[CNN cache] imagenes preparadas: {len(records)}/{len(cohort)}")
    manifest = pd.DataFrame(records).sort_values("uid").reset_index(drop=True)
    atomic_csv(manifest, manifest_path)
    atomic_json(
        {**cache_contract, "cache_hash": cache_hash, "n_complete": len(manifest)},
        config_path,
    )
    return manifest


def _rotation_matrix_xyz(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    rot_x = torch.stack(
        [one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=-1
    ).reshape(-1, 3, 3)
    rot_y = torch.stack(
        [cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=-1
    ).reshape(-1, 3, 3)
    rot_z = torch.stack(
        [cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=-1
    ).reshape(-1, 3, 3)
    return rot_z @ rot_y @ rot_x


class InvarianceAugment3D:
    def __init__(self, train_config: Any) -> None:
        self.config = train_config

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 4:
            raise ValueError("Se esperaba CxDxHxW.")
        result = volume.clone()
        if random.random() < self.config.left_right_flip_probability:
            result = torch.flip(result, dims=(-1,))
        device = result.device
        maximum = math.radians(self.config.rotation_degrees)
        angles = torch.empty(3, device=device).uniform_(-maximum, maximum)
        rotation = _rotation_matrix_xyz(angles[0:1], angles[1:2], angles[2:3])[0]
        scale = random.uniform(*self.config.scale_range)
        theta = torch.zeros((1, 3, 4), dtype=result.dtype, device=device)
        theta[0, :, :3] = rotation * scale
        theta[0, :, 3] = torch.empty(3, device=device).uniform_(
            -self.config.translation_fraction,
            self.config.translation_fraction,
        )
        batch = result.unsqueeze(0)
        grid = F.affine_grid(theta, batch.shape, align_corners=False)
        result = F.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        gain = random.uniform(*self.config.intensity_gain_range)
        gamma = random.uniform(*self.config.gamma_range)
        result = gain * torch.sign(result) * torch.abs(result).clamp_min(1e-6).pow(gamma)
        if self.config.noise_std > 0:
            result = result + torch.randn_like(result) * self.config.noise_std
        if random.random() < self.config.blur_probability:
            result = F.avg_pool3d(result[None], kernel_size=3, stride=1, padding=1)[0]
        if random.random() < self.config.resolution_probability:
            low = F.interpolate(
                result[None], scale_factor=0.75, mode="trilinear", align_corners=False
            )
            result = F.interpolate(
                low, size=result.shape[-3:], mode="trilinear", align_corners=False
            )[0]
        return result.clamp(-3.5, 3.5)


def volume_to_triplanar(
    volume: torch.Tensor,
    slices_per_view: int,
    output_size: int,
) -> torch.Tensor:
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError("Se esperaba un volumen 1xDxHxW.")
    _, depth, height, width = volume.shape
    half = slices_per_view // 2
    offsets = torch.arange(-half, half + 1)
    if len(offsets) > slices_per_view:
        offsets = offsets[:-1]

    def indices(center: int, size: int) -> torch.Tensor:
        return (offsets + center).clamp(0, size - 1).long()

    axial = volume[0, indices(depth // 2, depth), :, :]
    coronal = volume[0, :, indices(height // 2, height), :].permute(1, 0, 2)
    sagittal = volume[0, :, :, indices(width // 2, width)].permute(2, 0, 1)
    planes = []
    for plane in (axial, coronal, sagittal):
        resized = F.interpolate(
            plane[None], size=(output_size, output_size), mode="bilinear", align_corners=False
        )[0]
        planes.append(resized)
    return torch.stack(planes, dim=0)


class CropDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        model_config: ModelConfig,
        data_config: DataConfig,
        radiomics: np.ndarray | None = None,
        sbr: np.ndarray | None = None,
        sbr_valid: np.ndarray | None = None,
        augmenter: InvarianceAugment3D | None = None,
        paired_views: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.model_config = model_config
        self.data_config = data_config
        self.radiomics = radiomics
        self.sbr = sbr
        self.sbr_valid = sbr_valid
        self.augmenter = augmenter
        self.paired_views = paired_views
        expected = len(self.frame)
        for values in (radiomics, sbr, sbr_valid):
            if values is not None and len(values) != expected:
                raise ValueError("Las matrices tabulares no estan alineadas con el dataset.")

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, volume: torch.Tensor) -> torch.Tensor:
        if self.model_config.architecture == "3d":
            return volume
        return volume_to_triplanar(
            volume,
            self.model_config.slices_per_view,
            self.data_config.view_size_2d,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with np.load(Path(row["cache_path"]), allow_pickle=False) as payload:
            volume = torch.from_numpy(payload["volume"].astype(np.float32))[None]
            target_mask = torch.from_numpy(payload["target_mask"].astype(np.float32))[None]
        first = self.augmenter(volume) if self.augmenter is not None else volume
        result: dict[str, Any] = {
            "uid": str(row["uid"]),
            "image": self._image(first),
            "label": torch.tensor(float(row["is_pathologic"]), dtype=torch.float32),
            "target_mask": target_mask,
        }
        if self.paired_views:
            second = self.augmenter(volume) if self.augmenter is not None else volume.clone()
            result["image_view2"] = self._image(second)
        if self.radiomics is not None:
            result["radiomics"] = torch.from_numpy(self.radiomics[index])
        if self.sbr is not None:
            result["sbr"] = torch.from_numpy(self.sbr[index])
        if self.sbr_valid is not None:
            result["sbr_valid"] = torch.tensor(
                float(self.sbr_valid[index]), dtype=torch.float32
            )
        return result
