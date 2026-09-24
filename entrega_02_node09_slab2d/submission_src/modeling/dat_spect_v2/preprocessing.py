from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from modeling.cnn.io import atomic_csv, atomic_json, atomic_npz

from .config import DataConfig, RegistrationMode, stable_hash

REGION_ORDER = (
    "right_caudate",
    "right_putamen",
    "left_caudate",
    "left_putamen",
)


def self_normalize_striatum(
    volume: np.ndarray,
    target_mask: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Project positive striatal voxels onto the sphere of radius sqrt(d)."""
    image = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0)
    selected = np.asarray(target_mask, dtype=bool) & (image > 0)
    result = np.zeros_like(image, dtype=np.float32)
    d = int(selected.sum())
    if d == 0:
        return result
    norm = float(np.linalg.norm(image[selected].astype(np.float64)))
    if not np.isfinite(norm) or norm <= epsilon:
        return result
    result[selected] = math.sqrt(d) * image[selected] / norm
    return result


def _physical_centered_crop(
    array: np.ndarray,
    center_zyx: np.ndarray,
    source_spacing_xyz: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
    output_spacing_mm: float,
    *,
    order: int,
) -> np.ndarray:
    source_spacing_zyx = np.asarray(source_spacing_xyz, dtype=float)[::-1]
    coordinates = []
    for axis, size in enumerate(output_shape_zyx):
        offset_mm = (np.arange(size, dtype=np.float64) - (size - 1.0) / 2.0) * output_spacing_mm
        coordinates.append(center_zyx[axis] + offset_mm / source_spacing_zyx[axis])
    grid = np.meshgrid(*coordinates, indexing="ij")
    return ndimage.map_coordinates(
        np.asarray(array, dtype=np.float32),
        grid,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    ).astype(np.float32)


def physical_striatal_roi(
    volume: np.ndarray,
    *,
    spacing_mm: float,
    radii_zyx_mm: tuple[float, float, float],
    hot_percentile: float,
    recenter_max_mm: float,
) -> np.ndarray:
    """Build a label-independent physical classification ROI around striatal uptake."""
    image = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0)
    shape = np.asarray(image.shape, dtype=float)
    geometric_center = (shape - 1.0) / 2.0
    radii_voxels = np.asarray(radii_zyx_mm, dtype=float) / float(spacing_mm)
    coordinates = np.indices(image.shape, dtype=np.float32)

    def ellipsoid(center: np.ndarray) -> np.ndarray:
        distance = np.zeros(image.shape, dtype=np.float32)
        for axis in range(3):
            distance += ((coordinates[axis] - center[axis]) / radii_voxels[axis]) ** 2
        return distance <= 1.0

    initial = ellipsoid(geometric_center)
    positive = initial & (image > 0)
    if positive.any():
        threshold = float(np.percentile(image[positive], hot_percentile))
        hot = initial & (image >= threshold) & (image > 0)
        if hot.any():
            weights = image[hot].astype(np.float64)
            hot_coordinates = np.argwhere(hot).astype(np.float64)
            uptake_center = np.average(hot_coordinates, axis=0, weights=weights)
            maximum_shift_voxels = float(recenter_max_mm) / float(spacing_mm)
            delta = np.clip(
                uptake_center - geometric_center,
                -maximum_shift_voxels,
                maximum_shift_voxels,
            )
            geometric_center = geometric_center + delta
    return ellipsoid(geometric_center)


def split_striatal_regions(target_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Create fixed label-independent proxy ROIs in canonical LPS array order (z,y,x)."""
    target = np.asarray(target_mask, dtype=bool)
    coordinates = np.argwhere(target)
    if not len(coordinates):
        empty = np.zeros_like(target, dtype=bool)
        return {name: empty.copy() for name in REGION_ORDER}
    # The patient midline is the fixed center of the registered crop. Using the
    # mask median would let an uptake-dependent ROI shift redefine left/right.
    center_y = (target.shape[1] - 1.0) / 2.0
    center_x = (target.shape[2] - 1.0) / 2.0
    yy, xx = np.indices(target.shape[1:])
    # SimpleITK arrays are z,y,x in LPS: low x is patient-right; low y is anterior.
    right = xx[None] < center_x
    left = ~right
    anterior = yy[None] < center_y
    posterior = ~anterior
    return {
        "right_caudate": target & right & anterior,
        "right_putamen": target & right & posterior,
        "left_caudate": target & left & anterior,
        "left_putamen": target & left & posterior,
    }


def regional_targets(volume: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    regions = split_striatal_regions(target_mask)
    values: list[float] = []
    for name in REGION_ORDER:
        selected = np.asarray(volume, dtype=np.float32)[regions[name]]
        values.append(float(np.mean(selected)) if selected.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def canonicalize_by_putamen(
    volume: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    targets = regional_targets(volume, target_mask)
    right_putamen = float(targets[1])
    left_putamen = float(targets[3])
    affected = "right" if right_putamen <= left_putamen else "left"
    # Canonical convention: the weaker putamen is always patient-right (low x).
    flip = affected == "left"
    if not flip:
        return volume.copy(), target_mask.copy(), False, affected
    return (
        np.flip(volume, axis=-1).copy(),
        np.flip(target_mask, axis=-1).copy(),
        True,
        affected,
    )


def axial_slab(
    volume: np.ndarray,
    target_mask: np.ndarray,
    slab_slices: int,
) -> np.ndarray:
    profile = np.asarray(target_mask, dtype=np.float32).sum(axis=(1, 2))
    center = int(np.argmax(ndimage.gaussian_filter1d(profile, sigma=1.0)))
    start = center - slab_slices // 2
    indices = np.arange(start, start + slab_slices).clip(0, volume.shape[0] - 1)
    return np.asarray(volume, dtype=np.float32)[indices].mean(axis=0).astype(np.float32)


def prepare_base_cache(
    crops_dir: Path,
    cohort: pd.DataFrame,
    cache_root: Path,
    *,
    config: DataConfig,
    upstream_hash: str,
) -> pd.DataFrame:
    contract = {
        "preprocessing_version": config.preprocessing_version,
        "upstream_hash": upstream_hash,
        "uids": stable_hash(sorted(cohort["uid"].astype(str).tolist())),
        "output_spacing_mm": config.output_spacing_mm,
        "output_shape_zyx": config.output_shape_zyx,
        "slab_slices": config.slab_slices,
        "striatal_roi_radii_zyx_mm": config.striatal_roi_radii_zyx_mm,
        "striatal_roi_hot_percentile": config.striatal_roi_hot_percentile,
        "striatal_roi_recenter_max_mm": config.striatal_roi_recenter_max_mm,
        "normalization": "sqrt(d)*x/||x||2 over positive voxels in the physical striatal ROI",
    }
    cache_hash = stable_hash(contract)
    cache_dir = cache_root / "base" / cache_hash[:16]
    manifest_path = cache_dir / "manifest.csv"
    config_path = cache_dir / "config.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("cache_hash") != cache_hash:
            raise RuntimeError("El cache base del nodo 07 pertenece a otra configuracion.")
    existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    records = {
        str(record["uid"]): record
        for record in existing.to_dict("records")
        if Path(str(record.get("base_cache_path", ""))).exists()
    }
    pending = cohort.loc[~cohort["uid"].astype(str).isin(records)]
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        uid = str(row.uid)
        source = crops_dir / f"{uid}.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            original = np.asarray(payload["volume_intensity01"], dtype=np.float32)
            target = np.asarray(payload["target_mask"], dtype=np.uint8) > 0
            spacing_xyz = np.asarray(payload["spacing_xyz"], dtype=np.float32)
        coordinates = np.argwhere(target)
        if not len(coordinates):
            raise ValueError(f"{uid}: target_mask vacia.")
        center = coordinates.mean(axis=0)
        volume = _physical_centered_crop(
            original,
            center,
            spacing_xyz,
            config.output_shape_zyx,
            config.output_spacing_mm,
            order=1,
        )
        mask = physical_striatal_roi(
            volume,
            spacing_mm=config.output_spacing_mm,
            radii_zyx_mm=config.striatal_roi_radii_zyx_mm,
            hot_percentile=config.striatal_roi_hot_percentile,
            recenter_max_mm=config.striatal_roi_recenter_max_mm,
        )
        selfnorm = self_normalize_striatum(volume, mask)
        canonical, canonical_mask, was_flipped, affected_side = canonicalize_by_putamen(
            selfnorm, mask
        )
        destination = cache_dir / f"{uid}.npz"
        atomic_npz(
            destination,
            volume_native=selfnorm.astype(config.cache_dtype),
            volume_canonical=canonical.astype(config.cache_dtype),
            slab_native=axial_slab(selfnorm, mask, config.slab_slices).astype(config.cache_dtype),
            slab_canonical=axial_slab(canonical, canonical_mask, config.slab_slices).astype(
                config.cache_dtype
            ),
            target_mask_native=mask.astype(np.uint8),
            target_mask_canonical=canonical_mask.astype(np.uint8),
            auxiliary_native=regional_targets(selfnorm, mask),
            auxiliary_canonical=regional_targets(canonical, canonical_mask),
        )
        records[uid] = {
            "uid": uid,
            "base_cache_path": str(destination.resolve()),
            "lateral_flip_applied": bool(was_flipped),
            "affected_side_proxy": affected_side,
            "selfnorm_positive_voxels": int((mask & (volume > 0)).sum()),
        }
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame(records.values()).sort_values("uid"), manifest_path)
            print(f"[nodo07 base] {len(records)}/{len(cohort)}")
    manifest = pd.DataFrame(records.values()).sort_values("uid").reset_index(drop=True)
    atomic_csv(manifest, manifest_path)
    atomic_json({**contract, "cache_hash": cache_hash, "n_complete": len(manifest)}, config_path)
    return manifest


def audit_base_cache(
    manifest: pd.DataFrame,
    *,
    config: DataConfig,
    destination: Path,
) -> pd.DataFrame:
    """Audit every cached patient before any expensive fold-specific work."""
    inventory = [
        (
            str(row.uid),
            str(Path(row.base_cache_path).resolve()),
            Path(row.base_cache_path).stat().st_size,
            Path(row.base_cache_path).stat().st_mtime_ns,
        )
        for row in manifest.itertuples(index=False)
    ]
    contract = {
        "inventory_hash": stable_hash(inventory),
        "volume_shape_zyx": config.output_shape_zyx,
        "slab_shape_yx": config.output_shape_zyx[1:],
        "selfnorm_relative_tolerance": 5e-3,
        "canonical_rule": "right_putamen_proxy <= left_putamen_proxy",
    }
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("audit_hash") == stable_hash(contract):
            audit = pd.read_csv(destination)
            if len(audit) == len(manifest) and audit["uid"].astype(str).nunique() == len(manifest):
                return audit
    records: list[dict[str, Any]] = []
    for counter, row in enumerate(manifest.itertuples(index=False), start=1):
        with np.load(Path(row.base_cache_path), allow_pickle=False) as payload:
            native = np.asarray(payload["volume_native"], dtype=np.float32)
            canonical = np.asarray(payload["volume_canonical"], dtype=np.float32)
            slab_native = np.asarray(payload["slab_native"], dtype=np.float32)
            slab_canonical = np.asarray(payload["slab_canonical"], dtype=np.float32)
            native_mask = np.asarray(payload["target_mask_native"], dtype=np.uint8) > 0
            canonical_mask = np.asarray(payload["target_mask_canonical"], dtype=np.uint8) > 0
        selected = native_mask & (native > 0)
        d = int(selected.sum())
        norm = float(np.linalg.norm(native[selected].astype(np.float64))) if d else 0.0
        relative_error = abs(norm / math.sqrt(d) - 1.0) if d else float("inf")
        canonical_targets = regional_targets(canonical, canonical_mask)
        finite = bool(
            np.isfinite(native).all()
            and np.isfinite(canonical).all()
            and np.isfinite(slab_native).all()
            and np.isfinite(slab_canonical).all()
        )
        shape_valid = bool(
            native.shape == config.output_shape_zyx
            and canonical.shape == config.output_shape_zyx
            and slab_native.shape == config.output_shape_zyx[1:]
            and slab_canonical.shape == config.output_shape_zyx[1:]
        )
        canonical_rule = bool(canonical_targets[1] <= canonical_targets[3] + 1e-5)
        passed = bool(
            finite
            and shape_valid
            and d > 0
            and relative_error <= contract["selfnorm_relative_tolerance"]
            and canonical_rule
        )
        records.append(
            {
                "uid": str(row.uid),
                "node07_base_qc_pass": passed,
                "node07_base_finite": finite,
                "node07_base_shape_valid": shape_valid,
                "node07_selfnorm_positive_voxels": d,
                "node07_selfnorm_relative_error": relative_error,
                "node07_canonical_laterality_valid": canonical_rule,
            }
        )
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame.from_records(records), destination)
    audit = pd.DataFrame.from_records(records).sort_values("uid").reset_index(drop=True)
    atomic_csv(audit, destination)
    atomic_json(
        {
            **contract,
            "audit_hash": stable_hash(contract),
            "n_cases": len(audit),
            "n_failed": int((~audit["node07_base_qc_pass"]).sum()),
        },
        metadata_path,
    )
    return audit


def _masked_ncc(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    mask = (first.abs() + second.abs()) > 0
    count = mask.sum().clamp_min(8)
    first_mean = (first * mask).sum() / count
    second_mean = (second * mask).sum() / count
    a = (first - first_mean) * mask
    b = (second - second_mean) * mask
    return (a * b).sum() / (torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-8)))


def _load_native(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return (
            np.asarray(payload["volume_native"], dtype=np.float32),
            np.asarray(payload["target_mask_native"], dtype=np.uint8) > 0,
        )


def _template_groups(train_frame: pd.DataFrame) -> dict[str, list[str]]:
    records: list[dict[str, Any]] = []
    for row in train_frame.itertuples(index=False):
        volume, mask = _load_native(row.base_cache_path)
        values = regional_targets(volume, mask)
        records.append(
            {
                "uid": str(row.uid),
                "right": float(values[1]),
                "left": float(values[3]),
                "minimum": float(min(values[1], values[3])),
                "maximum": float(max(values[1], values[3])),
                "signed": float(values[3] - values[1]),
            }
        )
    stats = pd.DataFrame(records)
    low_min = stats["minimum"].quantile(0.25)
    high_min = stats["minimum"].quantile(0.75)
    groups = {
        "normal_like": stats.loc[stats["minimum"] >= high_min, "uid"].tolist(),
        "bilateral_strong": stats.loc[stats["maximum"] <= low_min, "uid"].tolist(),
        "left_reduced": stats.sort_values("signed", ascending=False)["uid"].tolist(),
        "right_reduced": stats.sort_values("signed", ascending=True)["uid"].tolist(),
    }
    return groups


def build_fold_template_bank(
    train_frame: pd.DataFrame,
    destination: Path,
    *,
    config: DataConfig,
    fold_contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    metadata_path = destination.with_suffix(".json")
    expected_hash = stable_hash(
        {**fold_contract, "template_sample": config.template_sample_per_prototype}
    )
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("template_hash") != expected_hash:
            raise RuntimeError("El banco multi-plantilla pertenece a otro fold/contrato.")
        with np.load(destination, allow_pickle=False) as payload:
            names = [str(value) for value in payload["names"].tolist()]
            return payload["templates"], payload["masks"] > 0, names
    groups = _template_groups(train_frame)
    names = ["normal_like", "left_reduced", "right_reduced", "bilateral_strong"]
    templates: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    members: dict[str, list[str]] = {}
    for name in names:
        selected = groups[name][: config.template_sample_per_prototype]
        if not selected:
            selected = train_frame["uid"].astype(str).tolist()[:1]
        lookup = train_frame.set_index(train_frame["uid"].astype(str))["base_cache_path"]
        volumes_and_masks = [_load_native(lookup.loc[uid]) for uid in selected]
        templates.append(np.median(np.stack([value[0] for value in volumes_and_masks]), axis=0))
        masks.append(np.mean(np.stack([value[1] for value in volumes_and_masks]), axis=0) >= 0.5)
        members[name] = selected
    atomic_npz(
        destination,
        templates=np.stack(templates).astype(config.cache_dtype),
        masks=np.stack(masks).astype(np.uint8),
        names=np.asarray(names, dtype="U32"),
    )
    atomic_json(
        {
            **fold_contract,
            "template_hash": expected_hash,
            "members": members,
            "selection": "label-independent posterior uptake quantiles/asymmetry",
        },
        metadata_path,
    )
    return np.stack(templates), np.stack(masks), names


def _affine_refine(
    volume: np.ndarray,
    templates: np.ndarray,
    *,
    config: DataConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    source = torch.from_numpy(volume.astype(np.float32))[None, None].to(device)
    targets = torch.from_numpy(templates.astype(np.float32))[:, None].to(device)
    with torch.no_grad():
        correlations = torch.stack([_masked_ncc(source, target[None]) for target in targets])
        weights = torch.softmax(correlations / config.template_softmax_temperature, dim=0)
        target = (weights[:, None, None, None, None] * targets).sum(dim=0, keepdim=True)
        pre_ncc = float(_masked_ncc(source, target).cpu())
    raw = torch.zeros((1, 3, 4), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=config.affine_learning_rate)
    identity = torch.eye(3, dtype=torch.float32, device=device)[None]
    for _ in range(config.affine_steps):
        linear = identity + config.affine_max_linear_delta * torch.tanh(raw[:, :, :3])
        translation = config.affine_max_translation_fraction * torch.tanh(raw[:, :, 3:])
        theta = torch.cat([linear, translation], dim=2)
        grid = F.affine_grid(theta, source.shape, align_corners=False)
        warped = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        loss = -_masked_ncc(warped, target) + 0.01 * raw.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        linear = identity + config.affine_max_linear_delta * torch.tanh(raw[:, :, :3])
        translation = config.affine_max_translation_fraction * torch.tanh(raw[:, :, 3:])
        theta = torch.cat([linear, translation], dim=2)
        grid = F.affine_grid(theta, source.shape, align_corners=False)
        warped = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        post_ncc = float(_masked_ncc(warped, target).cpu())
        determinant = float(torch.det(linear[0]).cpu())
    return (
        warped[0, 0].cpu().numpy(),
        weights.cpu().numpy(),
        pre_ncc,
        post_ncc,
        determinant,
    )


def prepare_fold_image_cache(
    cohort: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    fold: int,
    cache_root: Path,
    config: DataConfig,
    registration_mode: RegistrationMode,
    upstream_hash: str,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    fold_frame = cohort.merge(
        fold_manifest[["uid", "fold"]].assign(uid=lambda frame: frame["uid"].astype(str)),
        on="uid",
        how="inner",
        validate="one_to_one",
    )
    train_frame = fold_frame.loc[fold_frame["fold"] != fold]
    fold_contract = {
        "fold": fold,
        "registration_mode": registration_mode,
        "train_uids": stable_hash(sorted(train_frame["uid"].astype(str).tolist())),
        "upstream_hash": upstream_hash,
        "affine": {
            "steps": config.affine_steps,
            "learning_rate": config.affine_learning_rate,
            "max_linear_delta": config.affine_max_linear_delta,
            "max_translation_fraction": config.affine_max_translation_fraction,
        },
    }
    fold_hash = stable_hash(fold_contract)
    output_dir = cache_root / "fold_registered" / fold_hash[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    records = {
        str(record["uid"]): record
        for record in existing.to_dict("records")
        if Path(str(record.get("node7_cache_path", ""))).exists()
    }
    templates: np.ndarray | None = None
    if registration_mode == "fold_multitemplate_affine":
        templates, _, template_names = build_fold_template_bank(
            train_frame,
            output_dir / "template_bank.npz",
            config=config,
            fold_contract=fold_contract,
        )
    else:
        template_names = ["single_template_node4"]
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pending = fold_frame.loc[~fold_frame["uid"].astype(str).isin(records)]
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        uid = str(row.uid)
        volume, mask = _load_native(row.base_cache_path)
        weights = np.ones(1, dtype=np.float32)
        pre_ncc = post_ncc = determinant = float("nan")
        if templates is not None:
            volume, weights, pre_ncc, post_ncc, determinant = _affine_refine(
                volume,
                templates,
                config=config,
                device=resolved_device,
            )
            # The target proxy is transformed with a fixed threshold of the registered uptake.
            support = volume > 0
            mask = ndimage.binary_closing(support, iterations=1)
        volume = self_normalize_striatum(volume, mask)
        canonical, canonical_mask, was_flipped, affected_side = canonicalize_by_putamen(
            volume, mask
        )
        destination = output_dir / f"{uid}.npz"
        atomic_npz(
            destination,
            volume_native=volume.astype(config.cache_dtype),
            volume_canonical=canonical.astype(config.cache_dtype),
            slab_native=axial_slab(volume, mask, config.slab_slices).astype(config.cache_dtype),
            slab_canonical=axial_slab(canonical, canonical_mask, config.slab_slices).astype(
                config.cache_dtype
            ),
            target_mask_native=mask.astype(np.uint8),
            target_mask_canonical=canonical_mask.astype(np.uint8),
            auxiliary_native=regional_targets(volume, mask),
            auxiliary_canonical=regional_targets(canonical, canonical_mask),
            template_weights=weights.astype(np.float32),
        )
        record: dict[str, Any] = {
            "uid": uid,
            "node7_cache_path": str(destination.resolve()),
            "lateral_flip_applied": bool(was_flipped),
            "affected_side_proxy": affected_side,
            "registration_pre_ncc": pre_ncc,
            "registration_post_ncc": post_ncc,
            "registration_affine_determinant": determinant,
            "registration_mode": registration_mode,
        }
        record.update(
            {
                f"template_weight_{name}": float(weights[index])
                for index, name in enumerate(template_names)
            }
        )
        records[uid] = record
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame(records.values()).sort_values("uid"), manifest_path)
            print(f"[nodo07 multitemplate fold={fold}] {len(records)}/{len(fold_frame)}")
    manifest = pd.DataFrame(records.values()).sort_values("uid").reset_index(drop=True)
    atomic_csv(manifest, manifest_path)
    atomic_json(
        {**fold_contract, "fold_hash": fold_hash, "n_complete": len(manifest)},
        output_dir / "config.json",
    )
    return manifest
