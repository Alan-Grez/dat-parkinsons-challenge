from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area

from .config import DataConfig, stable_hash
from .io import atomic_csv, atomic_json

SBR_FEATURE_COLUMNS = [
    "semiquant_sbr",
    "semiquant_right_sbr",
    "semiquant_left_sbr",
    "semiquant_min_side_sbr",
    "semiquant_log_target_background",
    "firstorder_ratio_mean",
    "firstorder_ratio_std",
    "firstorder_ratio_p10",
    "firstorder_ratio_p50",
    "firstorder_ratio_p90",
    "active_threshold_sbr",
]


def robust_normalize_volume(volume: np.ndarray, config: DataConfig) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    finite_positive = np.isfinite(array) & (array > 0)
    values = array[finite_positive]
    if values.size < 32:
        return np.zeros_like(array, dtype=np.float32)
    clip_low, clip_high = np.percentile(values, config.clip_percentiles)
    scale_low, median, scale_high = np.percentile(
        values,
        [config.scale_percentiles[0], 50.0, config.scale_percentiles[1]],
    )
    clipped = np.clip(array, clip_low, clip_high)
    normalized = (clipped - median) / max(float(scale_high - scale_low), 1e-6)
    normalized = np.clip(normalized, *config.normalized_clip)
    normalized[~finite_positive] = 0.0
    return normalized.astype(np.float32, copy=False)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-6))


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    if values.size < 2 or float(np.nanmax(values)) <= float(np.nanmin(values)):
        return 0.0
    histogram, _ = np.histogram(values, bins=bins)
    probabilities = histogram[histogram > 0].astype(float)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _paired_slices(size: int, offset: int) -> tuple[slice, slice]:
    if offset > 0:
        return slice(0, size - offset), slice(offset, size)
    if offset < 0:
        return slice(-offset, size), slice(0, size + offset)
    return slice(0, size), slice(0, size)


def _texture_glcm_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    levels: int,
) -> dict[str, float]:
    values = volume[mask & np.isfinite(volume)]
    names = ("contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation")
    if values.size < 30 or float(values.max()) <= float(values.min()):
        return {f"core_texture3d_{name}": float("nan") for name in names}
    low, high = np.percentile(values, [1.0, 99.0])
    quantized = np.rint(
        np.clip((volume - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        * (levels - 1)
    ).astype(np.uint8)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    ]
    for dz, dy, dx in offsets:
        z0, z1 = _paired_slices(mask.shape[0], dz)
        y0, y1 = _paired_slices(mask.shape[1], dy)
        x0, x1 = _paired_slices(mask.shape[2], dx)
        valid = mask[z0, y0, x0] & mask[z1, y1, x1]
        if not valid.any():
            continue
        first = quantized[z0, y0, x0][valid]
        second = quantized[z1, y1, x1][valid]
        np.add.at(matrix, (first, second), 1)
        np.add.at(matrix, (second, first), 1)
    if matrix.sum() == 0:
        return {f"core_texture3d_{name}": float("nan") for name in names}
    probability = matrix / matrix.sum()
    i, j = np.indices(probability.shape)
    nonzero = probability[probability > 0]
    row = probability.sum(axis=1)
    column = probability.sum(axis=0)
    axis = np.arange(levels)
    mean_i = float((axis * row).sum())
    mean_j = float((axis * column).sum())
    std_i = math.sqrt(float((((axis - mean_i) ** 2) * row).sum()))
    std_j = math.sqrt(float((((axis - mean_j) ** 2) * column).sum()))
    return {
        "core_texture3d_contrast": float((probability * (i - j) ** 2).sum()),
        "core_texture3d_dissimilarity": float((probability * np.abs(i - j)).sum()),
        "core_texture3d_homogeneity": float(
            (probability / (1.0 + (i - j) ** 2)).sum()
        ),
        "core_texture3d_energy": float(np.sqrt((probability**2).sum())),
        "core_texture3d_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
        "core_texture3d_correlation": float(
            (probability * (i - mean_i) * (j - mean_j)).sum()
            / max(std_i * std_j, 1e-6)
        ),
    }


def _projection_features(volume: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masked = np.where(mask, volume, -np.inf)
    result: dict[str, float] = {}
    for axis, name in enumerate(("z", "y", "x")):
        projection = np.max(masked, axis=axis)
        values = projection[np.isfinite(projection)]
        result[f"core_mip_{name}_mean"] = float(np.mean(values)) if values.size else 0.0
        result[f"core_mip_{name}_std"] = float(np.std(values)) if values.size else 0.0
        result[f"core_mip_{name}_entropy"] = _entropy(values)
    return result


def _shape_features(
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    names = ("volume_ml", "surface_mm2", "sphericity", "elongation", "extent")
    coordinates = np.argwhere(mask)
    if len(coordinates) < 8:
        return {f"core_highuptake_{name}": float("nan") for name in names}
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    voxel_volume = float(np.prod(spacing_zyx))
    volume_mm3 = float(mask.sum() * voxel_volume)
    physical = coordinates * spacing_zyx[None]
    covariance = np.cov(physical, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
    elongation = float(
        np.sqrt(max(float(eigenvalues[-1]), 1e-8) / max(float(eigenvalues[0]), 1e-8))
    )
    spans = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    bounding_volume = float(np.prod(spans * spacing_zyx))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            vertices, faces, _normals, _values = marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=tuple(spacing_zyx)
            )
        surface = float(mesh_surface_area(vertices, faces))
    except (RuntimeError, ValueError):
        surface = float("nan")
    sphericity = (
        float(np.pi ** (1 / 3) * (6.0 * volume_mm3) ** (2 / 3) / surface)
        if np.isfinite(surface) and surface > 0
        else float("nan")
    )
    return {
        "core_highuptake_volume_ml": volume_mm3 / 1000.0,
        "core_highuptake_surface_mm2": surface,
        "core_highuptake_sphericity": sphericity,
        "core_highuptake_elongation": elongation,
        "core_highuptake_extent": volume_mm3 / max(bounding_volume, 1e-8),
    }


def extract_background_independent_features(
    volume: np.ndarray,
    target_mask: np.ndarray,
    config: DataConfig,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    normalized = robust_normalize_volume(volume, config)
    target = np.asarray(target_mask, dtype=bool) & np.isfinite(normalized)
    values = normalized[target]
    if values.size < 32:
        raise ValueError("La mascara target tiene soporte insuficiente.")
    p10, p25, p50, p75, p90, p99 = np.percentile(values, [10, 25, 50, 75, 90, 99])
    features: dict[str, float] = {
        "core_target_mean": float(values.mean()),
        "core_target_std": float(values.std()),
        "core_target_p10": float(p10),
        "core_target_p25": float(p25),
        "core_target_p50": float(p50),
        "core_target_p75": float(p75),
        "core_target_p90": float(p90),
        "core_target_p99": float(p99),
        "core_target_iqr": float(p75 - p25),
        "core_target_robust_width": float(p90 - p10),
        "core_target_p90_p50_ratio": _safe_ratio(float(p90), float(p50)),
        "core_target_p99_p50_ratio": _safe_ratio(float(p99), float(p50)),
        "core_target_entropy": _entropy(values, bins=config.texture_levels),
    }

    coordinates = np.argwhere(target)
    center = coordinates.mean(axis=0)
    _z_grid, y_grid, x_grid = np.indices(target.shape)
    # SimpleITK physical coordinates use LPS: increasing X points left and
    # increasing Y points posterior. The array is z,y,x after registration.
    right = target & (x_grid < center[2])
    left = target & ~right
    anterior = target & (y_grid < center[1])
    posterior = target & ~anterior
    left_mean = float(normalized[left].mean()) if left.any() else 0.0
    right_mean = float(normalized[right].mean()) if right.any() else 0.0
    posterior_mean = float(normalized[posterior].mean()) if posterior.any() else 0.0
    anterior_mean = float(normalized[anterior].mean()) if anterior.any() else 0.0
    side_mean = (abs(left_mean) + abs(right_mean)) / 2.0
    features.update(
        {
            "core_lr_asymmetry_abs": float(
                abs(left_mean - right_mean) / max(side_mean, 1e-6)
            ),
            "core_posterior_anterior_ratio": _safe_ratio(
                posterior_mean, anterior_mean
            ),
        }
    )

    threshold = float(np.percentile(values, config.high_uptake_percentile))
    high = target & (normalized >= threshold)
    high = ndimage.binary_closing(high, iterations=1)
    labels, components = ndimage.label(high)
    sizes = np.bincount(labels.ravel())[1:] if components else np.asarray([], dtype=int)
    largest_fraction = float(sizes.max() / max(high.sum(), 1)) if sizes.size else 0.0
    high_coordinates = np.argwhere(high)
    if high_coordinates.size:
        high_center = high_coordinates.mean(axis=0)
        span = np.maximum(coordinates.max(axis=0) - coordinates.min(axis=0), 1)
        relative_center = (high_center - center) / span
    else:
        relative_center = np.zeros(3, dtype=float)
    left_fraction = float(high[left].sum() / max(left.sum(), 1))
    right_fraction = float(high[right].sum() / max(right.sum(), 1))
    features.update(
        {
            "core_highuptake_fraction": float(high.sum() / max(target.sum(), 1)),
            "core_highuptake_components": float(components),
            "core_highuptake_largest_component_fraction": largest_fraction,
            "core_highuptake_bilateral_balance": float(
                min(left_fraction, right_fraction) / max(max(left_fraction, right_fraction), 1e-6)
            ),
            "core_highuptake_centroid_z": float(relative_center[0]),
            "core_highuptake_centroid_y": float(relative_center[1]),
            "core_highuptake_centroid_x": float(relative_center[2]),
        }
    )
    features.update(_texture_glcm_3d(normalized, target, config.texture_levels))
    features.update(_projection_features(normalized, target))
    features.update(_shape_features(high, spacing_xyz))
    return features


def build_derived_feature_cache(
    crops_dir: Path,
    cohort: pd.DataFrame,
    output_path: Path,
    *,
    config: DataConfig,
    upstream_hash: str,
) -> pd.DataFrame:
    manifest_path = output_path.with_suffix(".config.json")
    cache_contract = {
        "feature_schema_version": config.feature_schema_version,
        "data_config": config.__dict__,
        "upstream_hash": upstream_hash,
        "uid_hash": stable_hash(sorted(cohort["uid"].astype(str).tolist())),
    }
    cache_hash = stable_hash(cache_contract)
    if manifest_path.exists():
        import json

        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("cache_hash") != cache_hash:
            raise RuntimeError(
                "El cache de radiomics CNN pertenece a otra configuracion; usa otro run_id."
            )
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    processed = set(existing.get("uid", pd.Series(dtype=str)).astype(str))
    records = existing.to_dict("records")
    pending = cohort.loc[~cohort["uid"].astype(str).isin(processed)].copy()
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        uid = str(row.uid)
        crop_path = crops_dir / f"{uid}.npz"
        if not crop_path.exists():
            raise FileNotFoundError(crop_path)
        with np.load(crop_path, allow_pickle=False) as payload:
            features = extract_background_independent_features(
                payload["volume_intensity01"],
                payload["target_mask"],
                config,
                tuple(float(value) for value in payload["spacing_xyz"]),
            )
        records.append({"uid": uid, **features})
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame.from_records(records).sort_values("uid"), output_path)
            print(f"[CNN features] radiomics core: {len(records)}/{len(cohort)}")
    result = pd.DataFrame.from_records(records).sort_values("uid").reset_index(drop=True)
    atomic_csv(result, output_path)
    atomic_json(
        {
            **cache_contract,
            "cache_hash": cache_hash,
            "n_complete": len(result),
            "status": "complete",
        },
        manifest_path,
    )
    return result


def available_sbr_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SBR_FEATURE_COLUMNS if column in frame.columns]


def core_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("core_"))
