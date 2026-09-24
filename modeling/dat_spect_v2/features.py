from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from modeling.cnn.io import atomic_csv, atomic_json

from .config import DataConfig, LateralStrategy, stable_hash
from .preprocessing import split_striatal_regions

INTENSITY_STATS = (
    "mean",
    "std",
    "min",
    "max",
    "p01",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "mad",
    "energy",
)
THRESHOLD_PERCENTILES = (70.0, 80.0, 90.0, 95.0)
TEXTURE_LEVELS = (8, 12, 16, 24, 32)
TEXTURE_METRICS = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "entropy",
    "correlation",
)


def _region_masks(target_mask: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target_mask, dtype=bool)
    quadrants = split_striatal_regions(target)
    right = quadrants["right_caudate"] | quadrants["right_putamen"]
    left = quadrants["left_caudate"] | quadrants["left_putamen"]
    anterior = quadrants["right_caudate"] | quadrants["left_caudate"]
    posterior = quadrants["right_putamen"] | quadrants["left_putamen"]
    distance = ndimage.distance_transform_edt(target)
    positive = distance[target]
    cutoff = float(np.percentile(positive, 60)) if positive.size else float("inf")
    core = target & (distance >= cutoff)
    return {
        "target": target,
        "right": right,
        "left": left,
        "anterior": anterior,
        "posterior": posterior,
        **quadrants,
        "target_core": core,
    }


def _channels(volume: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    image = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0)
    values = image[target]
    median = float(np.median(values)) if values.size else 0.0
    q25, q75 = np.percentile(values, [25, 75]) if values.size else (0.0, 1.0)
    scale = max(float(q75 - q25), 1e-6)
    robust = np.zeros_like(image)
    robust[target] = (image[target] - median) / scale
    rank = np.zeros_like(image)
    if values.size:
        rank[target] = stats.rankdata(values, method="average") / max(len(values), 1)
    return {"selfnorm": image, "robust": robust, "rank": rank}


def _intensity_summary(values: np.ndarray) -> list[float]:
    selected = np.asarray(values, dtype=np.float64)
    if not selected.size:
        return [0.0] * len(INTENSITY_STATS)
    percentiles = np.percentile(selected, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    median = float(percentiles[4])
    return [
        float(selected.mean()),
        float(selected.std(ddof=0)),
        float(selected.min()),
        float(selected.max()),
        *[float(value) for value in percentiles],
        float(np.median(np.abs(selected - median))),
        float(np.mean(np.square(selected))),
    ]


def _morphology(active: np.ndarray, region: np.ndarray) -> list[float]:
    selected = np.asarray(active, dtype=bool) & np.asarray(region, dtype=bool)
    support = max(int(np.asarray(region, dtype=bool).sum()), 1)
    count = int(selected.sum())
    labels, components = ndimage.label(selected)
    sizes = np.bincount(labels.ravel())[1:] if components else np.asarray([], dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    coordinates = np.argwhere(selected)
    if len(coordinates):
        extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        region_center = np.argwhere(region).mean(axis=0)
        centroid_distance = float(np.linalg.norm(coordinates.mean(axis=0) - region_center))
    else:
        extent = np.zeros(3, dtype=float)
        centroid_distance = 0.0
    return [
        float(count / support),
        float(count),
        float(components),
        float(largest / max(count, 1)),
        float(extent[0]),
        float(extent[1]),
        float(extent[2]),
        centroid_distance,
    ]


def _glcm_metrics(volume: np.ndarray, mask: np.ndarray, levels: int) -> list[float]:
    values = np.asarray(volume, dtype=np.float32)
    selected = values[mask]
    if selected.size < 8 or float(selected.max()) <= float(selected.min()):
        return [0.0] * len(TEXTURE_METRICS)
    low, high = np.percentile(selected, [1, 99])
    scaled = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    quantized = np.minimum((scaled * levels).astype(np.int32), levels - 1)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, -1)
        second_slice[axis] = slice(1, None)
        first_mask = mask[tuple(first_slice)]
        second_mask = mask[tuple(second_slice)]
        valid = first_mask & second_mask
        first = quantized[tuple(first_slice)][valid]
        second = quantized[tuple(second_slice)][valid]
        if first.size:
            np.add.at(matrix, (first, second), 1)
            np.add.at(matrix, (second, first), 1)
    total = float(matrix.sum())
    if total <= 0:
        return [0.0] * len(TEXTURE_METRICS)
    probability = matrix / total
    i, j = np.indices(matrix.shape)
    contrast = float(np.sum(probability * (i - j) ** 2))
    dissimilarity = float(np.sum(probability * np.abs(i - j)))
    homogeneity = float(np.sum(probability / (1.0 + (i - j) ** 2)))
    energy = float(np.sqrt(np.sum(np.square(probability))))
    positive = probability > 0
    entropy = float(-np.sum(probability[positive] * np.log2(probability[positive])))
    mean_i = float(np.sum(probability * i))
    mean_j = float(np.sum(probability * j))
    std_i = math_sqrt(float(np.sum(probability * (i - mean_i) ** 2)))
    std_j = math_sqrt(float(np.sum(probability * (j - mean_j) ** 2)))
    correlation = float(
        np.sum(probability * (i - mean_i) * (j - mean_j)) / max(std_i * std_j, 1e-8)
    )
    return [contrast, dissimilarity, homogeneity, energy, entropy, correlation]


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))


def _relation_features(volume: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, float]:
    def mean(name: str) -> float:
        selected = volume[regions[name]]
        return float(selected.mean()) if selected.size else 0.0

    values = {name: mean(name) for name in regions}
    pairs = {
        "caudate_lr": (values["left_caudate"], values["right_caudate"]),
        "putamen_lr": (values["left_putamen"], values["right_putamen"]),
        "right_cp": (values["right_caudate"], values["right_putamen"]),
        "left_cp": (values["left_caudate"], values["left_putamen"]),
        "hemisphere_lr": (values["left"], values["right"]),
        "anterior_posterior": (values["anterior"], values["posterior"]),
        "minmax_putamen": (
            min(values["left_putamen"], values["right_putamen"]),
            max(values["left_putamen"], values["right_putamen"]),
        ),
        "minmax_caudate": (
            min(values["left_caudate"], values["right_caudate"]),
            max(values["left_caudate"], values["right_caudate"]),
        ),
        "weak_caudate_putamen": (
            min(values["left_caudate"], values["right_caudate"]),
            min(values["left_putamen"], values["right_putamen"]),
        ),
        "core_target": (values["target_core"], values["target"]),
    }
    result: dict[str, float] = {}
    for name, (first, second) in pairs.items():
        denominator = max(abs(first) + abs(second), 1e-8)
        result[f"relation_{name}_difference"] = float(first - second)
        result[f"relation_{name}_absolute_difference"] = float(abs(first - second))
        result[f"relation_{name}_ratio"] = float(first / max(abs(second), 1e-8))
        result[f"relation_{name}_asymmetry"] = float((first - second) / denominator)
    return result


def extract_regional930(volume: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    regions = _region_masks(target_mask)
    channels = _channels(volume, regions["target"])
    features: dict[str, float] = {}
    # 10 regions x 3 channels x 15 summaries = 450.
    for region_name, region in regions.items():
        for channel_name, channel in channels.items():
            summary = _intensity_summary(channel[region])
            for stat_name, value in zip(INTENSITY_STATS, summary, strict=True):
                features[f"intensity_{region_name}_{channel_name}_{stat_name}"] = value
    # 10 regions x 4 thresholds x 8 morphology summaries = 320.
    target_values = channels["selfnorm"][regions["target"]]
    thresholds = (
        np.percentile(target_values, THRESHOLD_PERCENTILES)
        if target_values.size
        else np.zeros(len(THRESHOLD_PERCENTILES))
    )
    morph_names = (
        "fraction",
        "voxels",
        "components",
        "largest_fraction",
        "extent_z",
        "extent_y",
        "extent_x",
        "centroid_distance",
    )
    for region_name, region in regions.items():
        for percentile, threshold in zip(THRESHOLD_PERCENTILES, thresholds, strict=True):
            summary = _morphology(channels["selfnorm"] >= threshold, region)
            for metric, value in zip(morph_names, summary, strict=True):
                features[f"morph_{region_name}_p{int(percentile):02d}_{metric}"] = value
    # Four clinical proxy regions x five quantizations x six GLCM metrics = 120.
    for region_name in (
        "right_caudate",
        "right_putamen",
        "left_caudate",
        "left_putamen",
    ):
        for levels in TEXTURE_LEVELS:
            summary = _glcm_metrics(channels["selfnorm"], regions[region_name], levels)
            for metric, value in zip(TEXTURE_METRICS, summary, strict=True):
                features[f"texture_{region_name}_l{levels:02d}_{metric}"] = value
    # Ten clinically interpretable pairings x four contrasts = 40.
    features.update(_relation_features(channels["selfnorm"], regions))
    if len(features) != 930:
        raise AssertionError(
            f"Se esperaban 930 features regionales, se obtuvieron {len(features)}."
        )
    return features


def build_regional_feature_cache(
    image_manifest: pd.DataFrame,
    destination: Path,
    *,
    config: DataConfig,
    lateral_strategy: LateralStrategy,
    source_hash: str,
) -> pd.DataFrame:
    contract = {
        "source_hash": source_hash,
        "lateral_strategy": lateral_strategy,
        "schema": config.regional_feature_schema,
        "budget": config.regional_feature_budget,
    }
    cache_hash = stable_hash(contract)
    metadata_path = destination.with_suffix(".json")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("cache_hash") != cache_hash:
            raise RuntimeError("El cache regional pertenece a otra configuracion.")
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    records = {str(record["uid"]): record for record in existing.to_dict("records")}
    pending = image_manifest.loc[~image_manifest["uid"].astype(str).isin(records)]
    suffix = "canonical" if lateral_strategy == "canonical" else "native"
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        with np.load(Path(row.node7_cache_path), allow_pickle=False) as payload:
            volume = np.asarray(payload[f"volume_{suffix}"], dtype=np.float32)
            mask = np.asarray(payload[f"target_mask_{suffix}"], dtype=np.uint8) > 0
        records[str(row.uid)] = {"uid": str(row.uid), **extract_regional930(volume, mask)}
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame(records.values()).sort_values("uid"), destination)
            print(f"[nodo07 features {lateral_strategy}] {len(records)}/{len(image_manifest)}")
    frame = pd.DataFrame(records.values()).sort_values("uid").reset_index(drop=True)
    feature_columns = [
        column
        for column in frame
        if column.startswith(("intensity_", "morph_", "texture_", "relation_"))
    ]
    if len(feature_columns) != config.regional_feature_budget:
        raise AssertionError("El manifiesto regional no conserva exactamente 930 variables.")
    atomic_csv(frame, destination)
    atomic_json({**contract, "cache_hash": cache_hash, "n_complete": len(frame)}, metadata_path)
    return frame


@dataclass
class FoldRegionalTransformer:
    top_k: int = 128
    pca_variance: float | None = None
    correlation_threshold: float = 0.98
    random_seed: int = 20260828
    median_: np.ndarray | None = None
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    retained_indices_: np.ndarray | None = None
    selected_indices_: np.ndarray | None = None
    pca_mean_: np.ndarray | None = None
    pca_components_: np.ndarray | None = None
    selected_names_: list[str] | None = None

    def fit(
        self,
        values: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> FoldRegionalTransformer:
        array = np.asarray(values, dtype=float)
        self.median_ = np.nanmedian(array, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(array), array, self.median_)
        self.center_ = np.median(filled, axis=0)
        q25, q75 = np.percentile(filled, [25, 75], axis=0)
        self.scale_ = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
        scaled = (filled - self.center_) / self.scale_
        varying = np.where(np.std(scaled, axis=0) > 1e-8)[0]
        retained: list[int] = []
        for index in varying:
            if not retained:
                retained.append(int(index))
                continue
            correlations = np.corrcoef(scaled[:, index], scaled[:, retained], rowvar=False)[0, 1:]
            if not np.any(np.abs(np.nan_to_num(correlations)) >= self.correlation_threshold):
                retained.append(int(index))
        self.retained_indices_ = np.asarray(retained, dtype=int)
        reduced = scaled[:, self.retained_indices_]
        scores = mutual_info_classif(
            reduced,
            np.asarray(y, dtype=int),
            random_state=self.random_seed,
        )
        order = np.argsort(-np.nan_to_num(scores))
        count = min(self.top_k, len(order))
        self.selected_indices_ = order[:count]
        selected = reduced[:, self.selected_indices_]
        original_indices = self.retained_indices_[self.selected_indices_]
        self.selected_names_ = [feature_names[index] for index in original_indices]
        if self.pca_variance is not None and selected.shape[1] > 1:
            pca = PCA(n_components=self.pca_variance, svd_solver="full")
            pca.fit(selected)
            self.pca_mean_ = pca.mean_.astype(np.float64)
            self.pca_components_ = pca.components_.astype(np.float64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        required = (
            self.median_,
            self.center_,
            self.scale_,
            self.retained_indices_,
            self.selected_indices_,
        )
        if any(value is None for value in required):
            raise RuntimeError("El transformador regional aun no fue ajustado.")
        array = np.asarray(values, dtype=float)
        filled = np.where(np.isfinite(array), array, self.median_)
        scaled = (filled - self.center_) / self.scale_
        selected = scaled[:, self.retained_indices_][:, self.selected_indices_]
        if self.pca_components_ is not None and self.pca_mean_ is not None:
            selected = (selected - self.pca_mean_) @ self.pca_components_.T
        return selected.astype(np.float32)

    @property
    def output_dim(self) -> int:
        if self.pca_components_ is not None:
            return int(self.pca_components_.shape[0])
        if self.selected_indices_ is not None:
            return len(self.selected_indices_)
        return 0

    def to_state(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "pca_variance": self.pca_variance,
            "correlation_threshold": self.correlation_threshold,
            "random_seed": self.random_seed,
            "median": self.median_,
            "center": self.center_,
            "scale": self.scale_,
            "retained_indices": self.retained_indices_,
            "selected_indices": self.selected_indices_,
            "pca_mean": self.pca_mean_,
            "pca_components": self.pca_components_,
            "selected_names": self.selected_names_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FoldRegionalTransformer:
        transformer = cls(
            top_k=int(state["top_k"]),
            pca_variance=state.get("pca_variance"),
            correlation_threshold=float(state["correlation_threshold"]),
            random_seed=int(state["random_seed"]),
        )
        for attribute, key, dtype in (
            ("median_", "median", float),
            ("center_", "center", float),
            ("scale_", "scale", float),
            ("retained_indices_", "retained_indices", int),
            ("selected_indices_", "selected_indices", int),
            ("pca_mean_", "pca_mean", float),
            ("pca_components_", "pca_components", float),
        ):
            value = state.get(key)
            setattr(transformer, attribute, None if value is None else np.asarray(value, dtype=dtype))
        selected_names = state.get("selected_names")
        transformer.selected_names_ = (
            None if selected_names is None else list(map(str, selected_names))
        )
        return transformer
