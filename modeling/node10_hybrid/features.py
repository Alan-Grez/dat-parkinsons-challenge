from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from modeling.cnn.io import atomic_csv, atomic_json, atomic_npz
from modeling.dat_spect_v2.config import stable_hash
from modeling.dat_spect_v2.features import extract_regional930
from modeling.dat_spect_v2.preprocessing import (
    _physical_centered_crop,
    physical_striatal_roi,
    self_normalize_striatum,
)

from .config import DataConfig

REGION_ORDER = (
    "right_caudate",
    "right_putamen_anterior",
    "right_putamen_posterior",
    "left_caudate",
    "left_putamen_anterior",
    "left_putamen_posterior",
)


def proxy_region_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Six fixed anatomical proxies in registered z-y-x LPS space.

    These are reproducible classification ROIs, not clinical segmentations. Low x
    is patient-right. The posterior half is split once more to expose the
    anterior/posterior putaminal gradient requested by the experiment.
    """

    target = np.asarray(mask, dtype=bool)
    _, y, x = target.shape
    yy, xx = np.indices((y, x))
    mid_x = (x - 1.0) / 2.0
    mid_y = (y - 1.0) / 2.0
    posterior_mid = mid_y + y / 4.0
    right = xx[None] < mid_x
    left = ~right
    caudate = yy[None] < mid_y
    putamen_anterior = (yy[None] >= mid_y) & (yy[None] < posterior_mid)
    putamen_posterior = yy[None] >= posterior_mid
    return {
        "right_caudate": target & right & caudate,
        "right_putamen_anterior": target & right & putamen_anterior,
        "right_putamen_posterior": target & right & putamen_posterior,
        "left_caudate": target & left & caudate,
        "left_putamen_anterior": target & left & putamen_anterior,
        "left_putamen_posterior": target & left & putamen_posterior,
    }


def _safe_stats(volume: np.ndarray, mask: np.ndarray) -> list[float]:
    values = np.asarray(volume, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if not values.size:
        return [0.0] * 8
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return [
        float(values.mean()),
        float(values.std()),
        float(p10),
        float(p50),
        float(p90),
        float(values.max()),
        float(np.mean(values**2)),
        float(np.median(np.abs(values - p50))),
    ]


def auxiliary_targets(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    regions = proxy_region_masks(mask)
    uptake = np.asarray(
        [float(raw[regions[name]].mean()) if regions[name].any() else 0.0 for name in REGION_ORDER],
        dtype=np.float32,
    )
    right_putamen = float(np.mean(uptake[1:3]))
    left_putamen = float(np.mean(uptake[4:6]))
    affected_left = float(left_putamen < right_putamen)
    asymmetry = abs(left_putamen - right_putamen) / max(
        abs(left_putamen) + abs(right_putamen), 1e-6
    )
    selected = raw[np.asarray(mask, dtype=bool)]
    threshold = float(np.percentile(selected, 85.0)) if selected.size else float("inf")
    _, components = ndimage.label((raw >= threshold) & np.asarray(mask, dtype=bool))
    fragmentation = float(min(components, 12) / 12.0)
    return np.concatenate(
        [uptake, np.asarray([affected_left, asymmetry, fragmentation], dtype=np.float32)]
    )


def topology_features(
    selfnorm: np.ndarray,
    mask: np.ndarray,
    percentiles: tuple[float, ...],
) -> dict[str, float]:
    regions = {"target": np.asarray(mask, dtype=bool), **proxy_region_masks(mask)}
    selected = selfnorm[regions["target"]]
    thresholds = (
        np.percentile(selected, percentiles) if selected.size else np.zeros(len(percentiles))
    )
    result: dict[str, float] = {}
    for region_name, region in regions.items():
        support = max(int(region.sum()), 1)
        for percentile, threshold in zip(percentiles, thresholds, strict=True):
            active = region & (selfnorm >= threshold)
            labels, components = ndimage.label(active)
            sizes = np.bincount(labels.ravel())[1:] if components else np.asarray([], dtype=int)
            largest = int(sizes.max()) if sizes.size else 0
            eroded = ndimage.binary_erosion(active)
            surface = int((active & ~eroded).sum())
            coordinates = np.argwhere(active)
            extent = (
                coordinates.max(axis=0) - coordinates.min(axis=0) + 1
                if len(coordinates)
                else np.zeros(3)
            )
            prefix = f"topo_{region_name}_p{int(percentile):02d}"
            result[f"{prefix}_fraction"] = float(active.sum() / support)
            result[f"{prefix}_components"] = float(components)
            result[f"{prefix}_largest_fraction"] = float(largest / max(int(active.sum()), 1))
            result[f"{prefix}_surface_fraction"] = float(surface / support)
            result[f"{prefix}_extent_z"] = float(extent[0])
            result[f"{prefix}_extent_y"] = float(extent[1])
            result[f"{prefix}_extent_x"] = float(extent[2])
    return result


def graph_features(
    raw: np.ndarray,
    selfnorm: np.ndarray,
    mask: np.ndarray,
    *,
    z_bands: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build subject-specific node features on a fixed anatomical graph."""

    if z_bands < 2:
        raise ValueError("El grafo requiere al menos dos bandas axiales.")

    regions = proxy_region_masks(mask)
    zz = np.indices(mask.shape)[0]
    z_edges = np.linspace(0, mask.shape[0], z_bands + 1)
    nodes: list[list[float]] = []
    for region_name in REGION_ORDER:
        for band in range(z_bands):
            band_mask = regions[region_name] & (zz >= z_edges[band]) & (zz < z_edges[band + 1])
            raw_stats = _safe_stats(raw, band_mask)
            norm_stats = _safe_stats(selfnorm, band_mask)
            coordinates = np.argwhere(band_mask & (raw > 0))
            if len(coordinates):
                centroid = coordinates.mean(axis=0) / np.maximum(np.asarray(mask.shape) - 1, 1)
            else:
                centroid = np.zeros(3)
            occupancy = float(np.mean(raw[band_mask] > 0)) if band_mask.any() else 0.0
            high = raw[band_mask]
            high_fraction = (
                float(np.mean(high >= np.percentile(raw[mask], 85.0)))
                if high.size and mask.any()
                else 0.0
            )
            nodes.append(
                raw_stats
                + norm_stats
                + [
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                    occupancy,
                    high_fraction,
                ]
            )
    node_values = np.asarray(nodes, dtype=np.float32)
    count = len(node_values)
    adjacency = np.eye(count, dtype=np.float32)

    def node(region_index: int, band: int) -> int:
        return region_index * z_bands + band

    # Same region across adjacent axial bands.
    for region in range(6):
        for first_band, second_band in pairwise(range(z_bands)):
            a, b = node(region, first_band), node(region, second_band)
            adjacency[a, b] = adjacency[b, a] = 1.0
    # Caudate -> anterior putamen -> posterior putamen within each side/band.
    for side_start in (0, 3):
        for band in range(z_bands):
            chain = [node(side_start + offset, band) for offset in range(3)]
            for a, b in pairwise(chain):
                adjacency[a, b] = adjacency[b, a] = 1.0
    # Homologous left/right regions.
    for offset in range(3):
        for band in range(z_bands):
            a, b = node(offset, band), node(3 + offset, band)
            adjacency[a, b] = adjacency[b, a] = 1.0
    degree = adjacency.sum(axis=1)
    inverse = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-6)))
    normalized = inverse @ adjacency @ inverse
    return node_values, normalized.astype(np.float32)


def build_hybrid_cache(
    crops_dir: Path,
    cohort: pd.DataFrame,
    cache_root: Path,
    *,
    config: DataConfig,
    upstream_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract = {
        "preprocessing_version": config.preprocessing_version,
        "upstream_hash": upstream_hash,
        "uids": stable_hash(sorted(cohort["uid"].astype(str).tolist())),
        "output_spacing_mm": config.output_spacing_mm,
        "output_shape_zyx": config.output_shape_zyx,
        "topology_percentiles": config.topology_percentiles,
        "graph_z_bands": config.graph_z_bands,
        "raw_view": (
            "node04 volume_intensity01 multiplied by fold-independent foreground_p99_registered; "
            "registered counts are clipped at the original p99 but retain between-study magnitude"
        ),
        "pattern_view": "sqrt(d)*x/||x||2 over the same positive striatal ROI",
    }
    cache_hash = stable_hash(contract)
    cache_dir = cache_root / cache_hash[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"
    topology_path = cache_dir / "topology_features.csv"
    regional_path = cache_dir / "regional930_features.csv"
    graph_path = cache_dir / "graph_features.csv"
    metadata_path = cache_dir / "config.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("cache_hash") != cache_hash:
            raise RuntimeError("El cache del nodo 10 pertenece a otro contrato.")
    existing_manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    existing_topology = pd.read_csv(topology_path) if topology_path.exists() else pd.DataFrame()
    existing_regional = pd.read_csv(regional_path) if regional_path.exists() else pd.DataFrame()
    existing_graph = pd.read_csv(graph_path) if graph_path.exists() else pd.DataFrame()
    records = {str(row["uid"]): row for row in existing_manifest.to_dict("records")}
    topo_records = {str(row["uid"]): row for row in existing_topology.to_dict("records")}
    regional_records = {str(row["uid"]): row for row in existing_regional.to_dict("records")}
    graph_records = {str(row["uid"]): row for row in existing_graph.to_dict("records")}
    pending = cohort.loc[
        ~cohort["uid"]
        .astype(str)
        .isin(set(records) & set(topo_records) & set(regional_records) & set(graph_records))
    ]
    for counter, row in enumerate(pending.itertuples(index=False), start=1):
        uid = str(row.uid)
        intensity_p99 = float(row.foreground_p99_registered)
        if not np.isfinite(intensity_p99) or intensity_p99 <= 0:
            raise ValueError(f"{uid}: foreground_p99_registered invalido ({intensity_p99}).")
        source = crops_dir / f"{uid}.npz"
        with np.load(source, allow_pickle=False) as payload:
            original = np.asarray(payload["volume_intensity01"], dtype=np.float32)
            source_mask = np.asarray(payload["target_mask"], dtype=np.uint8) > 0
            spacing_xyz = np.asarray(payload["spacing_xyz"], dtype=np.float32)
        coordinates = np.argwhere(source_mask)
        if not len(coordinates):
            raise ValueError(f"{uid}: target_mask vacia.")
        intensity01 = _physical_centered_crop(
            original,
            coordinates.mean(axis=0),
            spacing_xyz,
            config.output_shape_zyx,
            config.output_spacing_mm,
            order=1,
        )
        target = physical_striatal_roi(
            intensity01,
            spacing_mm=config.output_spacing_mm,
            radii_zyx_mm=config.striatal_roi_radii_zyx_mm,
            hot_percentile=config.striatal_roi_hot_percentile,
            recenter_max_mm=config.striatal_roi_recenter_max_mm,
        )
        intensity01 = np.nan_to_num(intensity01, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0, 1.0)
        raw = intensity01 * intensity_p99 * target
        selfnorm = self_normalize_striatum(raw, target)
        aux = auxiliary_targets(raw, target)
        nodes, adjacency = graph_features(raw, selfnorm, target, z_bands=config.graph_z_bands)
        destination = cache_dir / f"{uid}.npz"
        atomic_npz(
            destination,
            volume_intensity=raw.astype(config.cache_dtype),
            volume_selfnorm=selfnorm.astype(config.cache_dtype),
            target_mask=target.astype(np.uint8),
            auxiliary_targets=aux.astype(np.float32),
            # Volumes may use float16 to keep the cache compact, but raw-count
            # graph moments (especially E[x^2]) routinely exceed 65,504.
            graph_nodes=nodes.astype(np.float32),
            graph_adjacency=adjacency.astype(config.cache_dtype),
        )
        raw_selected = raw[target]
        records[uid] = {
            "uid": uid,
            "node10_cache_path": str(destination.resolve()),
            "intensity_l2_norm": float(np.linalg.norm(raw_selected.astype(np.float64))),
            "intensity_mean": float(raw_selected.mean()) if raw_selected.size else 0.0,
            "intensity_p90": float(np.percentile(raw_selected, 90)) if raw_selected.size else 0.0,
            "intensity_positive_voxels": int(np.sum(raw_selected > 0)),
            "intensity_reconstruction_p99": intensity_p99,
        }
        topo_records[uid] = {
            "uid": uid,
            **topology_features(selfnorm, target, config.topology_percentiles),
        }
        regional_records[uid] = {
            "uid": uid,
            **{
                f"regional930_{name}": value
                for name, value in extract_regional930(selfnorm, target).items()
            },
        }
        graph_records[uid] = {
            "uid": uid,
            **{
                f"graph_n{node_index:02d}_f{feature_index:02d}": float(
                    nodes[node_index, feature_index]
                )
                for node_index in range(nodes.shape[0])
                for feature_index in range(nodes.shape[1])
            },
        }
        if counter % config.cache_checkpoint_every == 0:
            atomic_csv(pd.DataFrame(records.values()).sort_values("uid"), manifest_path)
            atomic_csv(pd.DataFrame(topo_records.values()).sort_values("uid"), topology_path)
            atomic_csv(pd.DataFrame(regional_records.values()).sort_values("uid"), regional_path)
            atomic_csv(pd.DataFrame(graph_records.values()).sort_values("uid"), graph_path)
            print(f"[nodo10 cache] {len(records)}/{len(cohort)}")
    manifest = pd.DataFrame(records.values()).sort_values("uid").reset_index(drop=True)
    topology = pd.DataFrame(topo_records.values()).sort_values("uid").reset_index(drop=True)
    regional = pd.DataFrame(regional_records.values()).sort_values("uid").reset_index(drop=True)
    graph = pd.DataFrame(graph_records.values()).sort_values("uid").reset_index(drop=True)
    if len(manifest) != len(cohort) or manifest["uid"].nunique() != len(cohort):
        raise RuntimeError("El cache del nodo 10 no cubre exactamente la cohorte.")
    atomic_csv(manifest, manifest_path)
    atomic_csv(topology, topology_path)
    atomic_csv(regional, regional_path)
    atomic_csv(graph, graph_path)
    atomic_json({**contract, "cache_hash": cache_hash, "n_complete": len(manifest)}, metadata_path)
    return manifest, topology, regional, graph
