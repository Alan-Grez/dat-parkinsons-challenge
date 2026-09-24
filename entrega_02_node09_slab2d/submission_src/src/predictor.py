from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from modeling.dat_spect_v2.config import DataConfig, ModelConfig
from modeling.dat_spect_v2.features import extract_regional930
from modeling.dat_spect_v2.models import MultiTaskDaTClassifier
from modeling.dat_spect_v2.preprocessing import (
    _affine_refine,
    _physical_centered_crop,
    axial_slab,
    physical_striatal_roi,
    self_normalize_striatum,
)

from .raw_registration import ImageConfig, SubmissionPreprocessor

MIN_BACKGROUND_VOXELS = 64
MIN_BACKGROUND_SUPPORT_FRACTION = 0.20
BACKGROUND_SCALE_FLOOR_FRACTION = 0.05
ACTIVE_BACKGROUND_SD = 2.0


def _central_ellipsoid(shape: tuple[int, int, int]) -> np.ndarray:
    coordinates = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    radii = np.maximum(np.asarray(shape, dtype=np.float32) * 0.42, 1)
    distance = sum(
        ((coordinates[axis] - center[axis]) / radii[axis]) ** 2
        for axis in range(3)
    )
    return distance <= 1


def _robust_background(
    array: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    positive = np.isfinite(array) & (array > 0)
    foreground = masks.get("brain", _central_ellipsoid(array.shape)) & positive
    if int(foreground.sum()) < MIN_BACKGROUND_VOXELS:
        foreground = _central_ellipsoid(array.shape) & positive
    if int(foreground.sum()) < MIN_BACKGROUND_VOXELS:
        foreground = positive
    foreground_values = array[foreground]
    if foreground_values.size < MIN_BACKGROUND_VOXELS:
        raise ValueError("Insufficient positive foreground after registration.")
    foreground_p90 = float(np.percentile(foreground_values, 90))
    floor = max(1e-8, BACKGROUND_SCALE_FLOOR_FRACTION * foreground_p90)
    nominal = masks["background"]
    candidate = nominal & positive
    support_fraction = float(candidate.sum() / max(int(nominal.sum()), 1))
    enough = bool(
        int(candidate.sum()) >= MIN_BACKGROUND_VOXELS
        and support_fraction >= MIN_BACKGROUND_SUPPORT_FRACTION
    )
    source = "consensus_brain_background"
    if not enough:
        exclusion = ndimage.binary_dilation(masks["target"], iterations=5)
        candidate = foreground & (~exclusion)
        if candidate.any():
            candidate &= array <= np.percentile(array[candidate], 70)
        source = "scan_brain_fallback"
    values = array[candidate & positive]
    if values.size < MIN_BACKGROUND_VOXELS:
        values = foreground_values[
            foreground_values <= np.percentile(foreground_values, 50)
        ]
        source = "foreground_lower_half_fallback"
    if values.size < 10:
        raise ValueError("Insufficient robust background after registration.")
    low, high = np.percentile(values, [5, 95])
    trimmed = values[(values >= low) & (values <= high)]
    if trimmed.size < 10:
        trimmed = values
    mean = float(trimmed.mean())
    std = float(trimmed.std(ddof=1)) if trimmed.size > 1 else 0.0
    floor_applied = (not np.isfinite(mean)) or mean < floor
    return {
        "scale": max(mean if np.isfinite(mean) else 0.0, floor),
        "std": std,
        "valid": bool(
            enough
            and source == "consensus_brain_background"
            and not floor_applied
            and np.isfinite(std)
        ),
    }


def _mean(array: np.ndarray, mask: np.ndarray) -> float:
    values = array[mask & np.isfinite(array)]
    return float(values.mean()) if values.size else float("nan")


def _sbr(target: float, background: float) -> float:
    if not np.isfinite(background) or background <= 1e-8:
        return float("nan")
    return float((target - background) / background)


def extract_sbr_features(
    registered: np.ndarray, masks: dict[str, np.ndarray]
) -> tuple[dict[str, float], bool]:
    context = _robust_background(registered, masks)
    background = float(context["scale"])
    target = _mean(registered, masks["target"])
    right = _mean(registered, masks["right"])
    left = _mean(registered, masks["left"])
    ratio_values = registered[masks["target"] & np.isfinite(registered)] / background
    values = {
        "semiquant_sbr": _sbr(target, background),
        "semiquant_right_sbr": _sbr(right, background),
        "semiquant_left_sbr": _sbr(left, background),
        "semiquant_min_side_sbr": min(
            _sbr(right, background), _sbr(left, background)
        ),
        "semiquant_log_target_background": float(
            np.log1p(max(target, 0.0) / background)
        ),
        "firstorder_ratio_mean": float(ratio_values.mean()),
        "firstorder_ratio_std": float(ratio_values.std(ddof=1)),
        "firstorder_ratio_p10": float(np.percentile(ratio_values, 10)),
        "firstorder_ratio_p50": float(np.percentile(ratio_values, 50)),
        "firstorder_ratio_p90": float(np.percentile(ratio_values, 90)),
        "active_threshold_sbr": float(ACTIVE_BACKGROUND_SD * context["std"] / background),
    }
    return values, bool(context["valid"])


def _apply_regional_transform(
    features: dict[str, float], state: dict[str, Any]
) -> np.ndarray:
    names = list(map(str, state["selected_names"]))
    array = np.asarray([[features[name] for name in names]], dtype=np.float64)
    median = np.asarray(state["median"], dtype=np.float64)
    center = np.asarray(state["center"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    filled = np.where(np.isfinite(array), array, median)
    selected = (filled - center) / scale
    components = state.get("pca_components")
    mean = state.get("pca_mean")
    if components is not None and mean is not None:
        selected = (selected - np.asarray(mean)) @ np.asarray(components).T
    return selected.astype(np.float32)


def _apply_sbr_transform(
    features: dict[str, float], columns: list[str], state: dict[str, Any], valid: bool
) -> np.ndarray:
    array = np.asarray([[features[column] for column in columns]], dtype=np.float64)
    median = np.asarray(state["median"], dtype=np.float64)
    center = np.asarray(state["center"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    filled = np.where(np.isfinite(array), array, median)
    transformed = ((filled - center) / scale).astype(np.float32)
    if not valid:
        transformed[:] = 0.0
    return transformed


class Node09Predictor:
    def __init__(self, asset_root: Path) -> None:
        self.asset_root = Path(asset_root)
        self.manifest = json.loads(
            (self.asset_root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["n_fold_models"] != 5:
            raise RuntimeError("The node09 submission requires exactly five fold models.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_config = ImageConfig(
            image_shape_3d=(40, 72, 72),
            view_size_2d=72,
            slices_per_view=5,
            clip_percentiles=(0.5, 99.5),
            scale_percentiles=(25.0, 75.0),
            normalized_clip=(-4.0, 6.0),
        )
        registration_path = self.asset_root / self.manifest["registration_template_file"]
        self.raw = SubmissionPreprocessor(registration_path, image_config, self.device)
        with np.load(registration_path, allow_pickle=False) as payload:
            self.spacing_xyz = np.asarray(payload["spacing_xyz"], dtype=np.float64)
            self.crop_start = np.asarray(payload["crop_start_zyx"], dtype=int)
            self.crop_stop = np.asarray(payload["crop_stop_zyx"], dtype=int)
            self.full_masks = {
                name: np.asarray(payload[name], dtype=np.uint8) > 0
                for name in (
                    "target",
                    "background",
                    "brain",
                    "right",
                    "left",
                    "anterior",
                    "posterior",
                )
            }
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(self.crop_start, self.crop_stop, strict=True)
        )
        self.node4_target_crop = self.full_masks["target"][slices]
        self.folds: list[dict[str, Any]] = []
        for fold_entry in self.manifest["folds"]:
            state = torch.load(
                self.asset_root / fold_entry["checkpoint_file"],
                map_location=self.device,
                weights_only=False,
            )
            model_config = ModelConfig(**state["model_config"])
            model = MultiTaskDaTClassifier(model_config).to(self.device)
            model.load_state_dict(state["model_state"], strict=True)
            model.eval()
            with np.load(
                self.asset_root / fold_entry["template_file"], allow_pickle=False
            ) as templates:
                bank = np.asarray(templates["templates"], dtype=np.float32)
            self.folds.append(
                {
                    "state": state,
                    "model": model,
                    "templates": bank,
                    "data": DataConfig(**state["data_config"]),
                }
            )

    def _base_volume(self, registered: np.ndarray, data: DataConfig) -> np.ndarray:
        crop = self.raw._intensity01_crop(registered)
        coordinates = np.argwhere(self.node4_target_crop)
        if not len(coordinates):
            raise RuntimeError("The frozen node04 target crop is empty.")
        volume = _physical_centered_crop(
            crop,
            coordinates.mean(axis=0),
            self.spacing_xyz,
            tuple(data.output_shape_zyx),
            float(data.output_spacing_mm),
            order=1,
        )
        mask = physical_striatal_roi(
            volume,
            spacing_mm=float(data.output_spacing_mm),
            radii_zyx_mm=tuple(data.striatal_roi_radii_zyx_mm),
            hot_percentile=float(data.striatal_roi_hot_percentile),
            recenter_max_mm=float(data.striatal_roi_recenter_max_mm),
        )
        # Reproduce the float16 base-cache boundary used during training.
        return self_normalize_striatum(volume, mask).astype(np.float16).astype(np.float32)

    def _fold_logit(
        self,
        base: np.ndarray,
        sbr_features: dict[str, float],
        sbr_valid: bool,
        fold: dict[str, Any],
    ) -> float:
        data: DataConfig = fold["data"]
        volume, _, _, _, _ = _affine_refine(
            base, fold["templates"], config=data, device=self.device
        )
        mask = ndimage.binary_closing(volume > 0, iterations=1)
        volume = self_normalize_striatum(volume, mask)
        slab = axial_slab(volume, mask, data.slab_slices)
        # Reproduce the float16 fold-cache boundary used for training and CV5.
        volume = volume.astype(np.float16).astype(np.float32)
        slab = slab.astype(np.float16).astype(np.float32)
        regional_features = extract_regional930(volume, mask)
        state = fold["state"]
        regional = _apply_regional_transform(
            regional_features, state["regional_transformer"]
        )
        sbr = _apply_sbr_transform(
            sbr_features,
            list(map(str, state["sbr_columns"])),
            state["sbr_transformer"],
            sbr_valid,
        )
        image = torch.from_numpy(slab)[None, None].to(self.device)
        tabular = torch.from_numpy(regional).to(self.device)
        sbr_tensor = torch.from_numpy(sbr).to(self.device)
        valid_tensor = torch.tensor([float(sbr_valid)], device=self.device)
        with torch.inference_mode():
            output = fold["model"](
                image,
                radiomics=tabular,
                sbr=sbr_tensor,
                sbr_valid=valid_tensor,
            )
        return float(output["logit"].item())

    def predict_one(self, path: Path) -> float:
        registered = self.raw.register(Path(path))
        sbr_features, sbr_valid = extract_sbr_features(registered, self.full_masks)
        data: DataConfig = self.folds[0]["data"]
        base = self._base_volume(registered, data)
        logits = np.asarray(
            [
                self._fold_logit(base, sbr_features, sbr_valid, fold)
                for fold in self.folds
            ],
            dtype=np.float64,
        )
        temperature = max(float(self.manifest["temperature"]), 1e-6)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
        return float(probabilities.mean())

    def predict(self, paths: list[Path]) -> np.ndarray:
        return np.asarray([self.predict_one(path) for path in paths], dtype=np.float64)
