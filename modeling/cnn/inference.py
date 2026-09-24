from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .calibration import apply_temperature
from .config import DataConfig, ModelConfig, TrainConfig
from .io import atomic_csv
from .models import HybridDaTClassifier
from .preprocessing import FoldTabularTransformer, MaskedSBRTransformer
from .training import _make_loader, predict_loader


@dataclass
class EnsemblePredictionResult:
    predictions: pd.DataFrame
    manifest: dict[str, Any]


def _checkpoint_inputs(
    frame: pd.DataFrame,
    state: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    radiomics: np.ndarray | None = None
    sbr: np.ndarray | None = None
    sbr_valid: np.ndarray | None = None
    radiomics_state = state.get("radiomics_transformer")
    if radiomics_state is not None:
        transformer = FoldTabularTransformer.from_state(radiomics_state)
        columns = list(state["checkpoint_contract"]["radiomics_columns"])
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Faltan radiomics requeridos por el checkpoint: {missing}")
        radiomics = transformer.transform(frame[columns].to_numpy(dtype=float))
    sbr_state = state.get("sbr_transformer")
    if sbr_state is not None:
        transformer = MaskedSBRTransformer.from_state(sbr_state)
        columns = list(state["checkpoint_contract"]["sbr_columns"])
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Faltan SBR requeridos por el checkpoint: {missing}")
        if "background_qc_valid" not in frame:
            raise ValueError("La variante SBR requiere background_qc_valid.")
        sbr_valid = frame["background_qc_valid"].fillna(False).astype(bool).to_numpy()
        sbr = transformer.transform(frame[columns].to_numpy(dtype=float), sbr_valid)
    return radiomics, sbr, sbr_valid


def predict_fold_ensemble(
    cohort: pd.DataFrame,
    manifest_path: Path,
    *,
    data_config: DataConfig | None = None,
    output_path: Path | None = None,
    device: str | torch.device | None = None,
) -> EnsemblePredictionResult:
    """Load every frozen fold model and average calibrated probabilities.

    ``cohort`` may be labelled or unlabelled, but must already satisfy the same
    image-cache and derived-feature contract used during training. Acquisition
    family is deliberately ignored by the predictor.
    """

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_paths = [
        path if path.is_absolute() else manifest_path.parent / path
        for path in map(Path, manifest.get("fold_checkpoints", []))
    ]
    if not checkpoint_paths:
        raise ValueError("El manifiesto no contiene fold_checkpoints.")
    missing_paths = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Faltan checkpoints del ensemble: {missing_paths}")
    required = {"uid", "cache_path"}
    missing_columns = required - set(cohort.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas para inferencia: {sorted(missing_columns)}")

    frame = cohort.reset_index(drop=True).copy()
    if frame["uid"].astype(str).duplicated().any():
        raise ValueError("La cohorte de inferencia contiene UIDs duplicados.")
    labelled = "is_pathologic" in frame
    if not labelled:
        frame["is_pathologic"] = 0
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    temperature = float(manifest.get("temperature", 1.0))
    if data_config is None:
        if "data_config" not in manifest:
            raise ValueError("El manifiesto antiguo no contiene data_config; indicalo manualmente.")
        data_config = DataConfig(**manifest["data_config"])
    fold_probabilities: list[np.ndarray] = []
    expected_uids = frame["uid"].astype(str).to_numpy()

    for fold_index, checkpoint_path in enumerate(checkpoint_paths):
        state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
        contract = state.get("checkpoint_contract", {})
        model_config = ModelConfig(**contract["model_config"])
        train_config = TrainConfig(**contract["train_config"])
        model = HybridDaTClassifier(model_config).to(resolved_device)
        model.load_state_dict(state["model_state"])
        radiomics, sbr, sbr_valid = _checkpoint_inputs(frame, state)
        loader = _make_loader(
            frame,
            model_config=model_config,
            data_config=data_config,
            train_config=train_config,
            radiomics=radiomics,
            sbr=sbr,
            sbr_valid=sbr_valid,
            training=False,
            seed=train_config.seed + 900_001 + fold_index,
        )
        predicted = predict_loader(model, loader, resolved_device)
        predicted = predicted.set_index("uid").loc[expected_uids].reset_index()
        calibrated = apply_temperature(predicted["logit"].to_numpy(dtype=float), temperature)
        fold_probabilities.append(calibrated)

    matrix = np.column_stack(fold_probabilities)
    result = pd.DataFrame(
        {
            "uid": expected_uids,
            "probability": matrix.mean(axis=1),
            "fold_probability_std": matrix.std(axis=1, ddof=0),
            "n_models": matrix.shape[1],
        }
    )
    for fold_index in range(matrix.shape[1]):
        result[f"probability_fold_{fold_index}"] = matrix[:, fold_index]
    if labelled:
        result.insert(1, "is_pathologic", frame["is_pathologic"].to_numpy(dtype=int))
    if output_path is not None:
        atomic_csv(result, output_path)
    return EnsemblePredictionResult(result, manifest)
