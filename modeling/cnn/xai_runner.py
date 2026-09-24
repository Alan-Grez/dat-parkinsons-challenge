from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, stable_hash
from .io import atomic_csv, atomic_json, atomic_npz
from .models import HybridDaTClassifier
from .pipeline import PreparedCNNExperiment, load_finalists
from .preprocessing import FoldTabularTransformer, MaskedSBRTransformer
from .xai import (
    GradCAM3D,
    counterfactual_suite_3d,
    occlusion_sensitivity_3d,
    quantify_explanations,
)


def _select_3d_candidate(prepared: PreparedCNNExperiment) -> tuple[str, dict[str, Any]]:
    metrics_path = prepared.run_dir / "final" / "finalist_cv5_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Primero ejecuta la etapa final.")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["architecture"] == "3d"].sort_values(
        "cross_calibrated_log_loss"
    )
    if metrics.empty:
        raise RuntimeError("No existe un finalista 3D para aplicar Grad-CAM 3D.")
    candidate_id = str(metrics.iloc[0]["candidate_id"])
    candidate = next(
        item for item in load_finalists(prepared) if str(item["candidate_id"]) == candidate_id
    )
    return candidate_id, candidate


def _stratified_xai_sample(oof: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    frame = oof.copy()
    frame["absolute_error"] = (
        frame["probability"] - frame["is_pathologic"]
    ).abs()
    frame["background_qc_valid"] = frame["background_qc_valid"].fillna(False).astype(bool)
    selected: list[pd.DataFrame] = []
    groups = list(frame.groupby(["is_pathologic", "background_qc_valid"], sort=True))
    quota = max(1, n_cases // max(len(groups), 1))
    for _, group in groups:
        take = min(len(group), quota)
        hard = group.nlargest(max(1, take // 2), "absolute_error")
        remaining = group.drop(hard.index)
        typical = remaining.sort_values("absolute_error").iloc[
            np.linspace(
                0,
                max(len(remaining) - 1, 0),
                max(0, take - len(hard)),
                dtype=int,
            )
        ] if len(remaining) and take > len(hard) else remaining.iloc[:0]
        selected.extend([hard, typical])
    chosen = pd.concat(selected).drop_duplicates("uid")
    if len(chosen) < n_cases:
        remaining = frame.drop(chosen.index, errors="ignore").nlargest(
            n_cases - len(chosen), "absolute_error"
        )
        chosen = pd.concat([chosen, remaining]).drop_duplicates("uid")
    return chosen.head(n_cases).reset_index(drop=True)


def run_xai_audit(
    prepared: PreparedCNNExperiment,
    *,
    n_cases: int = 24,
    device: str | None = None,
    patch_size: tuple[int, int, int] = (10, 10, 10),
    stride: tuple[int, int, int] = (8, 8, 8),
) -> pd.DataFrame:
    candidate_id, _candidate = _select_3d_candidate(prepared)
    oof_path = prepared.run_dir / "final" / "oof" / candidate_id / "oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    sample = _stratified_xai_sample(oof, n_cases)
    output_dir = prepared.run_dir / "final" / "xai" / candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "xai_quantitative_audit.csv"
    config_payload = {
        "candidate_id": candidate_id,
        "n_cases": n_cases,
        "patch_size": patch_size,
        "stride": stride,
        "priority": "occlusion_and_counterfactual_over_gradcam_on_conflict",
        "counterfactual_scope": "visual_branch_conditioned_on_fixed_tabular_features",
        "background_roi_status": (
            "validated SBR background mask unavailable in node4 crops; "
            "non-target foreground is localization-only and is never erased"
        ),
    }
    config_hash = stable_hash(config_payload)
    config_path = output_dir / "xai_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise RuntimeError("El XAI existente pertenece a otra configuracion; usa otro run_id.")
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    processed = set(existing.get("uid", pd.Series(dtype=str)).astype(str))
    records = existing.to_dict("records")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cohort_by_uid = prepared.cohort.set_index("uid", drop=False)
    randomized_models: dict[int, HybridDaTClassifier] = {}
    for row in sample.itertuples(index=False):
        uid = str(row.uid)
        if uid in processed:
            continue
        fold = int(row.fold)
        checkpoint_path = (
            prepared.run_dir / "final" / "cv5" / candidate_id / f"fold_{fold}" / "last.pt"
        )
        state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
        model_config = ModelConfig(**state["checkpoint_contract"]["model_config"])
        model = HybridDaTClassifier(model_config).to(resolved_device)
        model.load_state_dict(state["model_state"])
        model.eval()
        patient = cohort_by_uid.loc[uid]
        with np.load(Path(patient["cache_path"]), allow_pickle=False) as payload:
            image = torch.from_numpy(payload["volume"].astype(np.float32))[None, None]
            target_mask = torch.from_numpy(payload["target_mask"].astype(np.float32))[None, None]
        image = image.to(resolved_device)
        target_mask = target_mask.to(resolved_device)
        non_target_foreground = ((image.abs() > 0.02) & (target_mask < 0.5)).float()
        tabular = None
        sbr = None
        sbr_valid = None
        radiomics_state = state.get("radiomics_transformer")
        if radiomics_state is not None:
            transformer = FoldTabularTransformer.from_state(radiomics_state)
            columns = list(state["checkpoint_contract"]["radiomics_columns"])
            values = patient[columns].to_numpy(dtype=float)[None]
            tabular = torch.from_numpy(transformer.transform(values)).to(resolved_device)
        sbr_state = state.get("sbr_transformer")
        if sbr_state is not None:
            transformer = MaskedSBRTransformer.from_state(sbr_state)
            columns = list(state["checkpoint_contract"]["sbr_columns"])
            valid = np.asarray([bool(patient["background_qc_valid"])])
            values = patient[columns].to_numpy(dtype=float)[None]
            sbr = torch.from_numpy(transformer.transform(values, valid)).to(resolved_device)
            sbr_valid = torch.tensor(valid.astype(np.float32), device=resolved_device)
        target_layer = model.image_encoder.gradcam_target_layer
        with GradCAM3D(model, target_layer) as gradcam_engine:
            gradcam = gradcam_engine(
                image, tabular=tabular, sbr=sbr, sbr_valid=sbr_valid, return_on_cpu=True
            )
        if fold not in randomized_models:
            devices = (
                [resolved_device.index or torch.cuda.current_device()]
                if resolved_device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(20260821 + fold)
                randomized_models[fold] = HybridDaTClassifier(model_config).to(resolved_device).eval()
        randomized_model = randomized_models[fold]
        with GradCAM3D(
            randomized_model, randomized_model.image_encoder.gradcam_target_layer
        ) as randomized_engine:
            randomized_gradcam = randomized_engine(
                image,
                tabular=tabular,
                sbr=sbr,
                sbr_valid=sbr_valid,
                return_on_cpu=True,
            )
        occlusion = occlusion_sensitivity_3d(
            model,
            image,
            tabular=tabular,
            sbr=sbr,
            sbr_valid=sbr_valid,
            patch_size=patch_size,
            stride=stride,
            inference_batch_size=16,
            return_on_cpu=True,
        )
        counterfactuals = counterfactual_suite_3d(
            model,
            image,
            tabular=tabular,
            sbr=sbr,
            sbr_valid=sbr_valid,
            target_mask=target_mask,
            background_mask=None,
            return_on_cpu=True,
        )
        quantitative = quantify_explanations(
            gradcam,
            occlusion,
            counterfactuals,
            target_mask=target_mask.cpu(),
            background_mask=non_target_foreground.cpu(),
        )[0]
        trained_flat = gradcam.numpy().reshape(-1)
        randomized_flat = randomized_gradcam.numpy().reshape(-1)
        if trained_flat.std() > 1e-8 and randomized_flat.std() > 1e-8:
            randomized_correlation = float(np.corrcoef(trained_flat, randomized_flat)[0, 1])
        else:
            randomized_correlation = float("nan")
        quantitative["gradcam_randomized_correlation"] = randomized_correlation
        quantitative["gradcam_sanity_pass"] = bool(
            np.isfinite(randomized_correlation) and abs(randomized_correlation) < 0.80
        )
        quantitative["counterfactual_scope"] = (
            "visual_branch_conditioned_on_fixed_tabular_features"
        )
        artifact_path = output_dir / "maps" / f"{uid}.npz"
        atomic_npz(
            artifact_path,
            gradcam=gradcam.numpy().astype(np.float16),
            occlusion_signed=occlusion.signed_delta.numpy().astype(np.float16),
            occlusion_importance=occlusion.importance.numpy().astype(np.float16),
            target_mask=target_mask.cpu().numpy().astype(np.uint8),
        )
        records.append(
            {
                "uid": uid,
                "fold": fold,
                "is_pathologic": int(row.is_pathologic),
                "probability": float(row.probability),
                "artifact_path": str(artifact_path.resolve()),
                **quantitative,
            }
        )
        atomic_csv(pd.DataFrame.from_records(records), results_path)
        print(f"[CNN XAI] {len(records)}/{len(sample)} uid={uid}")
    result = pd.DataFrame.from_records(records)
    atomic_csv(result, results_path)
    atomic_json({**config_payload, "config_hash": config_hash}, config_path)
    return result
