from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, WeightedRandomSampler

from modeling.cnn.features import available_sbr_columns
from modeling.cnn.io import (
    atomic_csv,
    atomic_torch_save,
    copy_with_retry,
    restore_rng_state,
    rng_state,
    torch_load_with_retry,
)
from modeling.cnn.preprocessing import MaskedSBRTransformer

from .augmentation import PhysicalUnrealisticAugment
from .config import DataConfig, ModelConfig, TrainConfig, stable_hash
from .data import DaTSpectDataset
from .features import FoldRegionalTransformer, build_regional_feature_cache
from .models import MultiTaskDaTClassifier, with_input_dimensions
from .preprocessing import prepare_fold_image_cache


@dataclass
class FoldTrainingResult:
    fold: int
    validation_log_loss: float
    selected_epoch: int
    checkpoint_path: Path
    predictions: pd.DataFrame
    model_config: ModelConfig
    feature_top_k: int
    pca_variance: float | None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _bernoulli_js(first_logit: torch.Tensor, second_logit: torch.Tensor) -> torch.Tensor:
    first = torch.sigmoid(first_logit).clamp(1e-5, 1 - 1e-5)
    second = torch.sigmoid(second_logit).clamp(1e-5, 1 - 1e-5)
    mixture = ((first + second) / 2).clamp(1e-5, 1 - 1e-5)

    def kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return probability * torch.log(probability / reference) + (1 - probability) * torch.log(
            (1 - probability) / (1 - reference)
        )

    return 0.5 * (kl(first, mixture) + kl(second, mixture)).mean()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward(
    model: MultiTaskDaTClassifier, batch: dict[str, Any], image_key: str
) -> dict[str, torch.Tensor]:
    return model(
        batch[image_key],
        radiomics=batch.get("radiomics"),
        sbr=batch.get("sbr"),
        sbr_valid=batch.get("sbr_valid"),
    )


def _batch_size(model: ModelConfig, train: TrainConfig) -> int:
    if model.architecture == "slab2d":
        return train.batch_size_slab
    if model.architecture == "2.5d":
        return train.batch_size_25d
    return train.batch_size_3d


def _make_loader(
    frame: pd.DataFrame,
    *,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    radiomics: np.ndarray | None,
    sbr: np.ndarray | None,
    sbr_valid: np.ndarray | None,
    auxiliary: np.ndarray,
    training: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    augmenter = (
        PhysicalUnrealisticAugment(
            train_config.augmentation,
            spacing_mm=data_config.output_spacing_mm,
            lateral_strategy=model_config.lateral_strategy,
        )
        if training
        else None
    )
    dataset = DaTSpectDataset(
        frame,
        model_config=model_config,
        data_config=data_config,
        radiomics=radiomics,
        sbr=sbr,
        sbr_valid=sbr_valid,
        auxiliary=auxiliary,
        augmenter=augmenter,
        paired_views=training and train_config.consistency_weight > 0,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = False
    if training:
        labels = frame["is_pathologic"].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=2).clip(min=1)
        weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=_batch_size(model_config, train_config),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_config.num_workers > 0,
        prefetch_factor=2 if train_config.num_workers > 0 else None,
        worker_init_fn=_worker_seed if train_config.num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def predict_loader(
    model: MultiTaskDaTClassifier,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        uids = list(batch["uid"])
        batch = _to_device(batch, device)
        output = _forward(model, batch, "image")
        probability = torch.sigmoid(output["logit"])
        for uid, label, logit, prob in zip(
            uids,
            batch["label"].detach().cpu().numpy(),
            output["logit"].detach().cpu().numpy(),
            probability.detach().cpu().numpy(),
            strict=True,
        ):
            records.append(
                {
                    "uid": str(uid),
                    "is_pathologic": int(label),
                    "logit": float(logit),
                    "probability": float(prob),
                }
            )
    return pd.DataFrame(records)


def _load_cached_auxiliary(frame: pd.DataFrame, lateral_strategy: str) -> np.ndarray:
    suffix = "canonical" if lateral_strategy == "canonical" else "native"
    values: list[np.ndarray] = []
    for path in frame["node7_cache_path"]:
        with np.load(Path(path), allow_pickle=False) as payload:
            values.append(np.asarray(payload[f"auxiliary_{suffix}"], dtype=np.float32))
    return np.stack(values)


def _auxiliary_scale(
    train_values: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    center = np.median(train_values, axis=0)
    q25, q75 = np.percentile(train_values, [25, 75], axis=0)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
    return ((values - center) / scale).astype(np.float32), {"center": center, "scale": scale}


def _rotate_checkpoint(path: Path) -> None:
    if path.exists():
        copy_with_retry(path, path.with_name("last.prev.pt"))


def _resume(path: Path, device: torch.device) -> tuple[dict[str, Any] | None, Path | None]:
    errors: list[str] = []
    for candidate in (path, path.with_name("last.prev.pt")):
        if not candidate.exists():
            continue
        try:
            return torch_load_with_retry(candidate, map_location=device), candidate
        except RuntimeError as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError(
            "No se pudo recuperar el checkpoint primario ni alternativo: " + " | ".join(errors)
        )
    return None, None


def train_one_fold(
    cohort: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    fold: int,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    cache_root: Path,
    output_dir: Path,
    upstream_hash: str,
    feature_top_k: int = 128,
    pca_variance: float | None = None,
    fixed_epochs: int | None = None,
    device: str | torch.device | None = None,
) -> FoldTrainingResult:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_manifest = prepare_fold_image_cache(
        cohort,
        fold_manifest,
        fold=fold,
        cache_root=cache_root,
        config=data_config,
        registration_mode=model_config.registration_mode,
        upstream_hash=upstream_hash,
        device=resolved_device,
    )
    fold_frame = cohort.merge(image_manifest, on="uid", how="inner", validate="one_to_one").merge(
        fold_manifest[["uid", "fold"]].assign(uid=lambda value: value["uid"].astype(str)),
        on="uid",
        how="inner",
        validate="one_to_one",
    )
    train_frame = fold_frame.loc[fold_frame["fold"] != fold].reset_index(drop=True)
    valid_frame = fold_frame.loc[fold_frame["fold"] == fold].reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"Fold {fold} vacio.")
    feature_columns: list[str] = []
    regional_transformer: FoldRegionalTransformer | None = None
    train_radiomics = valid_radiomics = None
    if model_config.feature_variant != "image_only":
        inventory = [
            (Path(path).name, Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
            for path in image_manifest["node7_cache_path"]
        ]
        source_hash = stable_hash(inventory)
        regional_path = Path(image_manifest["node7_cache_path"].iloc[0]).parent / (
            f"regional930_{model_config.lateral_strategy}.csv"
        )
        regional = build_regional_feature_cache(
            image_manifest,
            regional_path,
            config=data_config,
            lateral_strategy=model_config.lateral_strategy,
            source_hash=source_hash,
        )
        fold_frame = fold_frame.drop(
            columns=[column for column in regional if column != "uid"], errors="ignore"
        )
        fold_frame = fold_frame.merge(regional, on="uid", how="inner", validate="one_to_one")
        train_frame = fold_frame.loc[fold_frame["fold"] != fold].reset_index(drop=True)
        valid_frame = fold_frame.loc[fold_frame["fold"] == fold].reset_index(drop=True)
        feature_columns = [
            column
            for column in regional
            if column.startswith(("intensity_", "morph_", "texture_", "relation_"))
        ]
        regional_transformer = FoldRegionalTransformer(
            top_k=feature_top_k,
            pca_variance=pca_variance,
            random_seed=train_config.seed + fold,
        ).fit(
            train_frame[feature_columns].to_numpy(dtype=float),
            train_frame["is_pathologic"].to_numpy(dtype=int),
            feature_columns,
        )
        train_radiomics = regional_transformer.transform(
            train_frame[feature_columns].to_numpy(dtype=float)
        )
        valid_radiomics = regional_transformer.transform(
            valid_frame[feature_columns].to_numpy(dtype=float)
        )
    sbr_columns = available_sbr_columns(cohort)
    sbr_transformer: MaskedSBRTransformer | None = None
    train_sbr = valid_sbr = train_sbr_valid = valid_sbr_valid = None
    if model_config.feature_variant == "image_radiomics_sbr":
        train_sbr_valid = train_frame["background_qc_valid"].astype(bool).to_numpy()
        valid_sbr_valid = valid_frame["background_qc_valid"].astype(bool).to_numpy()
        sbr_transformer = MaskedSBRTransformer().fit(
            train_frame[sbr_columns].to_numpy(dtype=float), train_sbr_valid
        )
        train_sbr = sbr_transformer.transform(
            train_frame[sbr_columns].to_numpy(dtype=float), train_sbr_valid
        )
        valid_sbr = sbr_transformer.transform(
            valid_frame[sbr_columns].to_numpy(dtype=float), valid_sbr_valid
        )
    train_auxiliary_raw = _load_cached_auxiliary(train_frame, model_config.lateral_strategy)
    valid_auxiliary_raw = _load_cached_auxiliary(valid_frame, model_config.lateral_strategy)
    train_auxiliary, auxiliary_state = _auxiliary_scale(train_auxiliary_raw, train_auxiliary_raw)
    valid_auxiliary = (
        (valid_auxiliary_raw - auxiliary_state["center"]) / auxiliary_state["scale"]
    ).astype(np.float32)
    effective_model = with_input_dimensions(
        model_config,
        radiomics_dim=regional_transformer.output_dim if regional_transformer else 0,
        sbr_dim=sbr_transformer.output_dim if sbr_transformer else 0,
    )
    fold_seed = train_config.seed + fold * 1009
    _seed_everything(fold_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    epochs = fixed_epochs if fixed_epochs is not None else train_config.max_epochs_search
    contract = {
        "fold": fold,
        "model": asdict(effective_model),
        "data": asdict(data_config),
        "train": asdict(train_config),
        "feature_top_k": feature_top_k,
        "pca_variance": pca_variance,
        "fixed_epochs": fixed_epochs,
        "train_uids": stable_hash(sorted(train_frame["uid"].astype(str).tolist())),
        "valid_uids": stable_hash(sorted(valid_frame["uid"].astype(str).tolist())),
        "upstream_hash": upstream_hash,
        "regional_feature_columns": feature_columns,
        "sbr_columns": sbr_columns,
    }
    checkpoint_hash = stable_hash(contract)
    model = MultiTaskDaTClassifier(effective_model).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(train_config.amp and resolved_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_score = float("inf")
    best_epoch = -1
    patience_count = 0
    completed = False
    history: list[dict[str, Any]] = []
    state, resumed_from = _resume(last_path, resolved_device)
    if state is not None:
        if state.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError("El checkpoint del nodo 07 pertenece a otra configuracion/run_id.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        patience_count = int(state["patience_count"])
        history = list(state.get("history", []))
        completed = bool(state.get("completed", False))
        restore_rng_state(state.get("rng_state", {}))
        if resumed_from != last_path:
            atomic_torch_save(state, last_path)
    valid_loader = _make_loader(
        valid_frame,
        model_config=effective_model,
        data_config=data_config,
        train_config=train_config,
        radiomics=valid_radiomics,
        sbr=valid_sbr,
        sbr_valid=valid_sbr_valid,
        auxiliary=valid_auxiliary,
        training=False,
        seed=fold_seed + 1,
    )
    stopped_early = False
    epoch_range = range(epochs, epochs) if completed else range(start_epoch, epochs)
    for epoch in epoch_range:
        train_loader = _make_loader(
            train_frame,
            model_config=effective_model,
            data_config=data_config,
            train_config=train_config,
            radiomics=train_radiomics,
            sbr=train_sbr,
            sbr_valid=train_sbr_valid,
            auxiliary=train_auxiliary,
            training=True,
            seed=fold_seed + epoch * 1_000_003,
        )
        model.train()
        losses: list[float] = []
        consistency_weight = train_config.consistency_weight * min(
            1.0, (epoch + 1) / max(train_config.consistency_warmup_epochs, 1)
        )
        for batch in train_loader:
            batch = _to_device(batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16 if resolved_device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                first = _forward(model, batch, "image")
                classification = F.binary_cross_entropy_with_logits(first["logit"], batch["label"])
                auxiliary_loss = F.smooth_l1_loss(first["auxiliary"], batch["auxiliary"])
                loss = classification + train_config.auxiliary_weight * auxiliary_loss
                if "image_view2" in batch and consistency_weight > 0:
                    second = _forward(model, batch, "image_view2")
                    second_classification = F.binary_cross_entropy_with_logits(
                        second["logit"], batch["label"]
                    )
                    second_auxiliary = F.smooth_l1_loss(second["auxiliary"], batch["auxiliary"])
                    loss = 0.5 * (
                        loss
                        + second_classification
                        + train_config.auxiliary_weight * second_auxiliary
                    ) + consistency_weight * _bernoulli_js(first["logit"], second["logit"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        score_now = fixed_epochs is None or epoch == epochs - 1
        if score_now:
            validation = predict_loader(model, valid_loader, resolved_device)
            validation_score = float(
                log_loss(
                    validation["is_pathologic"],
                    validation["probability"].clip(1e-6, 1 - 1e-6),
                    labels=[0, 1],
                )
            )
        else:
            validation_score = float("nan")
        improved = bool(
            fixed_epochs is None
            and np.isfinite(validation_score)
            and validation_score < best_score - 1e-5
        )
        if fixed_epochs is None:
            if improved:
                best_score, best_epoch, patience_count = validation_score, epoch, 0
            else:
                patience_count += 1
        elif score_now:
            best_score, best_epoch = validation_score, epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_log_loss": validation_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "augmentation_magnitude": train_config.augmentation.magnitude,
                "augmentation_probability": train_config.augmentation.probability,
                "improved": improved,
            }
        )
        print(
            f"[nodo07] fold={fold} epoch={epoch + 1}/{epochs} "
            f"train={np.mean(losses):.4f} val={validation_score:.4f}"
        )
        state = {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_contract": contract,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "history": history,
            "rng_state": rng_state(),
            "regional_transformer": regional_transformer.to_state()
            if regional_transformer
            else None,
            "sbr_transformer": sbr_transformer.to_state() if sbr_transformer else None,
            "auxiliary_transformer": auxiliary_state,
            "completed": False,
        }
        _rotate_checkpoint(last_path)
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(copy.deepcopy(state), best_path)
        atomic_csv(pd.DataFrame(history), history_path)
        if fixed_epochs is None and patience_count >= train_config.patience:
            stopped_early = True
            break
    if not last_path.exists():
        raise RuntimeError("No se genero checkpoint del nodo 07.")
    final_state = torch_load_with_retry(last_path, map_location=resolved_device)
    if not final_state.get("completed", False):
        final_state["completed"] = True
        final_state["stopped_early"] = stopped_early
        atomic_torch_save(final_state, last_path)
    selected_path = last_path if fixed_epochs is not None else best_path
    if not selected_path.exists():
        selected_path = last_path
    selected_state = torch_load_with_retry(selected_path, map_location=resolved_device)
    model.load_state_dict(selected_state["model_state"])
    predictions = predict_loader(model, valid_loader, resolved_device)
    predictions["fold"] = fold
    predictions["selected_epoch"] = int(selected_state["epoch"])
    atomic_csv(predictions, output_dir / "validation_predictions.csv")
    score = float(
        log_loss(
            predictions["is_pathologic"],
            predictions["probability"].clip(1e-6, 1 - 1e-6),
            labels=[0, 1],
        )
    )
    return FoldTrainingResult(
        fold,
        score,
        int(selected_state["epoch"]),
        selected_path,
        predictions,
        effective_model,
        feature_top_k,
        pca_variance,
    )
