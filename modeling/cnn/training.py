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
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TrainConfig, stable_hash
from .data import CropDataset, InvarianceAugment3D
from .features import available_sbr_columns, core_feature_columns
from .io import (
    atomic_csv,
    atomic_torch_save,
    copy_with_retry,
    restore_rng_state,
    rng_state,
    torch_load_with_retry,
)
from .models import HybridDaTClassifier, with_input_dimensions
from .preprocessing import FoldTabularTransformer, MaskedSBRTransformer


@dataclass
class FoldTrainingResult:
    fold: int
    validation_log_loss: float
    selected_epoch: int
    checkpoint_path: Path
    predictions: pd.DataFrame
    model_config: ModelConfig
    radiomics_columns: list[str]
    sbr_columns: list[str]
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


def _bernoulli_js(logits_first: torch.Tensor, logits_second: torch.Tensor) -> torch.Tensor:
    # This divergence must not run in float16.  In fp16, ``1 - 1e-5`` rounds
    # back to exactly 1 and the complementary Bernoulli term becomes
    # ``0 * log(0 / 0)``.  AMP's GradScaler can skip the poisoned update, but
    # the reported loss is still NaN and batches are silently wasted.
    first = torch.sigmoid(logits_first.float()).clamp(1e-6, 1.0 - 1e-6)
    second = torch.sigmoid(logits_second.float()).clamp(1e-6, 1.0 - 1e-6)
    mixture = ((first + second) / 2.0).clamp(1e-6, 1.0 - 1e-6)

    def kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return probability * torch.log(probability / reference) + (
            1.0 - probability
        ) * torch.log((1.0 - probability) / (1.0 - reference))

    return 0.5 * (kl(first, mixture) + kl(second, mixture)).mean()


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_forward(model: HybridDaTClassifier, batch: dict[str, Any], image_key: str) -> torch.Tensor:
    return model(
        batch[image_key],
        radiomics=batch.get("radiomics"),
        sbr=batch.get("sbr"),
        sbr_valid=batch.get("sbr_valid"),
    )


@torch.no_grad()
def predict_loader(
    model: HybridDaTClassifier,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        uids = list(batch["uid"])
        batch = _batch_to_device(batch, device)
        logits = _model_forward(model, batch, "image")
        probabilities = torch.sigmoid(logits)
        labels = batch["label"]
        for uid, label, logit, probability in zip(
            uids,
            labels.detach().cpu().numpy(),
            logits.detach().cpu().numpy(),
            probabilities.detach().cpu().numpy(),
            strict=True,
        ):
            records.append(
                {
                    "uid": str(uid),
                    "is_pathologic": int(label),
                    "logit": float(logit),
                    "probability": float(probability),
                }
            )
    return pd.DataFrame.from_records(records)


def _make_loader(
    frame: pd.DataFrame,
    *,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    radiomics: np.ndarray | None,
    sbr: np.ndarray | None,
    sbr_valid: np.ndarray | None,
    training: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    augmenter = InvarianceAugment3D(train_config) if training else None
    dataset = CropDataset(
        frame,
        model_config=model_config,
        data_config=data_config,
        radiomics=radiomics,
        sbr=sbr,
        sbr_valid=sbr_valid,
        augmenter=augmenter,
        paired_views=training and train_config.consistency_weight > 0,
    )
    batch_size = (
        train_config.batch_size_3d
        if model_config.architecture == "3d"
        else train_config.batch_size_25d
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_config.num_workers > 0,
        prefetch_factor=2 if train_config.num_workers > 0 else None,
        worker_init_fn=_worker_seed if train_config.num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _rotate_last_checkpoint(last_path: Path) -> None:
    if not last_path.exists():
        return
    previous_path = last_path.with_name("last.prev.pt")
    copy_with_retry(last_path, previous_path)


def _load_resume_checkpoint(
    last_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any] | None, Path | None]:
    errors: list[str] = []
    for candidate in (last_path, last_path.with_name("last.prev.pt")):
        if not candidate.exists():
            continue
        try:
            return torch_load_with_retry(candidate, map_location=device), candidate
        except RuntimeError as error:
            errors.append(f"{candidate}: {error}")
    if errors:
        raise RuntimeError(
            "No fue posible recuperar last.pt ni last.prev.pt:\n" + "\n".join(errors)
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
    output_dir: Path,
    pca_variance: float | None = None,
    fixed_epochs: int | None = None,
    scheduler_epochs: int | None = None,
    device: str | torch.device | None = None,
) -> FoldTrainingResult:
    required = {
        "uid",
        "is_pathologic",
        "acquisition_family",
        "cache_path",
        "background_qc_valid",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Faltan columnas para entrenar CNN: {sorted(missing)}")
    cohort_for_fold = cohort.copy()
    cohort_for_fold["uid"] = cohort_for_fold["uid"].astype(str)
    fold_frame = cohort_for_fold.merge(
        fold_manifest[["uid", "fold"]].assign(uid=lambda frame: frame["uid"].astype(str)),
        on="uid",
        how="inner",
        validate="one_to_one",
    )
    train_frame = fold_frame.loc[fold_frame["fold"] != fold].reset_index(drop=True)
    valid_frame = fold_frame.loc[fold_frame["fold"] == fold].reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"Fold {fold} vacio.")
    radiomics_columns = core_feature_columns(cohort)
    sbr_columns = available_sbr_columns(cohort)
    radiomics_transformer: FoldTabularTransformer | None = None
    sbr_transformer: MaskedSBRTransformer | None = None
    train_radiomics = valid_radiomics = None
    train_sbr = valid_sbr = None
    train_sbr_valid = valid_sbr_valid = None
    if model_config.feature_variant != "image_only":
        if not radiomics_columns:
            raise ValueError("No se encontraron radiomics core independientes del fondo.")
        radiomics_transformer = FoldTabularTransformer(pca_variance=pca_variance).fit(
            train_frame[radiomics_columns].to_numpy(dtype=float)
        )
        train_radiomics = radiomics_transformer.transform(
            train_frame[radiomics_columns].to_numpy(dtype=float)
        )
        valid_radiomics = radiomics_transformer.transform(
            valid_frame[radiomics_columns].to_numpy(dtype=float)
        )
    if model_config.feature_variant == "image_radiomics_sbr":
        if not sbr_columns:
            raise ValueError("No se encontraron columnas SBR en el upstream.")
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
    effective_model_config = with_input_dimensions(
        model_config,
        radiomics_transformer.output_dim if radiomics_transformer is not None else 0,
        sbr_transformer.output_dim if sbr_transformer is not None else 0,
    )
    fold_seed = train_config.seed + fold * 1009
    _seed_everything(fold_seed)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "training_history.csv"
    epochs = fixed_epochs if fixed_epochs is not None else train_config.max_epochs_search
    scheduler_horizon = int(scheduler_epochs or epochs)
    if scheduler_horizon < epochs:
        raise ValueError("scheduler_epochs no puede ser menor que las epocas ejecutadas.")
    checkpoint_contract = {
        "model_config": asdict(effective_model_config),
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "pca_variance": pca_variance,
        "fold": fold,
        "fixed_epochs": fixed_epochs,
        "train_uid_hash": stable_hash(sorted(train_frame["uid"].astype(str).tolist())),
        "valid_uid_hash": stable_hash(sorted(valid_frame["uid"].astype(str).tolist())),
        "radiomics_columns": radiomics_columns,
        "sbr_columns": sbr_columns,
        "upstream_hash": (
            str(cohort_for_fold["cnn_upstream_hash"].iloc[0])
            if "cnn_upstream_hash" in cohort_for_fold
            and cohort_for_fold["cnn_upstream_hash"].nunique(dropna=False) == 1
            else None
        ),
    }
    # Preserve checkpoint compatibility for every pre-node11 caller.  Only an
    # explicitly decoupled scheduler horizon extends the scientific contract.
    if scheduler_epochs is not None:
        checkpoint_contract["scheduler_epochs"] = scheduler_horizon
    checkpoint_hash = stable_hash(checkpoint_contract)
    model = HybridDaTClassifier(effective_model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(scheduler_horizon, 1)
    )
    amp_enabled = bool(train_config.amp and resolved_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_score = float("inf")
    best_epoch = -1
    patience_count = 0
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    checkpoint_completed = False
    resume_state: dict[str, Any] | None = None
    state, resumed_from = _load_resume_checkpoint(last_path, resolved_device)
    if state is not None:
        resume_state = state
        if state.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError(
                f"El checkpoint {last_path} pertenece a otra configuracion; usa otro run_id."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        patience_count = int(state["patience_count"])
        checkpoint_completed = bool(state.get("completed", False))
        restore_rng_state(state.get("rng_state", {}))
        if "history" in state:
            history = list(state["history"])
            atomic_csv(pd.DataFrame(history), history_path)
        if resumed_from != last_path:
            print(f"[CNN] recuperado desde checkpoint alternativo: {resumed_from}")
            # Repair the primary atomically only after its contract has been
            # validated. The previous copy remains available during the write.
            atomic_torch_save(state, last_path)

    valid_loader = _make_loader(
        valid_frame,
        model_config=effective_model_config,
        data_config=data_config,
        train_config=train_config,
        radiomics=valid_radiomics,
        sbr=valid_sbr,
        sbr_valid=valid_sbr_valid,
        training=False,
        seed=fold_seed + 1,
    )
    stopped_early = False
    epoch_range = range(epochs, epochs) if checkpoint_completed else range(start_epoch, epochs)
    for epoch in epoch_range:
        # An epoch-specific loader seed makes a resumed run reproduce the same
        # shuffle/augmentations for the epoch it must repeat after an outage.
        train_loader = _make_loader(
            train_frame,
            model_config=effective_model_config,
            data_config=data_config,
            train_config=train_config,
            radiomics=train_radiomics,
            sbr=train_sbr,
            sbr_valid=train_sbr_valid,
            training=True,
            seed=fold_seed + epoch * 1_000_003,
        )
        model.train()
        epoch_losses: list[float] = []
        nonfinite_batches = 0
        consistency_weight = train_config.consistency_weight * min(
            1.0,
            (epoch + 1) / max(train_config.consistency_warmup_epochs, 1),
        )
        for batch in train_loader:
            batch = _batch_to_device(batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16 if resolved_device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits_first = _model_forward(model, batch, "image")
                supervised = F.binary_cross_entropy_with_logits(
                    logits_first, batch["label"]
                )
                if "image_view2" in batch and consistency_weight > 0:
                    logits_second = _model_forward(model, batch, "image_view2")
                    supervised = 0.5 * (
                        supervised
                        + F.binary_cross_entropy_with_logits(
                            logits_second, batch["label"]
                        )
                    )
                    loss = supervised + consistency_weight * _bernoulli_js(
                        logits_first, logits_second
                    )
                else:
                    loss = supervised
            if not bool(torch.isfinite(loss).all()):
                # Do not backpropagate a non-finite objective.  One isolated
                # bad batch is recoverable; repeated failures indicate a real
                # numerical/data problem and should stop the run loudly.
                nonfinite_batches += 1
                if nonfinite_batches >= 3:
                    raise FloatingPointError(
                        f"Perdida no finita en {nonfinite_batches} batches de "
                        f"fold={fold}, epoch={epoch + 1}."
                    )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise FloatingPointError(
                f"Ningun batch finito en fold={fold}, epoch={epoch + 1}."
            )
        scheduler.step()
        # During the three-fold search, validation drives early stopping and
        # Optuna's objective. During the frozen five-fold evaluation, the epoch
        # count is already fixed from the search; inspecting the outer fold at
        # every epoch would waste inference time and make it easier to select on
        # the evaluation fold by accident. Only score the final fixed epoch.
        score_this_epoch = fixed_epochs is None or epoch == epochs - 1
        if score_this_epoch:
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
                best_score = validation_score
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1
        elif score_this_epoch:
            best_score = validation_score
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_log_loss": validation_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "consistency_weight": consistency_weight,
                "nonfinite_batches": nonfinite_batches,
                "improved": improved,
            }
        )
        validation_text = (
            f"{validation_score:.4f}" if np.isfinite(validation_score) else "deferred"
        )
        best_text = f"{best_score:.4f}" if np.isfinite(best_score) else "pending"
        print(
            f"[CNN] fold={fold} epoch={epoch + 1}/{epochs} "
            f"train={np.mean(epoch_losses):.4f} val_log_loss={validation_text} "
            f"best={best_text} nonfinite_batches={nonfinite_batches}"
        )
        state = {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_contract": checkpoint_contract,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "rng_state": rng_state(),
            "history": history,
            "radiomics_transformer": (
                radiomics_transformer.to_state() if radiomics_transformer is not None else None
            ),
            "sbr_transformer": (
                sbr_transformer.to_state() if sbr_transformer is not None else None
            ),
            "completed": False,
        }
        _rotate_last_checkpoint(last_path)
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(copy.deepcopy(state), best_path)
        atomic_csv(pd.DataFrame(history), history_path)
        if fixed_epochs is None and patience_count >= train_config.patience:
            stopped_early = True
            break

    if not last_path.exists():
        raise RuntimeError("No se genero ningun checkpoint de entrenamiento.")
    if checkpoint_completed and resume_state is not None:
        final_state = resume_state
    else:
        final_state = torch_load_with_retry(last_path, map_location=resolved_device)
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
    selected_score = float(
        log_loss(
            predictions["is_pathologic"],
            predictions["probability"].clip(1e-6, 1 - 1e-6),
            labels=[0, 1],
        )
    )
    return FoldTrainingResult(
        fold=fold,
        validation_log_loss=selected_score,
        selected_epoch=int(selected_state["epoch"]),
        checkpoint_path=selected_path,
        predictions=predictions,
        model_config=effective_model_config,
        radiomics_columns=radiomics_columns,
        sbr_columns=sbr_columns,
        pca_variance=pca_variance,
    )
