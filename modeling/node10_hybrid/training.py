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
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from modeling.cnn.io import (
    atomic_csv,
    atomic_torch_save,
    copy_with_retry,
    restore_rng_state,
    rng_state,
    torch_load_with_retry,
)
from modeling.dat_spect_v2.config import stable_hash

from .augmentation import PairedAugmentationConfig, PairedPhysicalAugment
from .config import TrainConfig
from .models import DualStreamMultiTaskCNN, RegionalGCN, SubtypeMixture, TabularMLP


@dataclass
class FoldResult:
    fold: int
    validation_log_loss: float
    selected_epoch: int
    predictions: pd.DataFrame
    checkpoint_path: Path | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero torch.cuda.is_available() es False.")
    return selected


def _weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int), minlength=2).clip(min=1)
    weights = torch.as_tensor(1.0 / counts[labels.astype(int)], dtype=torch.double)
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class DualViewDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        magnitude: np.ndarray,
        auxiliary: np.ndarray,
        *,
        augmenter: PairedPhysicalAugment | None,
        intensity_scale: float = 1.0,
        consistency_view: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.magnitude = np.asarray(magnitude, dtype=np.float32)
        self.auxiliary = np.asarray(auxiliary, dtype=np.float32)
        self.intensity_scale = max(float(intensity_scale), 1e-6)
        self.augmenter = augmenter
        self.consistency_view = consistency_view

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with np.load(Path(row["node10_cache_path"]), allow_pickle=False) as payload:
            raw = torch.from_numpy(payload["volume_intensity"].astype(np.float32))[None]
            raw = raw / self.intensity_scale
            pattern = torch.from_numpy(payload["volume_selfnorm"].astype(np.float32))[None]
            mask = torch.from_numpy(payload["target_mask"].astype(np.float32))[None]
        clean_raw, clean_pattern = raw, pattern
        if self.augmenter is not None:
            raw, pattern = self.augmenter(clean_raw.clone(), clean_pattern.clone(), mask.clone())
        result = {
            "uid": str(row["uid"]),
            "intensity": raw,
            "selfnorm": pattern,
            "magnitude": torch.from_numpy(self.magnitude[index]),
            "auxiliary": torch.from_numpy(self.auxiliary[index]),
            "label": torch.tensor(float(row["is_pathologic"]), dtype=torch.float32),
        }
        if self.augmenter is not None and self.consistency_view:
            second_raw, second_pattern = self.augmenter(
                clean_raw.clone(), clean_pattern.clone(), mask.clone()
            )
            result["intensity_consistency"] = second_raw
            result["selfnorm_consistency"] = second_pattern
        return result


def _binary_js_divergence(first_logit: torch.Tensor, second_logit: torch.Tensor) -> torch.Tensor:
    first = torch.sigmoid(first_logit).clamp(1e-5, 1 - 1e-5)
    second = torch.sigmoid(second_logit).clamp(1e-5, 1 - 1e-5)
    midpoint = 0.5 * (first + second)

    def bernoulli_kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return probability * torch.log(probability / reference) + (1 - probability) * torch.log(
            (1 - probability) / (1 - reference)
        )

    return 0.5 * (bernoulli_kl(first, midpoint).mean() + bernoulli_kl(second, midpoint).mean())


def _rotate(path: Path) -> None:
    if path.exists():
        copy_with_retry(path, path.with_name("last.prev.pt"))


def _load_resume(path: Path, device: torch.device) -> dict[str, Any] | None:
    for candidate in (path, path.with_name("last.prev.pt")):
        if candidate.exists():
            try:
                return torch_load_with_retry(candidate, map_location=device)
            except RuntimeError:
                continue
    return None


def _magnitude_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["intensity_l2_norm"].to_numpy(float)),
            np.log1p(frame["intensity_mean"].to_numpy(float)),
            np.log1p(frame["intensity_p90"].to_numpy(float)),
            np.log1p(frame["intensity_positive_voxels"].to_numpy(float)),
        ]
    )


def _auxiliary_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for path in frame["node10_cache_path"]:
        with np.load(Path(path), allow_pickle=False) as payload:
            values.append(np.asarray(payload["auxiliary_targets"], dtype=np.float32))
    return np.stack(values)


@torch.no_grad()
def _predict_dual(
    model: DualStreamMultiTaskCNN,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        output = model(
            batch["intensity"].to(device),
            batch["selfnorm"].to(device),
            batch["magnitude"].to(device),
        )
        probability = torch.sigmoid(output["logit"]).cpu().numpy()
        gates = output["gates"].cpu().numpy()
        for uid, label, prob, gate in zip(
            batch["uid"], batch["label"].numpy(), probability, gates, strict=True
        ):
            rows.append(
                {
                    "uid": str(uid),
                    "is_pathologic": int(label),
                    "probability": float(prob),
                    "gate_intensity": float(gate[0]),
                    "gate_pattern": float(gate[1]),
                    "gate_magnitude": float(gate[2]),
                }
            )
    return pd.DataFrame(rows)


def train_dual_fold(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    fold: int,
    train_config: TrainConfig,
    parameters: dict[str, Any],
    output_dir: Path,
    fixed_epochs: int | None = None,
    device: str | None = None,
) -> FoldResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = cohort.merge(folds[["uid", "fold"]], on="uid", validate="one_to_one")
    train_frame = merged.loc[merged["fold"] != fold].reset_index(drop=True)
    valid_frame = merged.loc[merged["fold"] == fold].reset_index(drop=True)
    train_magnitude, valid_magnitude = (
        _magnitude_matrix(train_frame),
        _magnitude_matrix(valid_frame),
    )
    magnitude_scaler = RobustScaler().fit(train_magnitude)
    train_magnitude = magnitude_scaler.transform(train_magnitude).astype(np.float32)
    valid_magnitude = magnitude_scaler.transform(valid_magnitude).astype(np.float32)
    train_auxiliary, valid_auxiliary = (
        _auxiliary_matrix(train_frame),
        _auxiliary_matrix(valid_frame),
    )
    # Uptake, asymmetry and fragmentation are continuous; affected-side remains binary.
    continuous = [0, 1, 2, 3, 4, 5, 7, 8]
    auxiliary_scaler = RobustScaler().fit(train_auxiliary[:, continuous])
    train_auxiliary[:, continuous] = auxiliary_scaler.transform(train_auxiliary[:, continuous])
    valid_auxiliary[:, continuous] = auxiliary_scaler.transform(valid_auxiliary[:, continuous])
    seed = train_config.seed + fold * 1009
    _seed_everything(seed)
    resolved = _device(device)
    model = DualStreamMultiTaskCNN(
        base_channels=int(parameters.get("base_channels", train_config.base_channels)),
        embedding_dim=int(parameters.get("embedding_dim", train_config.embedding_dim)),
        dropout=float(parameters.get("dropout", train_config.dropout)),
    ).to(resolved)
    learning_rate = float(parameters.get("learning_rate", train_config.learning_rate))
    weight_decay = float(parameters.get("weight_decay", train_config.weight_decay))
    auxiliary_weight = float(parameters.get("auxiliary_weight", train_config.auxiliary_weight))
    consistency_weight = float(
        parameters.get("consistency_weight", train_config.consistency_weight)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = int(fixed_epochs or train_config.max_epochs_search)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = train_config.amp and resolved.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    intensity_scale = float(
        np.median(train_frame["intensity_reconstruction_p99"].to_numpy(dtype=float))
    )
    if not np.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("No se pudo ajustar la escala de intensidad dentro del fold.")
    augmentation = PairedPhysicalAugment(
        PairedAugmentationConfig(
            probability=float(
                parameters.get("augmentation_probability", train_config.augmentation_probability)
            ),
            magnitude=float(
                parameters.get("augmentation_magnitude", train_config.augmentation_magnitude)
            ),
            blur_base_fwhm_mm=train_config.blur_base_fwhm_mm,
            correlated_noise_fwhm_mm=train_config.correlated_noise_fwhm_mm,
            correlated_noise_sd_fraction=train_config.correlated_noise_sd_fraction,
            rotation_degrees=float(
                parameters.get("rotation_degrees", train_config.rotation_degrees)
            ),
            translation_fraction=train_config.translation_fraction,
            intensity_gain_range=train_config.intensity_gain_range,
        )
    )
    train_dataset = DualViewDataset(
        train_frame,
        train_magnitude,
        train_auxiliary,
        intensity_scale=intensity_scale,
        augmenter=augmentation,
        consistency_view=consistency_weight > 0,
    )
    valid_dataset = DualViewDataset(
        valid_frame,
        valid_magnitude,
        valid_auxiliary,
        intensity_scale=intensity_scale,
        augmenter=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size_3d,
        sampler=_weighted_sampler(train_frame["is_pathologic"].to_numpy(int), seed),
        num_workers=train_config.num_workers,
        pin_memory=resolved.type == "cuda",
        persistent_workers=train_config.num_workers > 0,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=train_config.batch_size_3d, shuffle=False)
    contract = {
        "fold": fold,
        "parameters": parameters,
        "train_uids": stable_hash(sorted(train_frame["uid"].astype(str))),
        "valid_uids": stable_hash(sorted(valid_frame["uid"].astype(str))),
        "fixed_epochs": fixed_epochs,
        "train_config": asdict(train_config),
        "intensity_scale": intensity_scale,
    }
    checkpoint_hash = stable_hash(contract)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    state = _load_resume(last_path, resolved)
    start_epoch, best_score, best_epoch, patience_count, history = 0, float("inf"), -1, 0, []
    if state is not None:
        if state.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError("Checkpoint dual-stream incompatible; usa otro run_id.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        restore_rng_state(state.get("rng_state", {}))
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        patience_count = int(state["patience_count"])
        history = list(state.get("history", []))
        if state.get("completed"):
            start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            label = batch["label"].to(resolved)
            auxiliary = batch["auxiliary"].to(resolved)
            with torch.amp.autocast(
                device_type=resolved.type,
                dtype=torch.float16 if resolved.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(
                    batch["intensity"].to(resolved),
                    batch["selfnorm"].to(resolved),
                    batch["magnitude"].to(resolved),
                )
                classification = F.binary_cross_entropy_with_logits(output["logit"], label)
                uptake_loss = F.smooth_l1_loss(output["uptake"], auxiliary[:, :6])
                side_loss = F.binary_cross_entropy_with_logits(
                    output["side_logit"], auxiliary[:, 6]
                )
                relation_loss = F.smooth_l1_loss(
                    torch.stack([output["asymmetry"], output["fragmentation"]], dim=1),
                    auxiliary[:, 7:9],
                )
                loss = classification + auxiliary_weight * (
                    0.55 * uptake_loss + 0.20 * side_loss + 0.25 * relation_loss
                )
                if consistency_weight > 0:
                    second = model(
                        batch["intensity_consistency"].to(resolved),
                        batch["selfnorm_consistency"].to(resolved),
                        batch["magnitude"].to(resolved),
                    )
                    loss = loss + consistency_weight * _binary_js_divergence(
                        output["logit"], second["logit"]
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        predictions = _predict_dual(model, valid_loader, resolved)
        score = float(
            log_loss(predictions["is_pathologic"], predictions["probability"], labels=[0, 1])
        )
        improved = score < best_score - 1e-5
        if improved:
            best_score, best_epoch, patience_count = score, epoch, 0
        else:
            patience_count += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_log_loss": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        payload = {
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
            "magnitude_scaler": magnitude_scaler,
            "auxiliary_scaler": auxiliary_scaler,
            "intensity_scale": intensity_scale,
            "completed": False,
        }
        _rotate(last_path)
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(copy.deepcopy(payload), best_path)
        atomic_csv(pd.DataFrame(history), output_dir / "history.csv")
        print(f"[nodo10 dual] fold={fold} epoch={epoch + 1}/{epochs} val={score:.4f}")
        if fixed_epochs is None and patience_count >= train_config.patience:
            break
    final = torch_load_with_retry(last_path, map_location=resolved)
    final["completed"] = True
    atomic_torch_save(final, last_path)
    selected_path = last_path if fixed_epochs is not None else best_path
    selected = torch_load_with_retry(selected_path, map_location=resolved)
    model.load_state_dict(selected["model_state"])
    predictions = _predict_dual(model, valid_loader, resolved)
    predictions["fold"] = fold
    predictions["selected_epoch"] = int(selected["epoch"])
    atomic_csv(predictions, output_dir / "validation_predictions.csv")
    score = float(log_loss(predictions["is_pathologic"], predictions["probability"], labels=[0, 1]))
    return FoldResult(fold, score, int(selected["epoch"]), predictions, selected_path)


def load_graph_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load graph tensors without inheriting lossy float16 node caches.

    The flattened ``graph_nXX_fXX`` columns are written from the original
    float32 graph before the NPZ cache is serialized. They are therefore the
    canonical source for node values and also let old, resumable v1 runs
    recover from graph caches whose raw second moments overflowed float16.
    Adjacency remains safe to load from NPZ because it is bounded in [0, 1].
    """

    graph_columns = sorted(column for column in frame if column.startswith("graph_n"))
    nodes_from_frame = (
        frame[graph_columns].to_numpy(dtype=np.float32) if graph_columns else None
    )
    cached_nodes, adjacency = [], []
    node_shape: tuple[int, ...] | None = None
    for path in frame["node10_cache_path"]:
        with np.load(Path(path), allow_pickle=False) as payload:
            cached = np.asarray(payload["graph_nodes"], dtype=np.float32)
            if node_shape is None:
                node_shape = cached.shape
            elif cached.shape != node_shape:
                raise ValueError(
                    f"Cache de grafo inconsistente: {cached.shape} != {node_shape} en {path}."
                )
            cached_nodes.append(cached)
            adjacency.append(np.asarray(payload["graph_adjacency"], dtype=np.float32))
    adjacency_array = np.stack(adjacency)
    if not np.isfinite(adjacency_array).all():
        raise ValueError("Las matrices de adyacencia contienen NaN o infinito.")
    if nodes_from_frame is not None:
        if node_shape is None or nodes_from_frame.shape[1] != int(np.prod(node_shape)):
            raise ValueError(
                "Las columnas graph_nXX_fXX no coinciden con la forma guardada del grafo."
            )
        nodes_array = nodes_from_frame.reshape(len(frame), *node_shape)
    else:
        nodes_array = np.stack(cached_nodes)
    if not np.isfinite(nodes_array).all():
        raise ValueError(
            "Las características nodales contienen NaN o infinito; reconstruye el cache del nodo 10."
        )
    return nodes_array, adjacency_array


def train_torch_tabular_fold(
    family: str,
    train_values: np.ndarray,
    valid_values: np.ndarray,
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    valid_uids: np.ndarray,
    *,
    fold: int,
    parameters: dict[str, Any],
    train_config: TrainConfig,
    output_dir: Path,
    train_adjacency: np.ndarray | None = None,
    valid_adjacency: np.ndarray | None = None,
    preprocessing_state: dict[str, Any] | None = None,
    fixed_epochs: int | None = None,
    device: str | None = None,
) -> FoldResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _device(device)
    seed = train_config.seed + fold * 2029
    _seed_everything(seed)
    hidden = int(parameters.get("hidden_dim", 96))
    dropout = float(parameters.get("dropout", 0.25))
    if family == "graph_gcn":
        model: nn.Module = RegionalGCN(train_values.shape[-1], hidden, dropout)
    elif family == "subtype_mixture":
        model = SubtypeMixture(
            train_values.shape[-1],
            hidden,
            int(parameters.get("n_subtypes", 4)),
            dropout,
        )
    else:
        model = TabularMLP(train_values.shape[-1], hidden, dropout)
    model = model.to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 2e-4)),
        weight_decay=float(parameters.get("weight_decay", 3e-5)),
    )
    epochs = int(fixed_epochs or train_config.max_epochs_search)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    train_tensors = [torch.from_numpy(train_values.astype(np.float32))]
    valid_tensors = [torch.from_numpy(valid_values.astype(np.float32))]
    if family == "graph_gcn":
        if train_adjacency is None or valid_adjacency is None:
            raise ValueError("GCN requiere matrices de adyacencia.")
        train_tensors.append(torch.from_numpy(train_adjacency.astype(np.float32)))
        valid_tensors.append(torch.from_numpy(valid_adjacency.astype(np.float32)))
    if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
        raise ValueError(f"{family}: los predictores contienen NaN o infinito tras preprocesar.")
    train_tensors.append(torch.from_numpy(train_labels.astype(np.float32)))
    valid_tensors.append(torch.from_numpy(valid_labels.astype(np.float32)))
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=train_config.batch_size_graph,
        sampler=_weighted_sampler(train_labels, seed),
    )
    valid_loader = DataLoader(TensorDataset(*valid_tensors), batch_size=128, shuffle=False)
    contract = {
        "family": family,
        "fold": fold,
        "parameters": parameters,
        "train_shape": list(train_values.shape),
        "valid_uids": stable_hash(sorted(map(str, valid_uids))),
        "fixed_epochs": fixed_epochs,
        "train_config": asdict(train_config),
    }
    checkpoint_hash = stable_hash(contract)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    state = _load_resume(last_path, resolved)
    start, best, best_epoch, patience_count, history = 0, float("inf"), -1, 0, []
    if state is not None:
        if state.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError("Checkpoint tabular/grafo incompatible; usa otro run_id.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        restore_rng_state(state.get("rng_state", {}))
        start = int(state["epoch"]) + 1
        best, best_epoch = float(state["best_score"]), int(state["best_epoch"])
        patience_count, history = int(state["patience_count"]), list(state.get("history", []))
        if state.get("completed"):
            start = epochs

    def logits(
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        values = batch[0].to(resolved)
        if family == "graph_gcn":
            return model(values, batch[1].to(resolved)), None  # type: ignore[misc]
        output = model(values)
        if isinstance(output, dict):
            return output["logit"], output
        return output, None

    @torch.no_grad()
    def predict() -> np.ndarray:
        model.eval()
        output: list[np.ndarray] = []
        for batch in valid_loader:
            value, _ = logits(batch)
            output.append(torch.sigmoid(value).cpu().numpy())
        probability = np.concatenate(output)
        if not np.isfinite(probability).all():
            raise FloatingPointError(
                f"{family}: la red produjo probabilidades no finitas; revisa escala y optimización."
            )
        return probability

    for epoch in range(start, epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            label = batch[-1].to(resolved)
            value, details = logits(batch)
            loss = F.binary_cross_entropy_with_logits(value, label)
            if details is not None and family == "subtype_mixture":
                weights = details["weights"]
                balance = weights.mean(dim=0)
                uniform = torch.full_like(balance, 1.0 / len(balance))
                load_balance = F.mse_loss(balance, uniform)
                diversity = -details["expert_logits"].std(dim=1).mean()
                loss = loss + float(parameters.get("balance_weight", 0.03)) * load_balance
                loss = loss + float(parameters.get("diversity_weight", 0.01)) * diversity
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{family}: pérdida no finita en fold={fold}, epoch={epoch + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        probability = predict()
        score = float(log_loss(valid_labels, probability, labels=[0, 1]))
        improved = score < best - 1e-5
        if improved:
            best, best_epoch, patience_count = score, epoch, 0
        else:
            patience_count += 1
        history.append(
            {"epoch": epoch, "train_loss": np.mean(losses), "validation_log_loss": score}
        )
        payload = {
            "checkpoint_hash": checkpoint_hash,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "history": history,
            "rng_state": rng_state(),
            "preprocessing_state": preprocessing_state,
            "completed": False,
        }
        _rotate(last_path)
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(copy.deepcopy(payload), best_path)
        atomic_csv(pd.DataFrame(history), output_dir / "history.csv")
        if fixed_epochs is None and patience_count >= train_config.patience:
            break
    final = torch_load_with_retry(last_path, map_location=resolved)
    final["completed"] = True
    atomic_torch_save(final, last_path)
    atomic_torch_save(preprocessing_state or {}, output_dir / "preprocessor.pt")
    selected_path = last_path if fixed_epochs is not None else best_path
    selected = torch_load_with_retry(selected_path, map_location=resolved)
    model.load_state_dict(selected["model_state"])
    probability = predict()
    predictions = pd.DataFrame(
        {
            "uid": valid_uids.astype(str),
            "is_pathologic": valid_labels.astype(int),
            "probability": probability,
            "fold": fold,
            "selected_epoch": int(selected["epoch"]),
        }
    )
    atomic_csv(predictions, output_dir / "validation_predictions.csv")
    score = float(log_loss(valid_labels, probability, labels=[0, 1]))
    return FoldResult(fold, score, int(selected["epoch"]), predictions, selected_path)
