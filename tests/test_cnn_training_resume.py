from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modeling.cnn import DataConfig, ModelConfig, TrainConfig
from modeling.cnn.io import atomic_npz
from modeling.cnn.training import _bernoulli_js, train_one_fold


def test_bernoulli_js_is_finite_for_extreme_fp16_logits() -> None:
    first = torch.tensor([[-20.0], [20.0], [0.0]], dtype=torch.float16)
    second = torch.tensor([[20.0], [-20.0], [0.0]], dtype=torch.float16)

    divergence = _bernoulli_js(first, second)

    assert divergence.dtype == torch.float32
    assert bool(torch.isfinite(divergence))
    assert float(divergence) >= 0.0


def _synthetic_cohort(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    folds = []
    rng = np.random.default_rng(20260821)
    for index in range(12):
        uid = f"case_{index:02d}"
        label = index % 2
        family = f"family_{index // 2:02d}"
        volume = rng.normal(0.1 + 0.15 * label, 0.2, size=(16, 16, 16)).astype(
            np.float32
        )
        target = np.zeros((16, 16, 16), dtype=np.uint8)
        target[4:12, 4:12, 4:12] = 1
        path = root / f"{uid}.npz"
        atomic_npz(path, volume=volume.astype(np.float16), target_mask=target)
        records.append(
            {
                "uid": uid,
                "is_pathologic": label,
                "acquisition_family": family,
                "cache_path": str(path),
                "background_qc_valid": index % 3 == 0,
                "core_mean": float(volume[target > 0].mean()),
                "core_std": float(volume[target > 0].std()),
                "semiquant_sbr": 2.0 + label,
            }
        )
        folds.append(
            {
                "uid": uid,
                "is_pathologic": label,
                "acquisition_family": family,
                "fold": (index // 2) % 3,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(folds)


def test_training_checkpoint_resumes_without_retraining(tmp_path: Path) -> None:
    cohort, folds = _synthetic_cohort(tmp_path / "cache")
    output = tmp_path / "training"
    model = ModelConfig(
        architecture="3d",
        feature_variant="image_radiomics_sbr",
        base_channels=4,
        image_embedding_dim=16,
        tabular_embedding_dim=8,
        sbr_hidden_dim=4,
        dropout=0.1,
        pooling="gap",
    )
    data = DataConfig(image_shape_3d=(16, 16, 16), view_size_2d=16)
    training = TrainConfig(
        batch_size_3d=2,
        batch_size_25d=2,
        max_epochs_search=1,
        patience=2,
        consistency_weight=0.0,
        num_workers=0,
        amp=False,
        seed=7,
    )
    first = train_one_fold(
        cohort,
        folds,
        fold=0,
        model_config=model,
        data_config=data,
        train_config=training,
        output_dir=output,
        device="cpu",
    )
    checkpoint_mtime = (output / "last.pt").stat().st_mtime_ns
    second = train_one_fold(
        cohort,
        folds,
        fold=0,
        model_config=model,
        data_config=data,
        train_config=training,
        output_dir=output,
        device="cpu",
    )
    assert checkpoint_mtime == (output / "last.pt").stat().st_mtime_ns
    assert np.allclose(
        first.predictions.sort_values("uid")["probability"],
        second.predictions.sort_values("uid")["probability"],
    )
    assert (output / "best.pt").exists()
    assert len(second.predictions) == 4
