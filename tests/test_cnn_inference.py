from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modeling.cnn import DataConfig, ModelConfig, TrainConfig, predict_fold_ensemble
from modeling.cnn.io import atomic_json, atomic_npz, atomic_torch_save
from modeling.cnn.models import HybridDaTClassifier


def _constant_checkpoint(path: Path, logit: float) -> None:
    model_config = ModelConfig(
        architecture="2.5d",
        feature_variant="image_only",
        base_channels=4,
        image_embedding_dim=16,
        dropout=0.0,
        pooling="gap",
    )
    train_config = TrainConfig(
        batch_size_25d=2,
        num_workers=0,
        amp=False,
        consistency_weight=0.0,
    )
    model = HybridDaTClassifier(model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fusion_head[-1].bias.fill_(logit)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "checkpoint_contract": {
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "radiomics_columns": [],
                "sbr_columns": [],
            },
            "radiomics_transformer": None,
            "sbr_transformer": None,
        },
        path,
    )


def test_fold_ensemble_averages_temperature_scaled_probabilities(tmp_path: Path) -> None:
    records = []
    for index in range(3):
        cache_path = tmp_path / "cache" / f"case_{index}.npz"
        atomic_npz(
            cache_path,
            volume=np.full((8, 8, 8), index / 10, dtype=np.float16),
            target_mask=np.ones((8, 8, 8), dtype=np.uint8),
        )
        records.append({"uid": f"case_{index}", "cache_path": str(cache_path)})
    checkpoints = []
    for fold, logit in enumerate((0.0, 2.0)):
        path = tmp_path / "models" / f"fold_{fold}.pt"
        _constant_checkpoint(path, logit)
        checkpoints.append(str(path.resolve()))
    manifest_path = tmp_path / "primary_ensemble_manifest.json"
    atomic_json(
        {
            "fold_checkpoints": checkpoints,
            "temperature": 2.0,
            "inference_rule": (
                "temperature_scale_each_fold_logit_then_mean_probabilities"
            ),
        },
        manifest_path,
    )

    result = predict_fold_ensemble(
        pd.DataFrame(records),
        manifest_path,
        data_config=DataConfig(
            image_shape_3d=(8, 8, 8),
            view_size_2d=16,
            slices_per_view=5,
        ),
        device="cpu",
    ).predictions

    expected = (0.5 + 1.0 / (1.0 + np.exp(-1.0))) / 2.0
    assert np.allclose(result["probability"], expected)
    assert result["n_models"].eq(2).all()
    assert "is_pathologic" not in result
