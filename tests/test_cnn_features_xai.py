from __future__ import annotations

import numpy as np
import torch

from modeling.cnn import SBR_FEATURE_COLUMNS, DataConfig, HybridDaTClassifier, ModelConfig
from modeling.cnn.features import extract_background_independent_features
from modeling.cnn.xai import (
    GradCAM3D,
    counterfactual_suite_3d,
    occlusion_sensitivity_3d,
    quantify_explanations,
)


def test_core_radiomics_are_invariant_to_global_intensity_scale() -> None:
    rng = np.random.default_rng(42)
    volume = rng.uniform(1, 100, size=(16, 16, 16)).astype(np.float32)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[3:13, 3:13, 3:13] = 1
    config = DataConfig(texture_levels=8)
    first = extract_background_independent_features(volume, mask, config)
    second = extract_background_independent_features(volume * 17.0, mask, config)
    for key in first:
        assert np.isclose(first[key], second[key], rtol=2e-4, atol=2e-4), key
    assert first["core_highuptake_volume_ml"] > 0
    assert np.isfinite(first["core_highuptake_sphericity"])


def test_sbr_branch_excludes_technical_stability_qc() -> None:
    assert not any(column.startswith("stability_") for column in SBR_FEATURE_COLUMNS)


def test_3d_xai_returns_quantitative_priority() -> None:
    model = HybridDaTClassifier(
        ModelConfig(
            architecture="3d",
            feature_variant="image_only",
            base_channels=4,
            image_embedding_dim=16,
            dropout=0.0,
            pooling="gap",
        )
    )
    image = torch.rand(1, 1, 16, 16, 16)
    target = torch.zeros_like(image)
    target[:, :, 4:12, 4:12, 4:12] = 1
    background = (1 - target) * (image > 0.05)
    with GradCAM3D(model, model.image_encoder.gradcam_target_layer) as engine:
        gradcam = engine(image)
    occlusion = occlusion_sensitivity_3d(
        model,
        image,
        patch_size=8,
        stride=8,
        inference_batch_size=4,
    )
    counterfactuals = counterfactual_suite_3d(
        model,
        image,
        target_mask=target,
        background_mask=background,
        noise_std=0.0,
    )
    row = quantify_explanations(
        gradcam,
        occlusion,
        counterfactuals,
        target_mask=target,
        background_mask=background,
    )[0]
    assert gradcam.shape == image.shape
    assert occlusion.importance.shape == image.shape
    assert row["evidence_priority"] == "occlusion+counterfactual"
    assert "contradiction" in row
