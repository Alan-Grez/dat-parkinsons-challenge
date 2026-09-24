from __future__ import annotations

import pytest
import torch

from modeling.cnn import HybridDaTClassifier, ModelConfig

MODEL_VARIANTS = [
    ("2.5d", "image_only"),
    ("2.5d", "image_radiomics"),
    ("2.5d", "image_radiomics_sbr"),
    ("3d", "image_only"),
    ("3d", "image_radiomics"),
    ("3d", "image_radiomics_sbr"),
]


def _model_inputs(
    architecture: str,
    feature_variant: str,
    *,
    batch_size: int = 2,
    slices_per_view: int = 5,
    radiomics_dim: int = 7,
    sbr_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if architecture == "2.5d":
        # Three orthogonal planes, K neighboring slices per plane.
        image = torch.randn(batch_size, 3, slices_per_view, 24, 24)
    else:
        image = torch.randn(batch_size, 1, 16, 20, 20)

    uses_radiomics = feature_variant in {"image_radiomics", "image_radiomics_sbr"}
    uses_sbr = feature_variant == "image_radiomics_sbr"
    radiomics = torch.randn(batch_size, radiomics_dim) if uses_radiomics else None
    sbr = torch.randn(batch_size, sbr_dim) if uses_sbr else None
    sbr_valid = torch.tensor([True, False]) if uses_sbr else None
    return image, radiomics, sbr, sbr_valid


@pytest.mark.parametrize(("architecture", "feature_variant"), MODEL_VARIANTS)
def test_hybrid_classifier_forward_for_all_six_variants(
    architecture: str,
    feature_variant: str,
) -> None:
    radiomics_dim = 7 if feature_variant != "image_only" else 0
    sbr_dim = 4 if feature_variant == "image_radiomics_sbr" else 0
    config = ModelConfig(
        architecture=architecture,
        feature_variant=feature_variant,
        in_channels=1,
        slices_per_view=5,
        radiomics_dim=radiomics_dim,
        sbr_dim=sbr_dim,
    )
    model = HybridDaTClassifier(config)
    image, radiomics, sbr, sbr_valid = _model_inputs(
        architecture,
        feature_variant,
        radiomics_dim=max(radiomics_dim, 1),
        sbr_dim=max(sbr_dim, 1),
    )

    logits = model(
        image,
        radiomics=radiomics,
        sbr=sbr,
        sbr_valid=sbr_valid,
    )

    assert logits.shape == (image.shape[0],)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("architecture", ["2.5d", "3d"])
def test_invalid_sbr_values_are_gated_out(architecture: str) -> None:
    config = ModelConfig(
        architecture=architecture,
        feature_variant="image_radiomics_sbr",
        in_channels=1,
        slices_per_view=5,
        radiomics_dim=7,
        sbr_dim=4,
    )
    model = HybridDaTClassifier(config).eval()
    image, radiomics, sbr, sbr_valid = _model_inputs(
        architecture,
        "image_radiomics_sbr",
    )
    assert sbr is not None
    assert sbr_valid is not None

    changed_sbr = sbr.clone()
    changed_sbr[1] = torch.tensor([1.0e4, -1.0e4, 5.0e3, -5.0e3])
    with torch.no_grad():
        baseline = model(
            image,
            radiomics=radiomics,
            sbr=sbr,
            sbr_valid=sbr_valid,
        )
        changed = model(
            image,
            radiomics=radiomics,
            sbr=changed_sbr,
            sbr_valid=sbr_valid,
        )

    # The second observation explicitly declares its background/SBR invalid;
    # therefore arbitrary SBR values must not alter its prediction.
    torch.testing.assert_close(baseline[1], changed[1], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("architecture", ["2.5d", "3d"])
def test_neutral_valid_sbr_does_not_add_a_validity_offset(architecture: str) -> None:
    config = ModelConfig(
        architecture=architecture,
        feature_variant="image_radiomics_sbr",
        base_channels=4,
        image_embedding_dim=16,
        tabular_embedding_dim=8,
        radiomics_dim=7,
        sbr_dim=4,
        dropout=0.0,
    )
    model = HybridDaTClassifier(config).eval()
    image, radiomics, _sbr, _valid = _model_inputs(
        architecture,
        "image_radiomics_sbr",
        radiomics_dim=7,
        sbr_dim=4,
    )
    with torch.no_grad():
        _logit, embeddings = model(
            image,
            radiomics=radiomics,
            sbr=torch.zeros(image.shape[0], 4),
            sbr_valid=torch.ones(image.shape[0]),
            return_embeddings=True,
        )
    torch.testing.assert_close(
        embeddings["sbr_delta"],
        torch.zeros_like(embeddings["sbr_delta"]),
        rtol=0.0,
        atol=1e-7,
    )
