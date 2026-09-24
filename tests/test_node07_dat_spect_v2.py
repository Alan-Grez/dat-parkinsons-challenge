from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modeling.dat_spect_v2.augmentation import PhysicalUnrealisticAugment
from modeling.dat_spect_v2.config import (
    AugmentationConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from modeling.dat_spect_v2.features import (
    FoldRegionalTransformer,
    build_regional_feature_cache,
    extract_regional930,
)
from modeling.dat_spect_v2.folds import create_or_load_patient_stratified_folds
from modeling.dat_spect_v2.models import MultiTaskDaTClassifier
from modeling.dat_spect_v2.preprocessing import (
    audit_base_cache,
    axial_slab,
    canonicalize_by_putamen,
    physical_striatal_roi,
    prepare_fold_image_cache,
    regional_targets,
    self_normalize_striatum,
)
from modeling.dat_spect_v2.training import train_one_fold


def _synthetic_striatum(shape: tuple[int, int, int] = (16, 24, 24)) -> tuple[np.ndarray, np.ndarray]:
    zz, yy, xx = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    volume = np.zeros(shape, dtype=np.float32)
    for center, value in (
        ((8, 8, 7), 1.4),
        ((8, 15, 7), 0.7),
        ((8, 8, 17), 1.3),
        ((8, 15, 17), 1.1),
    ):
        selected = (
            ((zz - center[0]) / 2.2) ** 2
            + ((yy - center[1]) / 3.0) ** 2
            + ((xx - center[2]) / 2.6) ** 2
            <= 1
        )
        mask |= selected
        volume[selected] = value
    return volume, mask


def test_self_normalization_is_multiplicatively_invariant() -> None:
    volume, mask = _synthetic_striatum()
    first = self_normalize_striatum(volume, mask)
    second = self_normalize_striatum(volume * 37.0, mask)
    np.testing.assert_allclose(first, second, atol=1e-6)
    selected = mask & (first > 0)
    assert np.isclose(np.linalg.norm(first[selected]), np.sqrt(selected.sum()), atol=1e-5)


def test_physical_striatal_roi_is_bounded_and_scale_invariant() -> None:
    volume, _ = _synthetic_striatum(shape=(40, 72, 72))
    first = physical_striatal_roi(
        volume,
        spacing_mm=2.0,
        radii_zyx_mm=(18.0, 34.0, 44.0),
        hot_percentile=85.0,
        recenter_max_mm=10.0,
    )
    second = physical_striatal_roi(
        volume * 100.0,
        spacing_mm=2.0,
        radii_zyx_mm=(18.0, 34.0, 44.0),
        hot_percentile=85.0,
        recenter_max_mm=10.0,
    )
    np.testing.assert_array_equal(first, second)
    assert 5_000 < int(first.sum()) < 30_000


def test_slab_has_requested_shape_and_lateral_canonicalization() -> None:
    volume, mask = _synthetic_striatum()
    normalized = self_normalize_striatum(volume, mask)
    native_targets = regional_targets(normalized, mask)
    canonical, canonical_mask, flipped, affected = canonicalize_by_putamen(normalized, mask)
    canonical_targets = regional_targets(canonical, canonical_mask)
    assert affected == "right"
    assert not flipped
    assert native_targets[1] <= native_targets[3]
    assert canonical_targets[1] <= canonical_targets[3]
    assert axial_slab(canonical, canonical_mask, 6).shape == volume.shape[1:]


def test_regional_bank_has_exactly_930_features() -> None:
    volume, mask = _synthetic_striatum()
    features = extract_regional930(self_normalize_striatum(volume, mask), mask)
    assert len(features) == 930
    assert np.isfinite(np.asarray(list(features.values()), dtype=float)).all()


def test_fold_transformer_fits_selector_and_pca_only_from_given_training_values() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(40, 80))
    y = np.tile([0, 1], 20)
    transformer = FoldRegionalTransformer(top_k=16, pca_variance=0.95).fit(
        values, y, [f"f{index}" for index in range(values.shape[1])]
    )
    transformed = transformer.transform(values[:5])
    assert transformed.shape[0] == 5
    assert 1 <= transformed.shape[1] <= 16
    assert len(transformer.selected_names_ or []) == 16
    restored = FoldRegionalTransformer.from_state(transformer.to_state())
    np.testing.assert_allclose(restored.transform(values[:5]), transformed, atol=1e-6)


def test_patient_folds_are_size_class_and_family_balanced(tmp_path) -> None:
    n = 101
    uids = np.asarray([f"u{index:03d}" for index in range(n)])
    labels = np.asarray(([0, 1] * 51)[:n])
    # A family larger than one target fold proves that family-exclusive CV is
    # impossible, while patient-level folds can still be balanced.
    families = np.asarray(["giant"] * 46 + [f"f{index % 9}" for index in range(55)])
    background = np.asarray([(index % 5) == 0 for index in range(n)])
    folds = create_or_load_patient_stratified_folds(
        uids,
        labels,
        families,
        tmp_path / "folds.csv",
        n_splits=5,
        random_seed=31,
        n_candidates=32,
        background_valid=background,
    )
    summary = folds.groupby("fold").agg(
        n=("uid", "size"),
        prevalence=("is_pathologic", "mean"),
    )
    assert int(summary["n"].max() - summary["n"].min()) <= 1
    assert float(summary["prevalence"].max() - summary["prevalence"].min()) < 0.06
    assert folds.loc[folds["acquisition_family"] == "giant", "fold"].nunique() == 5
    pd.testing.assert_frame_equal(
        folds,
        create_or_load_patient_stratified_folds(
            uids,
            labels,
            families,
            tmp_path / "folds.csv",
            n_splits=5,
            random_seed=31,
            n_candidates=32,
            background_valid=background,
        ),
    )


def test_fold_multitemplate_cache_and_regional_features_resume(tmp_path) -> None:
    rows = []
    fold_rows = []
    for index in range(8):
        volume, mask = _synthetic_striatum()
        volume = np.roll(volume, shift=(index % 3) - 1, axis=-1)
        mask = np.roll(mask, shift=(index % 3) - 1, axis=-1)
        volume = self_normalize_striatum(volume * (1.0 + 0.1 * index), mask)
        canonical, canonical_mask, _, _ = canonicalize_by_putamen(volume, mask)
        path = tmp_path / f"case{index}.npz"
        np.savez_compressed(
            path,
            volume_native=volume,
            target_mask_native=mask.astype(np.uint8),
            volume_canonical=canonical,
            target_mask_canonical=canonical_mask.astype(np.uint8),
        )
        rows.append(
            {
                "uid": f"case{index}",
                "base_cache_path": str(path),
                "is_pathologic": index % 2,
                "acquisition_family": f"f{index % 3}",
                "background_qc_valid": index % 4 == 0,
            }
        )
        fold_rows.append({"uid": f"case{index}", "fold": index % 2})
    cohort = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    config = replace(
        DataConfig(),
        template_sample_per_prototype=2,
        affine_steps=1,
        cache_checkpoint_every=2,
    )
    manifest = prepare_fold_image_cache(
        cohort,
        folds,
        fold=0,
        cache_root=tmp_path / "cache",
        config=config,
        registration_mode="fold_multitemplate_affine",
        upstream_hash="synthetic",
        device="cpu",
    )
    assert len(manifest) == len(cohort)
    assert manifest["node7_cache_path"].map(lambda path: Path(path).exists()).all()
    assert np.isfinite(manifest["registration_post_ncc"]).all()
    features = build_regional_feature_cache(
        manifest,
        tmp_path / "features.csv",
        config=config,
        lateral_strategy="canonical",
        source_hash="synthetic-fold",
    )
    assert features.shape == (len(cohort), 931)
    resumed = prepare_fold_image_cache(
        cohort,
        folds,
        fold=0,
        cache_root=tmp_path / "cache",
        config=config,
        registration_mode="fold_multitemplate_affine",
        upstream_hash="synthetic",
        device="cpu",
    )
    assert resumed["node7_cache_path"].tolist() == manifest["node7_cache_path"].tolist()


def test_base_cache_audit_detects_valid_self_normalized_cases(tmp_path) -> None:
    rows = []
    for index in range(3):
        volume, mask = _synthetic_striatum()
        volume = self_normalize_striatum(volume * (index + 1), mask)
        canonical, canonical_mask, _, _ = canonicalize_by_putamen(volume, mask)
        path = tmp_path / f"audited{index}.npz"
        np.savez_compressed(
            path,
            volume_native=volume.astype(np.float16),
            volume_canonical=canonical.astype(np.float16),
            slab_native=axial_slab(volume, mask, 6).astype(np.float16),
            slab_canonical=axial_slab(canonical, canonical_mask, 6).astype(np.float16),
            target_mask_native=mask.astype(np.uint8),
            target_mask_canonical=canonical_mask.astype(np.uint8),
        )
        rows.append({"uid": f"audited{index}", "base_cache_path": str(path)})
    audit = audit_base_cache(
        pd.DataFrame(rows),
        config=replace(DataConfig(), output_shape_zyx=_synthetic_striatum()[0].shape),
        destination=tmp_path / "audit.csv",
    )
    assert audit["node07_base_qc_pass"].all()
    assert float(audit["node07_selfnorm_relative_error"].max()) < 5e-3


def test_one_epoch_training_checkpoint_and_completed_resume(tmp_path) -> None:
    rows = []
    folds = []
    for index in range(8):
        volume, mask = _synthetic_striatum()
        if index % 2:
            volume = volume.copy()
            volume[mask & (np.indices(mask.shape)[-1] < mask.shape[-1] // 2)] *= 0.55
        volume = self_normalize_striatum(volume, mask)
        canonical, canonical_mask, _, _ = canonicalize_by_putamen(volume, mask)
        path = tmp_path / f"train{index}.npz"
        np.savez_compressed(
            path,
            volume_native=volume,
            target_mask_native=mask.astype(np.uint8),
            volume_canonical=canonical,
            target_mask_canonical=canonical_mask.astype(np.uint8),
        )
        rows.append(
            {
                "uid": f"train{index}",
                "base_cache_path": str(path),
                "is_pathologic": index % 2,
                "acquisition_family": f"f{index % 2}",
                "background_qc_valid": False,
            }
        )
        folds.append({"uid": f"train{index}", "fold": (index // 2) % 2})
    cohort = pd.DataFrame(rows)
    manifest = pd.DataFrame(folds)
    model = ModelConfig(
        architecture="slab2d",
        feature_variant="image_only",
        lateral_strategy="canonical",
        registration_mode="single_template",
        base_channels=8,
        image_embedding_dim=16,
        auxiliary_hidden_dim=8,
        pooling="gmp",
    )
    train = TrainConfig(
        max_epochs_search=1,
        patience=1,
        batch_size_slab=4,
        num_workers=0,
        amp=False,
        consistency_weight=0.0,
        augmentation=AugmentationConfig(
            probability=0.0,
            rotation_degrees=0.0,
            translation_fraction=0.0,
            left_right_flip_probability=0.0,
        ),
    )
    kwargs = {
        "fold": 0,
        "model_config": model,
        "data_config": replace(DataConfig(), cache_checkpoint_every=2),
        "train_config": train,
        "cache_root": tmp_path / "training_cache",
        "output_dir": tmp_path / "training_run",
        "upstream_hash": "synthetic-training",
        "device": "cpu",
    }
    first = train_one_fold(cohort, manifest, **kwargs)
    assert first.checkpoint_path.exists()
    assert len(first.predictions) == 4
    second = train_one_fold(cohort, manifest, **kwargs)
    assert second.validation_log_loss == first.validation_log_loss
    assert second.predictions["uid"].tolist() == first.predictions["uid"].tolist()


def test_physical_augmentation_preserves_mask_and_self_normalization() -> None:
    volume, mask = _synthetic_striatum()
    image = torch.from_numpy(self_normalize_striatum(volume, mask))[None]
    target = torch.from_numpy(mask.astype(np.float32))[None]
    augmenter = PhysicalUnrealisticAugment(
        AugmentationConfig(
            magnitude=2.5,
            probability=1.0,
            rotation_degrees=0.0,
            translation_fraction=0.0,
            left_right_flip_probability=0.0,
        ),
        spacing_mm=2.0,
        lateral_strategy="canonical",
    )
    augmented = augmenter(image, target)
    assert augmented.shape == image.shape
    assert torch.isfinite(augmented).all()
    assert torch.all(augmented[target == 0] == 0)
    selected = augmented[target > 0]
    assert torch.isclose(selected.norm(), torch.sqrt(torch.tensor(float((selected > 0).sum()))), atol=1e-4)


def test_all_node07_architectures_and_fusion_branches_forward() -> None:
    for architecture, image in (
        ("slab2d", torch.rand(2, 1, 72, 72)),
        ("2.5d", torch.rand(2, 3, 5, 72, 72)),
        ("3d", torch.rand(2, 1, 24, 40, 40)),
    ):
        config = ModelConfig(architecture=architecture, pooling="gap_max")
        image_only = MultiTaskDaTClassifier(config)
        output = image_only(image)
        assert output["logit"].shape == (2,)
        assert output["auxiliary"].shape == (2, 4)
        fused_config = replace(
            config,
            feature_variant="image_radiomics_sbr",
            radiomics_dim=12,
            sbr_dim=3,
        )
        fused = MultiTaskDaTClassifier(fused_config)
        output = fused(
            image,
            radiomics=torch.rand(2, 12),
            sbr=torch.rand(2, 3),
            sbr_valid=torch.tensor([1.0, 0.0]),
        )
        assert output["logit"].shape == (2,)
        assert output["sbr_delta"][1].item() == 0.0
