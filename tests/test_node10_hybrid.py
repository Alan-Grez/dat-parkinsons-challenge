from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import torch

from modeling.node10_hybrid.augmentation import (
    PairedAugmentationConfig,
    PairedPhysicalAugment,
)
from modeling.node10_hybrid.config import DataConfig, ExperimentConfig
from modeling.node10_hybrid.diffusion import DiffusionMapTransformer
from modeling.node10_hybrid.evaluation import (
    _bootstrap_log_loss,
    _pairwise_complementarity,
    _prespecified_mean,
    _subgroup_audit,
)
from modeling.node10_hybrid.features import (
    auxiliary_targets,
    build_hybrid_cache,
    graph_features,
    proxy_region_masks,
    topology_features,
)
from modeling.node10_hybrid.models import DualStreamMultiTaskCNN, RegionalGCN
from modeling.node10_hybrid.search import completed_trials, optimize_until_complete
from modeling.node10_hybrid.training import load_graph_arrays


def _synthetic_volume() -> tuple[np.ndarray, np.ndarray]:
    shape = (16, 24, 24)
    zz, yy, xx = np.indices(shape)
    mask = (((zz - 8) / 6) ** 2 + ((yy - 12) / 9) ** 2 + ((xx - 12) / 10) ** 2) <= 1
    first = np.exp(-((zz - 8) ** 2 + (yy - 8) ** 2 + (xx - 7) ** 2) / 24)
    second = 0.7 * np.exp(-((zz - 8) ** 2 + (yy - 15) ** 2 + (xx - 17) ** 2) / 22)
    volume = ((first + second) * mask).astype(np.float32)
    return volume, mask


def test_dual_stream_outputs_requested_auxiliary_heads() -> None:
    model = DualStreamMultiTaskCNN(base_channels=4, embedding_dim=16, dropout=0.1)
    output = model(
        torch.rand(2, 1, 16, 24, 24),
        torch.rand(2, 1, 16, 24, 24),
        torch.rand(2, 4),
    )
    assert output["logit"].shape == (2,)
    assert output["uptake"].shape == (2, 6)
    assert output["auxiliary"].shape == (2, 9)
    assert output["gates"].shape == (2, 3)
    assert torch.all((output["gates"] >= 0) & (output["gates"] <= 1))


def test_proxy_features_and_graph_are_finite_and_symmetric() -> None:
    raw, mask = _synthetic_volume()
    norm = raw * np.sqrt(mask.sum()) / np.linalg.norm(raw[mask])
    regions = proxy_region_masks(mask)
    assert set(regions) == {
        "right_caudate",
        "right_putamen_anterior",
        "right_putamen_posterior",
        "left_caudate",
        "left_putamen_anterior",
        "left_putamen_posterior",
    }
    auxiliary = auxiliary_targets(raw, mask)
    topology = topology_features(norm, mask, (60.0, 80.0, 90.0))
    nodes, adjacency = graph_features(raw, norm, mask)
    assert auxiliary.shape == (9,)
    assert nodes.shape == (12, 21)
    assert adjacency.shape == (12, 12)
    assert np.isfinite(nodes).all() and np.isfinite(adjacency).all()
    assert np.allclose(adjacency, adjacency.T)
    assert topology and all(np.isfinite(list(topology.values())))


def test_cache_reconstructs_registered_counts_and_materializes_930_features(
    tmp_path: Path,
) -> None:
    import pandas as pd

    raw, mask = _synthetic_volume()
    crops = tmp_path / "crops"
    crops.mkdir()
    np.savez_compressed(
        crops / "case.npz",
        volume_intensity01=raw,
        target_mask=mask.astype(np.uint8),
        spacing_xyz=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
    cohort = pd.DataFrame({"uid": ["case"], "foreground_p99_registered": [125.0]})
    config = DataConfig(
        output_shape_zyx=raw.shape,
        striatal_roi_radii_zyx_mm=(12.0, 18.0, 20.0),
        cache_checkpoint_every=1,
    )
    manifest, topology, regional, graph = build_hybrid_cache(
        crops,
        cohort,
        tmp_path / "cache",
        config=config,
        upstream_hash="synthetic",
    )
    with np.load(Path(manifest.loc[0, "node10_cache_path"])) as payload:
        reconstructed = payload["volume_intensity"].astype(np.float32)
        assert payload["graph_nodes"].dtype == np.float32
    assert reconstructed.max() > 1.0
    assert np.isclose(manifest.loc[0, "intensity_reconstruction_p99"], 125.0)
    assert len([column for column in regional if column.startswith("regional930_")]) == 930
    assert len(topology) == len(graph) == 1


def test_dense_gcn_forward_shape() -> None:
    model = RegionalGCN(input_dim=21, hidden_dim=16, dropout=0.1)
    nodes = torch.rand(3, 12, 21)
    adjacency = torch.eye(12)[None].repeat(3, 1, 1)
    assert model(nodes, adjacency).shape == (3,)


def test_graph_loader_recovers_float16_overflow_from_canonical_columns(
    tmp_path: Path,
) -> None:
    import pandas as pd

    nodes = np.arange(12 * 21, dtype=np.float32).reshape(12, 21)
    nodes[0, 6] = 1.0e8
    adjacency = np.eye(12, dtype=np.float32)
    cache_path = tmp_path / "overflowed_graph.npz"
    with np.errstate(over="ignore"):
        np.savez_compressed(
            cache_path,
            graph_nodes=nodes.astype(np.float16),
            graph_adjacency=adjacency.astype(np.float16),
        )
    row = {
        "node10_cache_path": str(cache_path),
        **{
            f"graph_n{node_index:02d}_f{feature_index:02d}": float(
                nodes[node_index, feature_index]
            )
            for node_index in range(12)
            for feature_index in range(21)
        },
    }
    loaded_nodes, loaded_adjacency = load_graph_arrays(pd.DataFrame([row]))
    assert np.isfinite(loaded_nodes).all()
    assert loaded_nodes[0, 0, 6] == nodes[0, 6]
    assert np.allclose(loaded_adjacency[0], adjacency)


def test_diffusion_map_has_out_of_sample_extension() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(30, 7))
    valid = rng.normal(size=(5, 7))
    transformer = DiffusionMapTransformer(n_components=4, epsilon_quantile=0.5).fit(train)
    embedded = transformer.transform(valid)
    assert embedded.shape == (5, 4)
    assert np.isfinite(embedded).all()
    training_embedding = transformer.transform(train)
    expected = (
        transformer.eigenvectors_
        * np.power(transformer.eigenvalues_, transformer.diffusion_time)[None, :]
    )
    assert np.allclose(training_embedding, expected, atol=2e-5)


def test_paired_augmentation_preserves_alignment_and_reprojects_pattern() -> None:
    raw, mask = _synthetic_volume()
    pattern = raw * np.sqrt(mask.sum()) / np.linalg.norm(raw[mask])
    augmenter = PairedPhysicalAugment(
        PairedAugmentationConfig(
            probability=0.0,
            rotation_degrees=0.0,
            translation_fraction=0.0,
            intensity_gain_range=(1.0, 1.0),
        )
    )
    augmented_raw, augmented_pattern = augmenter(
        torch.from_numpy(raw)[None],
        torch.from_numpy(pattern.astype(np.float32))[None],
        torch.from_numpy(mask.astype(np.float32))[None],
    )
    assert torch.allclose(augmented_raw, torch.from_numpy(raw)[None])
    positive = augmented_pattern > 0
    expected_norm = torch.sqrt(positive.sum().to(torch.float32))
    assert torch.allclose(torch.linalg.vector_norm(augmented_pattern), expected_norm, atol=1e-4)


def test_prespecified_mean_uses_no_labels_to_weight_experts() -> None:
    import pandas as pd

    matrix = pd.DataFrame(
        {
            "uid": ["a", "b"],
            "is_pathologic": [0, 1],
            "fold": [0, 1],
            "p__first": [0.2, 0.8],
            "p__second": [0.4, 0.6],
        }
    )
    result = _prespecified_mean(matrix)
    assert np.allclose(result["probability"], [0.3, 0.7])


def test_final_audits_use_oof_predictions_and_posthoc_acquisition_metadata() -> None:
    import pandas as pd

    rows = []
    for family, probabilities in (
        ("first", [0.2, 0.8, 0.3, 0.7]),
        ("second", [0.3, 0.7, 0.4, 0.6]),
    ):
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "uid": f"case_{index}",
                    "is_pathologic": index % 2,
                    "fold": index % 2,
                    "probability": probability,
                    "probability_cross_calibrated": probability,
                    "family": family,
                    "candidate_id": family,
                }
            )
    oof = pd.DataFrame(rows)
    cohort = pd.DataFrame(
        {"uid": [f"case_{index}" for index in range(4)], "acquisition_family": ["A"] * 4}
    )
    subgroup = _subgroup_audit(oof, cohort)
    bootstrap = _bootstrap_log_loss(oof, seed=1, draws=20)
    complementarity = _pairwise_complementarity(oof)
    assert len(subgroup) == 2
    assert (subgroup["n"] == 4).all()
    assert len(bootstrap) == 2
    assert len(complementarity) == 1
    assert np.isclose(complementarity.loc[0, "mean_absolute_probability_difference"], 0.1)


def test_optuna_replaces_pruned_trials_until_ten_complete(tmp_path: Path) -> None:
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{(tmp_path / 'study.sqlite3').as_posix()}",
        study_name="complete_contract",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        if trial.number % 3 == 0:
            raise optuna.TrialPruned()
        return float(trial.number)

    optimize_until_complete(study, objective, target=2, timeout_seconds=None)
    assert len(completed_trials(study)) == 10
    assert len(study.trials) > 10


def test_default_experiment_enforces_minimum_trials() -> None:
    experiment = ExperimentConfig()
    assert experiment.search.effective_completed_trials >= 10
    assert experiment.search.effective_stack_trials >= 10
    assert "dual_stream_multitask" in experiment.model_families
    assert "regional_hgb" in experiment.model_families
    assert "graph_gcn" in experiment.model_families
    assert experiment.data.preprocessing_version.endswith("_v2")
