from __future__ import annotations

import numpy as np
import pytest

from modeling import run_embedding_search


def _arguments(tmp_path, methods: tuple[str, ...]) -> dict:
    rng = np.random.default_rng(20260821)
    y = np.repeat([0, 1], 40)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    X[:, 0] += y * 0.8
    groups = np.asarray([f"family_{index % 8}" for index in range(len(y))])
    spacing = np.asarray([1.5, 2.5, 3.5, 4.0] * 20)
    uids = [f"case_{index:03d}" for index in range(len(y))]
    return {
        "X": X,
        "y": y,
        "groups": groups,
        "spacing": spacing,
        "uids": uids,
        "output_dir": tmp_path / "embedding_run",
        "experiment_config": {"test": "resume"},
        "n_trials_per_method": 1,
        "seeds": (20260821, 20260822),
        "cpu_jobs": 2,
        "evaluation_rows": 80,
        "evaluation_neighbors": 8,
        "methods": methods,
    }


def test_tsne_embedding_search_resume_and_outputs(tmp_path) -> None:
    arguments = _arguments(tmp_path, ("tsne",))
    y = arguments["y"]

    first = run_embedding_search(**arguments)
    second = run_embedding_search(**arguments)

    assert set(first.comparison["method"]) == {"tsne"}
    assert set(first.comparison["selection"]) == {"balanced", "structure"}
    assert len(first.coordinates) == len(y)
    assert first.coordinates.filter(regex=r"_(balanced|structure)_[12]$").notna().all().all()
    complete = second.trials.loc[second.trials["state"] == "COMPLETE"]
    assert complete.groupby("method").size().eq(1).all()
    assert (tmp_path / "embedding_run" / "embedding_search_config.json").exists()


def test_umap_embedding_search_outputs(tmp_path) -> None:
    pytest.importorskip("umap")
    arguments = _arguments(tmp_path, ("umap",))
    result = run_embedding_search(**arguments)

    assert set(result.comparison["method"]) == {"umap"}
    assert result.coordinates.filter(regex=r"^umap_.*_[12]$").notna().all().all()
