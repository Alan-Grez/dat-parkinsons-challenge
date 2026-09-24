from __future__ import annotations

from pathlib import Path

import torch

from modeling.cnn.io import atomic_torch_save, restore_rng_state, rng_state
from modeling.cnn.training import _load_resume_checkpoint


def test_resume_falls_back_to_previous_checkpoint(tmp_path: Path) -> None:
    last = tmp_path / "last.pt"
    previous = tmp_path / "last.prev.pt"
    atomic_torch_save({"epoch": 3, "checkpoint_hash": "valid"}, previous)
    last.write_bytes(b"interrupted checkpoint")

    state, recovered_from = _load_resume_checkpoint(last, torch.device("cpu"))

    assert recovered_from == previous
    assert state is not None
    assert state["epoch"] == 3


def test_restore_rng_normalizes_non_byte_cpu_state() -> None:
    state = rng_state()
    expected = state["torch_cpu"].clone()
    # Reproduces the incompatible dtype/device contract seen after loading a
    # training checkpoint through a non-CPU map_location.
    loaded_device = "cuda" if torch.cuda.is_available() else "cpu"
    state["torch_cpu"] = state["torch_cpu"].to(
        device=loaded_device,
        dtype=torch.int64,
    )

    restore_rng_state(state)

    assert torch.equal(torch.get_rng_state(), expected)
