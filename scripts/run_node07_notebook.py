"""Execute notebook 07 and persist a private copy with embedded outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbformat import NotebookNode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cnn.io import RunLock
from modeling.cnn.pipeline import validate_path_component

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_dat_spect_slab_multitemplate.ipynb"
HEARTBEAT_SECONDS = 60


def _atomic_notebook(notebook: NotebookNode, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    nbformat.write(notebook, temporary)
    os.replace(temporary, destination)


def _completed_trials(run_dir: Path) -> int:
    complete = 0
    for path in (run_dir / "search" / "studies").glob("*_trials.csv"):
        try:
            trials = pd.read_csv(path, usecols=["state"])
            complete += int(trials["state"].eq("COMPLETE").sum())
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
            continue
    return complete


def _heartbeat(stop: threading.Event, run_dir: Path, cache_root: Path, started: float) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        elapsed = (time.monotonic() - started) / 60
        prepared = run_dir / "config" / "prepared_cohort.csv"
        cohort_rows = 0
        if prepared.exists():
            try:
                cohort_rows = len(pd.read_csv(prepared, usecols=["uid"]))
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pass
        base_cases = len(list((cache_root / "base").glob("*/*.npz")))
        fold_cases = len(list(cache_root.glob("**/fold_registered/*/*.npz")))
        final_models = len(list((run_dir / "final" / "cv5").glob("*/fold_*/last.pt")))
        print(
            f"[nodo07 heartbeat] {elapsed:.1f} min · cohorte={cohort_rows:,} · "
            f"base={base_cases:,} · registros-fold={fold_cases:,} · "
            f"trials={_completed_trials(run_dir):,} · modelos-finales={final_models:,}",
            flush=True,
        )


def _execute(run_dir: Path, cache_root: Path) -> Path:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    output_dir = run_dir / "executed_notebooks"
    destination = output_dir / f"{NOTEBOOK_PATH.stem}.executed.ipynb"
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(stop, run_dir, cache_root, time.monotonic()),
        daemon=True,
        name="node07-notebook-heartbeat",
    )
    heartbeat.start()
    try:
        client.execute()
    except BaseException:
        partial = output_dir / f"{NOTEBOOK_PATH.stem}.partial.ipynb"
        _atomic_notebook(notebook, partial)
        print(f"Notebook parcial guardado en: {partial}", flush=True)
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2)
    _atomic_notebook(notebook, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="dat_spect_slab_v4")
    parser.add_argument("--node4-profile", default="v3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_path_component(args.run_id, "--run-id")
    validate_path_component(args.node4_profile, "--node4-profile")
    run_dir = PROJECT_ROOT / "outputs" / "private_eda" / "node07_runs" / args.run_id
    cache_root = (
        PROJECT_ROOT / "outputs" / "private_eda" / "node07_cache" / args.node4_profile
    )
    variables = {
        "DAT_NODE07_RUN_ID": args.run_id,
        "DAT_NODE07_NODE4_PROFILE": args.node4_profile,
    }
    if args.dry_run:
        print(
            json.dumps(
                {"notebook": str(NOTEBOOK_PATH), "run_dir": str(run_dir), **variables},
                indent=2,
            )
        )
        return
    for name, value in variables.items():
        os.environ[name] = value
    # Regenerate the clean source before executing the private copy.
    import scripts.materialize_node07_notebook  # noqa: F401

    with RunLock(run_dir / ".run.lock"):
        destination = _execute(run_dir, cache_root)
    print(f"Notebook ejecutado y persistido en: {destination}", flush=True)


if __name__ == "__main__":
    main()
