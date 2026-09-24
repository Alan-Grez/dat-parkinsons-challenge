"""Execute notebook 06 and persist a private copy with embedded outputs."""

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
from scripts.materialize_cnn_notebook import NOTEBOOK_PATH
from scripts.materialize_cnn_notebook import main as materialize_notebook

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


def _xai_rows(run_dir: Path) -> int:
    rows = 0
    for path in (run_dir / "final" / "xai").glob("*/xai_quantitative_audit.csv"):
        try:
            rows += len(pd.read_csv(path, usecols=["uid"]))
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
    return rows


def _heartbeat(stop: threading.Event, run_dir: Path, started_at: float) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        elapsed = (time.monotonic() - started_at) / 60
        cohort_rows = 0
        prepared = run_dir / "config" / "prepared_cohort_manifest.csv"
        if prepared.exists():
            try:
                cohort_rows = len(pd.read_csv(prepared, usecols=["uid"]))
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pass
        final_models = len(list((run_dir / "final" / "cv5").glob("*/fold_*/last.pt")))
        print(
            f"[CNN heartbeat] {elapsed:.1f} min · cohorte={cohort_rows:,} · "
            f"trials completos={_completed_trials(run_dir):,} · "
            f"modelos finales={final_models:,} · XAI={_xai_rows(run_dir):,}",
            flush=True,
        )


def _execute(run_dir: Path) -> Path:
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
        args=(stop, run_dir, time.monotonic()),
        daemon=True,
        name="cnn-notebook-heartbeat",
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
    parser.add_argument("--run-id", default="cnn_compact_v1")
    parser.add_argument("--node4-profile", default="v3")
    parser.add_argument("--trials-image", type=int, default=20)
    parser.add_argument("--trials-fusion", type=int, default=10)
    parser.add_argument("--trials-sbr", type=int, default=8)
    parser.add_argument("--epochs-search", type=int, default=12)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--xai-cases", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_path_component(args.run_id, "--run-id")
    validate_path_component(args.node4_profile, "--node4-profile")
    positive = {
        "--trials-image": args.trials_image,
        "--trials-fusion": args.trials_fusion,
        "--trials-sbr": args.trials_sbr,
        "--epochs-search": args.epochs_search,
        "--finalists": args.finalists,
        "--xai-cases": args.xai_cases,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"Estos parametros deben ser >= 1: {invalid}")

    run_dir = PROJECT_ROOT / "outputs" / "private_eda" / "cnn_runs" / args.run_id
    variables = {
        "DAT_CNN_RUN_ID": args.run_id,
        "DAT_CNN_NODE4_PROFILE": args.node4_profile,
        "DAT_CNN_TRIALS_IMAGE": str(args.trials_image),
        "DAT_CNN_TRIALS_FUSION": str(args.trials_fusion),
        "DAT_CNN_TRIALS_SBR": str(args.trials_sbr),
        "DAT_CNN_EPOCHS_SEARCH": str(args.epochs_search),
        "DAT_CNN_FINALISTS": str(args.finalists),
        "DAT_CNN_XAI_CASES": str(args.xai_cases),
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
    materialize_notebook()
    with RunLock(run_dir / ".run.lock"):
        os.environ["DAT_CNN_LOCK_HELD"] = "1"
        try:
            destination = _execute(run_dir)
        finally:
            os.environ.pop("DAT_CNN_LOCK_HELD", None)
    print(f"Notebook ejecutado y persistido en: {destination}", flush=True)


if __name__ == "__main__":
    main()
