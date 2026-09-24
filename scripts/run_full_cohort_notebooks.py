"""Execute the full-cohort EDA notebooks sequentially without overwriting them."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "private_eda"
NODE4_PROFILE = "v3"
NODE4_OUTPUT_DIR = PRIVATE_OUTPUT_DIR / f"full_cohort_{NODE4_PROFILE}"
DEFAULT_NODE5_RUN_ID = "optuna_radiomics_v1"
NOTEBOOKS = (
    PROJECT_ROOT / "notebooks" / "03_eda_cohorte_3d_dat.ipynb",
    PROJECT_ROOT / "notebooks" / "04_registro_biomarcadores_radiomica.ipynb",
    PROJECT_ROOT / "notebooks" / "05_embeddings_outliers_3d.ipynb",
)
RUN_LOCK_PATH = PRIVATE_OUTPUT_DIR / "run_full_cohort_notebooks.lock"
REGISTRATION_PROGRESS_PATH = NODE4_OUTPUT_DIR / "registration_qc_full.csv"
HEARTBEAT_SECONDS = 60
NODE5_RUN_ID = DEFAULT_NODE5_RUN_ID
OPTUNA_TRIALS_PER_MODEL = 50
OPTUNA_MODEL_COUNT = 3
EMBEDDING_RUN_ID = "umap_tsne_v1"
EMBEDDING_TRIALS_PER_METHOD = 30
EMBEDDING_METHOD_COUNT = 2


def node5_output_dir() -> Path:
    return PRIVATE_OUTPUT_DIR / "node5_runs" / NODE5_RUN_ID


def oof_progress_path() -> Path:
    return node5_output_dir() / "optuna_models" / "oof_fold_checkpoint.csv"


def embedding_output_dir() -> Path:
    return node5_output_dir() / "embedding_runs" / EMBEDDING_RUN_ID


@contextmanager
def exclusive_run_lock() -> Iterator[None]:
    """Prevent concurrent notebook runners from sharing GPU/checkpoint files."""
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = RUN_LOCK_PATH.open("a+", encoding="utf-8")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write("\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(
            "Ya hay otro run_full_cohort_notebooks.py activo. "
            "No inicies dos ejecuciones sobre los mismos checkpoints."
        ) from error

    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            RUN_LOCK_PATH.unlink(missing_ok=True)


def executed_path(notebook_path: Path) -> Path:
    """Return the private path used for the notebook with embedded outputs."""
    output_dir = node5_output_dir() if notebook_path.name.startswith("05_") else NODE4_OUTPUT_DIR
    return output_dir / "executed_notebooks" / f"{notebook_path.stem}.executed.ipynb"


def persist(notebook: NotebookNode, destination: Path) -> None:
    """Write atomically so an interruption cannot leave a corrupt notebook."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary_path)
    temporary_path.replace(destination)


def completed_csv_rows(path: Path) -> int:
    """Count durable rows without loading a private table into memory."""
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def completed_optuna_trials() -> int:
    """Read atomic per-model progress files without touching Optuna's live journal."""
    studies_dir = node5_output_dir() / "optuna_models" / "studies"
    completed = 0
    for path in studies_dir.glob("*_progress.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed += int(payload.get("completed_trials", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return completed


def completed_embedding_trials() -> int:
    """Read resumable t-SNE/UMAP progress without opening live journals."""
    studies_dir = embedding_output_dir() / "studies"
    completed = 0
    for path in studies_dir.glob("*_progress.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed += int(payload.get("completed_trials", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return completed


def report_progress(stop_event: threading.Event, started_at: float) -> None:
    """Emit a heartbeat while nbclient is blocked inside a long notebook cell."""
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed_minutes = (time.monotonic() - started_at) / 60
        registrations = completed_csv_rows(REGISTRATION_PROGRESS_PATH)
        oof_rows = completed_csv_rows(oof_progress_path())
        optuna_trials = completed_optuna_trials()
        optuna_target = OPTUNA_TRIALS_PER_MODEL * OPTUNA_MODEL_COUNT
        embedding_trials = completed_embedding_trials()
        embedding_target = EMBEDDING_TRIALS_PER_METHOD * EMBEDDING_METHOD_COUNT
        print(
            f"[heartbeat] {elapsed_minutes:.1f} min · "
            f"registros confirmados: {registrations:,} · "
            f"Optuna: {optuna_trials:,}/{optuna_target:,} ensayos · "
            f"embeddings: {embedding_trials:,}/{embedding_target:,} ensayos · "
            f"predicciones OOF confirmadas: {oof_rows:,}",
            flush=True,
        )


def execute(notebook_path: Path) -> Path:
    """Run one notebook and save a private copy containing its cell outputs."""
    notebook = nbformat.read(notebook_path, as_version=4)
    kernel_name = notebook.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    destination = executed_path(notebook_path)
    stop_event = threading.Event()
    started_at = time.monotonic()
    heartbeat = threading.Thread(
        target=report_progress,
        args=(stop_event, started_at),
        name="notebook-progress-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        client.execute()
    except BaseException:
        partial_path = destination.with_name(f"{notebook_path.stem}.partial.ipynb")
        persist(notebook, partial_path)
        print(f"Partial notebook saved: {partial_path}", flush=True)
        raise
    finally:
        stop_event.set()
        heartbeat.join(timeout=2)
    persist(notebook, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the execution order without starting a kernel.",
    )
    parser.add_argument(
        "--start-at",
        choices=("03", "04", "05"),
        default="03",
        help="Start at this notebook while preserving the remaining order.",
    )
    parser.add_argument(
        "--node5-run-id",
        default=DEFAULT_NODE5_RUN_ID,
        help=(
            "Independent notebook-05 experiment directory below "
            "outputs/private_eda/node5_runs/."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Completed Optuna trials requested per supervised classifier.",
    )
    parser.add_argument(
        "--embedding-run-id",
        default="umap_tsne_v1",
        help="Independent resumable t-SNE/UMAP experiment below the node-05 run.",
    )
    parser.add_argument(
        "--embedding-trials",
        type=int,
        default=30,
        help="Completed Optuna trials requested for each of t-SNE and UMAP.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.node5_run_id):
        raise ValueError("--node5-run-id only accepts letters, numbers, dot, _ and -.")
    if args.optuna_trials < 1:
        raise ValueError("--optuna-trials must be >= 1.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.embedding_run_id):
        raise ValueError("--embedding-run-id only accepts letters, numbers, dot, _ and -.")
    if args.embedding_trials < 1:
        raise ValueError("--embedding-trials must be >= 1.")
    global NODE5_RUN_ID, OPTUNA_TRIALS_PER_MODEL
    global EMBEDDING_RUN_ID, EMBEDDING_TRIALS_PER_METHOD
    NODE5_RUN_ID = args.node5_run_id
    OPTUNA_TRIALS_PER_MODEL = args.optuna_trials
    EMBEDDING_RUN_ID = args.embedding_run_id
    EMBEDDING_TRIALS_PER_METHOD = args.embedding_trials
    os.environ["DAT_NODE5_RUN_ID"] = NODE5_RUN_ID
    os.environ["DAT_OPTUNA_TRIALS"] = str(args.optuna_trials)
    os.environ["DAT_EMBEDDING_RUN_ID"] = EMBEDDING_RUN_ID
    os.environ["DAT_EMBEDDING_TRIALS"] = str(args.embedding_trials)

    missing = [path for path in NOTEBOOKS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing notebooks: {missing}")

    start_index = next(
        index
        for index, notebook_path in enumerate(NOTEBOOKS)
        if notebook_path.name.startswith(f"{args.start_at}_")
    )
    selected_notebooks = NOTEBOOKS[start_index:]
    if args.dry_run:
        for position, notebook_path in enumerate(selected_notebooks, start=1):
            print(f"[{position}/{len(selected_notebooks)}] {notebook_path.name}", flush=True)
    else:
        with exclusive_run_lock():
            print(f"Exclusive run lock: {RUN_LOCK_PATH}", flush=True)
            for position, notebook_path in enumerate(selected_notebooks, start=1):
                print(
                    f"[{position}/{len(selected_notebooks)}] {notebook_path.name}",
                    flush=True,
                )
                destination = execute(notebook_path)
                print(f"Completed: {notebook_path.name}", flush=True)
                print(f"Saved with outputs: {destination}", flush=True)

    print("Dry run completed." if args.dry_run else "Selected notebooks completed.")


if __name__ == "__main__":
    main()
