"""Execute notebook 08 and persist a private copy with Plotly outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_trajectory_notebook import NOTEBOOK_PATH, main as materialize


def _atomic_notebook(notebook: NotebookNode, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    nbformat.write(notebook, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-key",
        action="append",
        default=None,
        help="Repeatable key such as node06:cnn_compact_v1.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    materialize()
    if args.run_key:
        os.environ["DAT_TRAJECTORY_RUNS"] = ",".join(args.run_key)
    if args.dry_run:
        print(NOTEBOOK_PATH)
        return
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    output_dir = PROJECT_ROOT / "outputs" / "private_eda" / "trajectory_dashboard" / "latest"
    destination = output_dir / f"{NOTEBOOK_PATH.stem}.executed.ipynb"
    try:
        client.execute()
    except BaseException:
        partial = output_dir / f"{NOTEBOOK_PATH.stem}.partial.ipynb"
        _atomic_notebook(notebook, partial)
        print(f"Notebook parcial guardado en: {partial}", flush=True)
        raise
    _atomic_notebook(notebook, destination)
    print(f"Notebook ejecutado y persistido en: {destination}", flush=True)


if __name__ == "__main__":
    main()
