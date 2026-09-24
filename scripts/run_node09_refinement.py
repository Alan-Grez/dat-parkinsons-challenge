from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from modeling.cnn.io import RunLock
from modeling.node09_refinement import (
    RefinementExperiment,
    prepare_refinement,
    run_final_stage,
    run_search_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nodo 09: refinamiento de tres finalistas.")
    parser.add_argument(
        "--stage", choices=("prepare", "search", "final", "all", "status"), default="status"
    )
    parser.add_argument("--run-id", default="node09_fine_v1")
    parser.add_argument("--source-node07-run-id", default="dat_spect_slab_v4")
    parser.add_argument("--node4-profile", default="v3")
    parser.add_argument("--trials-per-model", type=int, default=18)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_experiment(args: argparse.Namespace) -> RefinementExperiment:
    experiment = RefinementExperiment(
        run_id=args.run_id,
        source_node07_run_id=args.source_node07_run_id,
    )
    return replace(
        experiment,
        data=replace(experiment.data, node4_profile=args.node4_profile),
        search=replace(experiment.search, trials_per_model=args.trials_per_model),
    )


def show_status(root: Path, experiment: RefinementExperiment) -> None:
    source = (
        root
        / "outputs"
        / "private_eda"
        / "node07_runs"
        / experiment.source_node07_run_id
    )
    run_dir = root / "outputs" / "private_eda" / "node09_runs" / experiment.run_id
    paths = {
        "node07_final_metrics": source / "final" / "final_metrics.csv",
        "selected_references": run_dir / "config" / "selected_references.csv",
        "optuna_database": run_dir / "search" / "optuna_node09.sqlite3",
        "search_summary": run_dir / "search" / "search_summary.csv",
        "final_metrics": run_dir / "final" / "final_metrics.csv",
        "deployment_manifest": run_dir / "final" / "deployment_manifest.json",
    }
    for name, path in paths.items():
        print(f"{name:24s} {'OK' if path.exists() else '--'}  {path}")
    config_path = run_dir / "config" / "experiment_config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        print(f"config_hash: {payload.get('config_hash')}")


def main() -> None:
    args = parse_args()
    root = SCRIPT_ROOT
    experiment = build_experiment(args)
    if args.stage == "status":
        show_status(root, experiment)
        return
    run_dir = root / "outputs" / "private_eda" / "node09_runs" / experiment.run_id
    with RunLock(run_dir / "run.lock"):
        prepared = prepare_refinement(experiment, project_root=root)
        print(
            f"Preparado: {len(prepared.cohort):,} pacientes; "
            f"{len(prepared.references)} ramas fuente."
        )
        if args.stage in {"search", "all"}:
            result = run_search_stage(prepared, experiment, device=args.device)
            print(result.summary.to_string(index=False))
        if args.stage in {"final", "all"}:
            result = run_final_stage(prepared, experiment, device=args.device)
            print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()

