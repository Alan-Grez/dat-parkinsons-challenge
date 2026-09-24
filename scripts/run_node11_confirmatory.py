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
from modeling.node11_confirmatory import (
    ExperimentConfig,
    prepare_experiment,
    run_final_stage,
    run_search_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nodo 11: HGB regional + CNN 2.5D.")
    parser.add_argument(
        "--stage", choices=("prepare", "search", "final", "all", "status"), default="status"
    )
    parser.add_argument("--run-id", default="node11_best_shot_v1")
    parser.add_argument("--completed-trials", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    experiment = ExperimentConfig(run_id=args.run_id)
    return replace(
        experiment,
        search=replace(
            experiment.search,
            completed_trials_per_expert=max(10, int(args.completed_trials)),
        ),
    )


def show_status(experiment: ExperimentConfig) -> None:
    run_dir = SCRIPT_ROOT / "outputs" / "private_eda" / "node11_runs" / experiment.run_id
    paths = {
        "config": run_dir / "config" / "experiment_config.json",
        "common_search_folds": run_dir / "config" / "search_common_3fold.csv",
        "common_final_folds": run_dir / "config" / "final_common_5fold.csv",
        "optuna_database": run_dir / "search" / "optuna_node11.sqlite3",
        "search_summary": run_dir / "search" / "search_summary.csv",
        "final_metrics": run_dir / "final" / "final_metrics.csv",
        "deployment_manifest": run_dir / "final" / "deployment_manifest.json",
    }
    for name, path in paths.items():
        print(f"{name:24s} {'OK' if path.exists() else '--'}  {path}")
    if paths["config"].exists():
        payload = json.loads(paths["config"].read_text(encoding="utf-8"))
        print(f"config_hash: {payload.get('config_hash')}")


def main() -> None:
    args = parse_args()
    experiment = build_experiment(args)
    if args.stage == "status":
        show_status(experiment)
        return
    run_dir = SCRIPT_ROOT / "outputs" / "private_eda" / "node11_runs" / experiment.run_id
    with RunLock(run_dir / "run.lock"):
        prepared = prepare_experiment(experiment, project_root=SCRIPT_ROOT)
        print(f"Nodo 11 preparado: {len(prepared.cohort):,} pacientes.")
        if args.stage in {"search", "all"}:
            result = run_search_stage(prepared, experiment, device=args.device)
            print(result.summary.to_string(index=False))
        if args.stage in {"final", "all"}:
            result = run_final_stage(prepared, experiment, device=args.device)
            print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
