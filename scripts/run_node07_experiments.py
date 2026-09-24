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
from modeling.dat_spect_v2 import (
    ExperimentConfig,
    prepare_experiment,
    run_final_stage,
    run_search_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nodo 07 DaT-SPECT reanudable.")
    parser.add_argument(
        "--stage",
        choices=("prepare", "search", "final", "all", "status"),
        default="prepare",
    )
    parser.add_argument("--run-id", default="dat_spect_slab_v4")
    parser.add_argument("--node4-profile", default="v3")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Completa un trial por estudio; luego puede continuarse sin cambiar run_id.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    experiment = ExperimentConfig(run_id=args.run_id)
    experiment = replace(
        experiment,
        data=replace(experiment.data, node4_profile=args.node4_profile),
    )
    if args.quick:
        search_values = {
            name: 1
            for name in experiment.search.__dataclass_fields__
            if name.startswith("trials_")
        }
        experiment = replace(
            experiment,
            search=replace(experiment.search, **search_values),
        )
    return experiment


def show_status(root: Path, experiment: ExperimentConfig) -> None:
    run_dir = root / "outputs" / "private_eda" / "node07_runs" / experiment.run_id
    paths = {
        "config": run_dir / "config" / "experiment_config.json",
        "prepared_cohort": run_dir / "config" / "prepared_cohort.csv",
        "search_summary": run_dir / "search" / "search_summary.csv",
        "final_metrics": run_dir / "final" / "final_metrics.csv",
        "deployment_manifest": run_dir / "final" / "deployment_manifest.json",
    }
    for name, path in paths.items():
        print(f"{name:22s} {'OK' if path.exists() else '--'}  {path}")
    if paths["config"].exists():
        payload = json.loads(paths["config"].read_text(encoding="utf-8"))
        print(f"config_hash: {payload.get('config_hash')}")


def main() -> None:
    args = parse_args()
    root = project_root()
    experiment = build_config(args)
    if args.stage == "status":
        show_status(root, experiment)
        return
    run_dir = root / "outputs" / "private_eda" / "node07_runs" / experiment.run_id
    with RunLock(run_dir / "run.lock"):
        prepared = prepare_experiment(experiment, project_root=root)
        print(
            f"Preparado: {len(prepared.cohort):,} casos; "
            f"{prepared.cohort['acquisition_family'].nunique():,} familias."
        )
        if args.stage in {"search", "all"}:
            result = run_search_stage(prepared, experiment, device=args.device)
            print(result.summary.to_string(index=False))
        if args.stage in {"final", "all"}:
            result = run_final_stage(prepared, experiment, device=args.device)
            print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
