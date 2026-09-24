from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cnn.config import ExperimentConfig
from modeling.cnn.io import RunLock
from modeling.cnn.pipeline import (
    prepare_experiment,
    run_final_stage,
    run_search_stage,
    validate_path_component,
)
from modeling.cnn.xai_runner import run_xai_audit


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Pipeline reanudable de CNN 2.5D/3D para DaT."
    )
    command.add_argument("--run-id", default="cnn_compact_v1")
    command.add_argument("--node4-profile", default="v3")
    command.add_argument(
        "--stage", choices=["prepare", "search", "final", "xai", "all"], default="all"
    )
    command.add_argument("--trials-image", type=int, default=20)
    command.add_argument("--trials-fusion", type=int, default=10)
    command.add_argument("--trials-sbr", type=int, default=8)
    command.add_argument("--epochs-search", type=int, default=12)
    command.add_argument("--finalists", type=int, default=3)
    command.add_argument("--xai-cases", type=int, default=24)
    command.add_argument("--device", default=None)
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    validate_path_component(args.run_id, "--run-id")
    validate_path_component(args.node4_profile, "--node4-profile")
    base = ExperimentConfig(run_id=args.run_id)
    experiment = replace(
        base,
        data=replace(base.data, node4_profile=args.node4_profile),
        train=replace(base.train, max_epochs_search=args.epochs_search),
        search=replace(
            base.search,
            trials_image=args.trials_image,
            trials_fusion=args.trials_fusion,
            trials_sbr=args.trials_sbr,
            finalists=args.finalists,
        ),
    )
    project_root = PROJECT_ROOT
    run_dir = project_root / "outputs" / "private_eda" / "cnn_runs" / args.run_id
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dry_run:
        print(f"run_dir={run_dir}")
        print(f"stage={args.stage} device={device} config_hash={experiment.config_hash}")
        return
    with RunLock(run_dir / ".run.lock"):
        print(f"[CNN] preparando run={args.run_id} device={device}")
        prepared = prepare_experiment(experiment, project_root=project_root)
        print(
            f"[CNN] cohorte={len(prepared.cohort)} "
            f"familias={prepared.cohort['acquisition_family'].nunique()}"
        )
        if args.stage in {"search", "all"}:
            search = run_search_stage(prepared, experiment, device=device)
            print(search.summary.to_string(index=False))
        if args.stage in {"final", "all"}:
            final = run_final_stage(prepared, experiment, device=device)
            print(final.metrics.to_string(index=False))
        if args.stage in {"xai", "all"}:
            xai = run_xai_audit(prepared, n_cases=args.xai_cases, device=device)
            print(xai.to_string(index=False))


if __name__ == "__main__":
    main()
