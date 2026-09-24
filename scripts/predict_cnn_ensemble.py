"""Apply the frozen fold ensemble to a prepared private cohort manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cnn import DataConfig, predict_fold_ensemble
from modeling.cnn.pipeline import validate_path_component


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cnn_compact_v1")
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=None,
        help="CSV with cache_path and the same derived-feature schema used in training.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    validate_path_component(args.run_id, "--run-id")
    run_dir = PROJECT_ROOT / "outputs" / "private_eda" / "cnn_runs" / args.run_id
    config_path = run_dir / "config" / "experiment_config.json"
    manifest_path = run_dir / "final" / "primary_ensemble_manifest.json"
    cohort_path = args.cohort_manifest or (
        run_dir / "config" / "prepared_cohort_manifest.csv"
    )
    output_path = args.output or (run_dir / "final" / "ensemble_predictions.csv")
    for required in (config_path, manifest_path, cohort_path):
        if not required.exists():
            raise FileNotFoundError(required)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort = pd.read_csv(cohort_path)
    result = predict_fold_ensemble(
        cohort,
        manifest_path,
        data_config=DataConfig(**config["data"]),
        output_path=output_path,
        device=args.device,
    )
    print(result.predictions.head().to_string(index=False))
    print(f"Predicciones guardadas en: {output_path}")


if __name__ == "__main__":
    main()
