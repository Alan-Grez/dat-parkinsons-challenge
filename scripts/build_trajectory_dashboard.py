"""Build a read-only Plotly snapshot for node 06 and node 07 trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.trajectory_viz import load_trajectory_snapshot, write_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-key",
        action="append",
        default=None,
        help="Repeatable key such as node06:cnn_compact_v1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "private_eda" / "trajectory_dashboard" / "latest",
    )
    args = parser.parse_args()
    snapshot = load_trajectory_snapshot(PROJECT_ROOT, args.run_key)
    dashboard = write_dashboard(snapshot, args.output_dir)
    print(
        json.dumps(
            {
                "dashboard": str(dashboard),
                "run_keys": snapshot.catalog["run_key"].tolist(),
                "trials": len(snapshot.trials),
                "history_rows": len(snapshot.histories),
                "warnings": list(snapshot.warnings),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
