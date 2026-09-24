from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from src.predictor import Node09Predictor

WORK_ROOT = Path(os.environ.get("DAT_WORK_ROOT", "/code_execution"))
DATA_ROOT = Path(os.environ.get("DAT_DATA_ROOT", WORK_ROOT / "data"))
NIFTI_DIR = DATA_ROOT / "niftis"
FORMAT_PATH = DATA_ROOT / "submission_format.csv"
OUTPUT_PATH = Path(os.environ.get("DAT_OUTPUT_PATH", WORK_ROOT / "submission.csv"))
ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def main() -> None:
    submission = pd.read_csv(FORMAT_PATH)
    if list(submission.columns) != ["uid", "is_pathologic"]:
        raise ValueError("submission_format.csv must contain exactly uid,is_pathologic.")
    if submission["uid"].astype(str).duplicated().any():
        raise ValueError("submission_format.csv contains duplicated UIDs.")
    paths = [NIFTI_DIR / f"{uid}.nii.gz" for uid in submission["uid"].astype(str)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} requested NIfTI files.")
    predictor = Node09Predictor(ASSET_ROOT)
    probabilities = predictor.predict(paths)
    if len(probabilities) != len(submission) or not np.isfinite(probabilities).all():
        raise RuntimeError("Inference returned invalid probabilities.")
    submission["is_pathologic"] = np.clip(probabilities, 1e-6, 1 - 1e-6)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print("Inference completed and submission.csv was written.", flush=True)


if __name__ == "__main__":
    main()
