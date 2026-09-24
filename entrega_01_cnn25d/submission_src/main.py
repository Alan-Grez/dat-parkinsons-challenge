from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import DaTEnsemblePredictor


SOURCE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DAT_DATA_DIR", "/code_execution/data"))
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path(os.environ.get("DAT_OUTPUT_PATH", "submission.csv"))


def main() -> None:
    torch.set_num_threads(max(1, min(24, os.cpu_count() or 1)))
    submission = pd.read_csv(SUBMISSION_FORMAT_PATH)
    if list(submission.columns) != ["uid", "is_pathologic"]:
        raise ValueError("submission_format.csv must contain exactly uid,is_pathologic.")
    if submission["uid"].astype(str).duplicated().any():
        raise ValueError("submission_format.csv contains duplicated UIDs.")
    paths = [NIFTI_DIR / f"{uid}.nii.gz" for uid in submission["uid"].astype(str)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("At least one required NIfTI file is missing.")

    print("Starting DaT inference.", flush=True)
    predictor = DaTEnsemblePredictor(SOURCE_ROOT / "assets")
    probabilities = predictor.predict_paths(paths)
    if len(probabilities) != len(submission) or not np.isfinite(probabilities).all():
        raise RuntimeError("Inference did not produce one finite probability per examination.")
    submission["is_pathologic"] = probabilities.astype(float)
    WRITE_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(WRITE_SUBMISSION_PATH, index=False)
    print("Inference completed and submission.csv was written.", flush=True)


if __name__ == "__main__":
    main()
