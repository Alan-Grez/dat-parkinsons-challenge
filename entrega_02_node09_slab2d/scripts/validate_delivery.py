from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPECTED_CANDIDATE = "555b81376733bb67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_static(project_root: Path) -> dict:
    delivery = project_root / "entrega_02_node09_slab2d"
    source = delivery / "submission_src"
    archive_path = delivery / "submission" / "submission.zip"
    build = json.loads(
        (delivery / "validation" / "build_manifest.json").read_text(encoding="utf-8")
    )
    if sha256(archive_path) != build["submission_zip_sha256"]:
        raise RuntimeError("submission.zip differs from its build manifest.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if "main.py" not in names:
            raise RuntimeError("main.py is not at the ZIP root.")
        if any(name.startswith("submission_src/") for name in names):
            raise RuntimeError("ZIP contains an unwanted wrapper directory.")
        suspicious = [
            name
            for name in names
            if name.lower().endswith((".nii", ".nii.gz", ".csv"))
        ]
        if suspicious:
            raise RuntimeError(f"Unexpected data-like files in ZIP: {suspicious}")
    manifest = json.loads((source / "assets" / "manifest.json").read_text())
    if manifest["candidate_id"] != EXPECTED_CANDIDATE:
        raise RuntimeError("Packaged candidate is not the frozen node09 winner.")
    if manifest["n_fold_models"] != 5 or len(manifest["folds"]) != 5:
        raise RuntimeError("Submission does not contain five fold models.")
    if manifest["acquisition_family_used_as_predictor"]:
        raise RuntimeError("Acquisition family must not be a predictor.")
    for fold in manifest["folds"]:
        for key in ("checkpoint_file", "template_file"):
            if not (source / "assets" / fold[key]).exists():
                raise FileNotFoundError(fold[key])
    return {
        "archive_sha256": sha256(archive_path),
        "archive_files": len(names),
        "archive_bytes": archive_path.stat().st_size,
        "candidate_id": manifest["candidate_id"],
        "fold_models": 5,
        "temperature": manifest["temperature"],
        "cv5_cross_calibrated_log_loss": manifest["cross_validated_log_loss"],
    }


def validate_model_equivalence(project_root: Path) -> dict:
    source = project_root / "entrega_02_node09_slab2d" / "submission_src"
    sys.path.insert(0, str(source))
    from modeling.dat_spect_v2.config import ModelConfig
    from modeling.dat_spect_v2.models import MultiTaskDaTClassifier

    maximum = 0.0
    for fold in range(5):
        slim = torch.load(
            source / "assets" / f"fold_{fold}.pt", map_location="cpu", weights_only=False
        )
        original = torch.load(
            project_root
            / "outputs"
            / "private_eda"
            / "node09_runs"
            / "node09_fine_v1"
            / "final"
            / "cv5"
            / EXPECTED_CANDIDATE
            / f"fold_{fold}"
            / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        first = MultiTaskDaTClassifier(ModelConfig(**slim["model_config"]))
        second = MultiTaskDaTClassifier(
            ModelConfig(**original["checkpoint_contract"]["model"])
        )
        first.load_state_dict(slim["model_state"], strict=True)
        second.load_state_dict(original["model_state"], strict=True)
        first.eval()
        second.eval()
        generator = torch.Generator().manual_seed(9000 + fold)
        image = torch.rand((2, 1, 72, 72), generator=generator)
        regional = torch.rand((2, 32), generator=generator)
        sbr = torch.rand((2, 11), generator=generator)
        valid = torch.tensor([0.0, 1.0])
        with torch.inference_mode():
            a = first(image, radiomics=regional, sbr=sbr, sbr_valid=valid)["logit"]
            b = second(image, radiomics=regional, sbr=sbr, sbr_valid=valid)["logit"]
        maximum = max(maximum, float(torch.max(torch.abs(a - b))))
    if maximum > 1e-6:
        raise RuntimeError(f"Packaged architecture changed logits by {maximum}.")
    return {"folds": 5, "source_model_max_abs_logit_difference": maximum}


def run_archive_smoke(project_root: Path, data_dir: Path) -> dict:
    delivery = project_root / "entrega_02_node09_slab2d"
    archive_path = delivery / "submission" / "submission.zip"
    with tempfile.TemporaryDirectory(prefix="dat_node09_submission_") as temporary:
        extracted = Path(temporary) / "src"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        output = Path(temporary) / "submission.csv"
        environment = os.environ.copy()
        environment["DAT_DATA_ROOT"] = str(data_dir.resolve())
        environment["DAT_OUTPUT_PATH"] = str(output)
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(extracted / "main.py")],
            cwd=extracted,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(
                "Archive smoke failed:\n" + completed.stdout[-1000:] + completed.stderr[-3000:]
            )
        expected = pd.read_csv(data_dir / "submission_format.csv")
        actual = pd.read_csv(output)
        if list(actual.columns) != ["uid", "is_pathologic"]:
            raise RuntimeError("Generated submission.csv has incorrect columns.")
        if actual["uid"].astype(str).tolist() != expected["uid"].astype(str).tolist():
            raise RuntimeError("Generated submission.csv does not preserve UID order.")
        probabilities = actual["is_pathologic"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not (
            (probabilities >= 0).all() and (probabilities <= 1).all()
        ):
            raise RuntimeError("Generated probabilities are invalid.")
        return {
            "rows": len(actual),
            "columns": list(actual.columns),
            "probabilities_valid": True,
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "elapsed_seconds": elapsed,
            "seconds_per_case_including_startup": elapsed / max(len(actual), 1),
            "stdout_lines": len(completed.stdout.splitlines()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else project_root / "official-runtime" / "data-demo"
    )
    report = {
        "static": validate_static(project_root),
        "model_equivalence": validate_model_equivalence(project_root),
        "official_demo_archive_smoke": run_archive_smoke(project_root, data_dir),
    }
    destination = (
        project_root
        / "entrega_02_node09_slab2d"
        / "validation"
        / "validation_report.json"
    )
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
