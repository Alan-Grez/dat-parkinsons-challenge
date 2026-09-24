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
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static(delivery: Path) -> dict:
    source = delivery / "submission_src"
    archive_path = delivery / "submission" / "submission.zip"
    build_manifest_path = delivery / "validation" / "build_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    if sha256(archive_path) != build_manifest["submission_zip_sha256"]:
        raise RuntimeError("submission.zip hash differs from build_manifest.json.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if "main.py" not in names:
            raise RuntimeError("main.py is not at the ZIP root.")
        if any(name.startswith("submission_src/") for name in names):
            raise RuntimeError("ZIP contains an unwanted submission_src wrapper directory.")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise RuntimeError("ZIP contains an unsafe path.")
        bad = [
            name
            for name in names
            if name.endswith((".nii", ".nii.gz", ".csv"))
        ]
        if bad:
            raise RuntimeError(f"Unexpected data-like files inside submission.zip: {bad}")
    assets_manifest = json.loads(
        (source / "assets" / "manifest.json").read_text(encoding="utf-8")
    )
    if assets_manifest["candidate_id"] != "43765c3f64a985b1":
        raise RuntimeError("The package does not contain the selected primary candidate.")
    if assets_manifest["architecture"] != "2.5d" or assets_manifest["feature_variant"] != "image_only":
        raise RuntimeError("Unexpected primary architecture or feature variant.")
    with np.load(source / "assets" / "registration_template.npz", allow_pickle=False) as template:
        shape = tuple(template["fixed"].shape)
        if shape != tuple(template["brain_mask"].shape):
            raise RuntimeError("Template and brain mask shapes differ.")
        if not np.isfinite(template["fixed"]).all():
            raise RuntimeError("Registration template contains non-finite values.")
    sys.path.insert(0, str(source))
    from src import DaTEnsemblePredictor

    predictor = DaTEnsemblePredictor(source / "assets", device="cpu")
    slices = int(assets_manifest["data_config"]["slices_per_view"])
    view = int(assets_manifest["data_config"]["view_size_2d"])
    dummy = torch.zeros((2, 3, slices, view, view), dtype=torch.float32)
    probabilities = predictor.predict_batch(dummy)
    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise RuntimeError("CPU model smoke test failed.")
    project_root = delivery.parent
    sys.path.insert(0, str(project_root))
    from modeling.cnn.config import ModelConfig as TrainingModelConfig
    from modeling.cnn.models import HybridDaTClassifier as TrainingModel
    from src.model import HybridDaTClassifier as SubmissionModel
    from src.model import ModelConfig as SubmissionModelConfig

    fold_state = torch.load(
        source / "assets" / assets_manifest["fold_files"][0],
        map_location="cpu",
        weights_only=True,
    )
    training_model = TrainingModel(
        TrainingModelConfig(**fold_state["model_config"])
    ).eval()
    submission_model = SubmissionModel(
        SubmissionModelConfig.from_dict(fold_state["model_config"])
    ).eval()
    training_model.load_state_dict(fold_state["model_state"], strict=True)
    submission_model.load_state_dict(fold_state["model_state"], strict=True)
    generator = torch.Generator().manual_seed(20260829)
    equivalence_input = torch.randn(
        (2, 3, slices, view, view), generator=generator, dtype=torch.float32
    )
    with torch.inference_mode():
        training_logits = training_model(equivalence_input)
        submission_logits = submission_model(equivalence_input)
    maximum_difference = float(
        torch.max(torch.abs(training_logits - submission_logits)).item()
    )
    if maximum_difference > 1e-7:
        raise RuntimeError(
            f"Packaged model differs from the training architecture: {maximum_difference}."
        )
    return {
        "archive_sha256": sha256(archive_path),
        "archive_files": len(names),
        "archive_bytes": archive_path.stat().st_size,
        "candidate_id": assets_manifest["candidate_id"],
        "fold_models": len(predictor.models),
        "cpu_model_smoke_probabilities": probabilities.tolist(),
        "source_model_max_abs_logit_difference": maximum_difference,
        "template_shape_zyx": list(shape),
    }


def validate_end_to_end(delivery: Path, data_dir: Path) -> dict:
    source = delivery / "submission_src"
    with tempfile.TemporaryDirectory(prefix="dat_submission_output_") as temporary:
        output = Path(temporary) / "submission.csv"
        environment = os.environ.copy()
        environment["DAT_DATA_DIR"] = str(data_dir.resolve())
        environment["DAT_OUTPUT_PATH"] = str(output.resolve())
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=source,
            env=environment,
            text=True,
            capture_output=True,
            timeout=3600,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "End-to-end inference failed:\n"
                + completed.stdout[-2000:]
                + completed.stderr[-4000:]
            )
        import pandas as pd

        expected = pd.read_csv(data_dir / "submission_format.csv")
        result = pd.read_csv(output)
        if list(result.columns) != ["uid", "is_pathologic"]:
            raise RuntimeError("Generated submission.csv has incorrect columns.")
        if result["uid"].astype(str).tolist() != expected["uid"].astype(str).tolist():
            raise RuntimeError("Generated submission.csv does not preserve UID order.")
        probabilities = result["is_pathologic"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not ((probabilities > 0) & (probabilities < 1)).all():
            raise RuntimeError("Generated probabilities are not finite and strictly inside (0,1).")
        elapsed_seconds = time.perf_counter() - started
        return {
            "rows": len(result),
            "columns": result.columns.tolist(),
            "probabilities_valid": True,
            "stdout_lines": len(completed.stdout.splitlines()),
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_case_including_startup": elapsed_seconds / max(len(result), 1),
        }


def validate_local_archive_smoke(delivery: Path, n_cases: int) -> dict:
    if n_cases < 1:
        raise ValueError("--local-smoke-cases must be at least 1.")
    project_root = delivery.parent
    archive_path = project_root / "data" / "raw" / "niftis.zip"
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix="dat_submission_smoke_") as temporary:
        data_dir = Path(temporary) / "data"
        nifti_dir = data_dir / "niftis"
        nifti_dir.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".nii.gz") and not name.endswith("/")
            )[:n_cases]
            if len(members) != n_cases:
                raise RuntimeError(
                    f"Requested {n_cases} smoke cases but found {len(members)} in the archive."
                )
            uids = []
            for member in members:
                filename = Path(member).name
                uid = filename[:-7]
                uids.append(uid)
                with archive.open(member) as source, (nifti_dir / filename).open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
        pd.DataFrame({"uid": uids, "is_pathologic": np.nan}).to_csv(
            data_dir / "submission_format.csv", index=False
        )
        report = validate_end_to_end(delivery, data_dir)
        source = delivery / "submission_src"
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(project_root))
        from modeling.cnn.data import volume_to_triplanar
        from src import DaTEnsemblePredictor

        cache_manifest_path = (
            project_root
            / "outputs"
            / "private_eda"
            / "cnn_cache"
            / "v3"
            / "4b7d20cc4552de54"
            / "image_cache_manifest.csv"
        )
        cache_manifest = pd.read_csv(cache_manifest_path)
        cache_by_uid = cache_manifest.set_index(cache_manifest["uid"].astype(str))[
            "cache_path"
        ].to_dict()
        predictor = DaTEnsemblePredictor(source / "assets")
        correlations: list[float] = []
        mean_absolute_errors: list[float] = []
        for uid in uids:
            if uid not in cache_by_uid:
                raise RuntimeError("A smoke case is absent from the training image cache.")
            packaged_view = predictor.preprocess(nifti_dir / f"{uid}.nii.gz").numpy()
            with np.load(Path(cache_by_uid[uid]), allow_pickle=False) as payload:
                cached_volume = torch.from_numpy(
                    payload["volume"].astype(np.float32)
                )[None]
            cached_view = volume_to_triplanar(
                cached_volume,
                predictor.image_config.slices_per_view,
                predictor.image_config.view_size_2d,
            ).numpy()
            correlations.append(
                float(np.corrcoef(packaged_view.ravel(), cached_view.ravel())[0, 1])
            )
            mean_absolute_errors.append(
                float(np.mean(np.abs(packaged_view - cached_view)))
            )
        if min(correlations) < 0.995 or max(mean_absolute_errors) > 0.02:
            raise RuntimeError(
                "Packaged preprocessing does not reproduce the training image cache closely enough."
            )
        report["source"] = "temporary_local_archive_smoke"
        report["preprocessing_regression"] = {
            "cases": len(correlations),
            "minimum_pearson_correlation": min(correlations),
            "maximum_mean_absolute_error": max(mean_absolute_errors),
        }
        return report


def main() -> None:
    delivery = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--local-smoke-cases",
        type=int,
        default=0,
        help="Temporarily extract N local raw cases and run the packaged main.py end to end.",
    )
    args = parser.parse_args()
    report = {"static": validate_static(delivery)}
    if args.data_dir is not None:
        report["end_to_end"] = validate_end_to_end(delivery, args.data_dir)
    if args.local_smoke_cases:
        report["local_archive_smoke"] = validate_local_archive_smoke(
            delivery, args.local_smoke_cases
        )
    destination = delivery / "validation" / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
