"""Execute the three EDA notebooks end-to-end on synthetic NIfTI volumes only."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import nbformat
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    PROJECT_ROOT / "notebooks" / "03_eda_cohorte_3d_dat.ipynb",
    PROJECT_ROOT / "notebooks" / "04_registro_biomarcadores_radiomica.ipynb",
    PROJECT_ROOT / "notebooks" / "05_embeddings_outliers_3d.ipynb",
]


def synthetic_volume(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    is_pathologic: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = [
        (np.arange(size, dtype=np.float32) - (size - 1) / 2) * step
        for size, step in zip(shape, spacing)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    broad_background = 90 * np.exp(-(x**2 + y**2 + z**2) / (2 * 34**2))
    right = 900 * np.exp(-((x + 9) ** 2 / 20 + y**2 / 34 + z**2 / 22))
    left_scale = 0.42 if is_pathologic else 0.96
    left = 900 * left_scale * np.exp(-((x - 9) ** 2 / 20 + y**2 / 34 + z**2 / 22))
    posterior_loss = 1 - (0.30 * is_pathologic / (1 + np.exp(-(y - 1) / 2)))
    expected = np.clip((broad_background + right + left) * posterior_loss, 0, None)
    noisy = rng.poisson(expected).astype(np.uint16)
    return noisy


def build_synthetic_project(root: Path, scans: int = 12) -> None:
    raw_dir = root / "data" / "raw"
    staging_dir = root / "staging"
    raw_dir.mkdir(parents=True)
    staging_dir.mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "modeling", root / "modeling")
    archive_path = raw_dir / "niftis.zip"
    rows: list[dict[str, object]] = []
    geometries = [
        ((32, 32, 32), (2.5, 2.5, 2.5)),
        ((30, 34, 32), (2.7, 2.7, 2.7)),
        ((34, 30, 30), (2.4, 2.4, 2.4)),
    ]

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(scans):
            uid = f"synthetic_{index:03d}"
            is_pathologic = index % 2
            shape, spacing = geometries[index % len(geometries)]
            volume = synthetic_volume(shape, spacing, is_pathologic, seed=1000 + index)
            affine = np.eye(4, dtype=float)
            affine[:3, :3] = np.diag(spacing)
            affine[:3, 3] = -0.5 * np.asarray(shape) * np.asarray(spacing)
            nifti_path = staging_dir / f"{uid}.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine), nifti_path)
            archive.write(nifti_path, arcname=f"niftis/{uid}.nii.gz")
            rows.append({"uid": uid, "is_pathologic": float(is_pathologic)})

    pd.DataFrame(rows).to_csv(raw_dir / "train_labels.csv", index=False)


def execute_notebook(
    notebook_path: Path,
    working_directory: Path,
    interrupt_after_registrations: int | None = None,
) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    if interrupt_after_registrations is not None:
        injected = False
        marker = "    try:\n        if uid == REFERENCE_UID:\n"
        replacement = (
            "    try:\n"
            + "        if len(mapped_uids) >= "
            + str(interrupt_after_registrations)
            + ":\n"
            + "            raise KeyboardInterrupt('synthetic interruption')\n"
            + "        if uid == REFERENCE_UID:\n"
        )
        for cell in notebook.cells:
            if cell.cell_type == "code" and "Full-cohort streaming" in cell.source:
                if marker not in cell.source:
                    raise AssertionError("Could not inject synthetic interruption.")
                cell.source = cell.source.replace(marker, replacement, 1)
                injected = True
                break
        if not injected:
            raise AssertionError("Streaming cell not found for interruption test.")
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
        allow_errors=False,
    )
    client.execute()


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["DAT_NODE5_RUN_ID"] = "synthetic_test"
    os.environ["DAT_OPTUNA_TRIALS"] = "2"
    with tempfile.TemporaryDirectory(prefix="dat_eda_synthetic_") as temporary_directory:
        synthetic_root = Path(temporary_directory)
        build_synthetic_project(synthetic_root)
        for notebook_path in NOTEBOOKS:
            print(f"Executing {notebook_path.name} ...", flush=True)
            if notebook_path.name.startswith("04_"):
                try:
                    execute_notebook(
                        notebook_path,
                        synthetic_root,
                        interrupt_after_registrations=5,
                    )
                except CellExecutionError:
                    partial_features = pd.read_csv(
                        synthetic_root
                        / "outputs"
                        / "private_eda"
                        / "full_cohort_v3"
                        / "dat_radiomics_features_full.csv"
                    )
                    if len(partial_features) != 5:
                        raise AssertionError(
                            "Ctrl+C checkpoint did not preserve five completed registrations."
                        )
                    print("Intentional node 04 interruption checkpointed correctly.", flush=True)
                else:
                    raise AssertionError("Intentional node 04 interruption did not occur.")
                execute_notebook(notebook_path, synthetic_root)
            else:
                execute_notebook(notebook_path, synthetic_root)
            if notebook_path.name.startswith("04_"):
                print("Re-executing 04 to validate Windows-safe resume ...", flush=True)
                execute_notebook(notebook_path, synthetic_root)
            if notebook_path.name.startswith("05_"):
                node5_output_dir = (
                    synthetic_root
                    / "outputs"
                    / "private_eda"
                    / "node5_runs"
                    / "synthetic_test"
                )
                optuna_output_dir = node5_output_dir / "optuna_models"
                checkpoint_path = optuna_output_dir / "oof_fold_checkpoint.csv"
                checkpoint = pd.read_csv(checkpoint_path)
                first = checkpoint.iloc[0]
                incomplete = checkpoint.loc[
                    ~(
                        (checkpoint["model"] == first["model"])
                        & (checkpoint["fold"] == first["fold"])
                    )
                ]
                incomplete.to_csv(checkpoint_path, index=False)
                for filename in [
                    "optuna_oof_predictions.csv",
                    "optuna_oof_metrics.csv",
                ]:
                    (optuna_output_dir / filename).unlink(missing_ok=True)
                for filename in ["label_audit_full.csv", "protocol_diagnostics_full.csv"]:
                    (node5_output_dir / filename).unlink(missing_ok=True)
                print("Re-executing 05 from an intentionally incomplete fold checkpoint ...", flush=True)
                execute_notebook(notebook_path, synthetic_root)

        output_dir = synthetic_root / "outputs" / "private_eda"
        full_output_dir = output_dir / "full_cohort_v3"
        node5_output_dir = output_dir / "node5_runs" / "synthetic_test"
        optuna_output_dir = node5_output_dir / "optuna_models"
        expected = {
            "image_qc_manifest.csv": 12,
        }
        for filename, expected_rows in expected.items():
            path = output_dir / filename
            if not path.exists():
                raise AssertionError(f"Missing expected artifact: {filename}")
            rows = len(pd.read_csv(path))
            if rows != expected_rows:
                raise AssertionError(f"{filename}: expected {expected_rows} rows, found {rows}")

        expected_node4 = {
            "dat_radiomics_features_full.csv": 12,
            "registration_qc_full.csv": 12,
        }
        expected_node5 = {
            "embedding_coordinates_full.csv": 12,
            "outlier_scores_full.csv": 12,
            "label_audit_full.csv": 12,
        }
        for filename, expected_rows in expected_node4.items():
            path = full_output_dir / filename
            if not path.exists():
                raise AssertionError(f"Missing expected full-cohort artifact: {filename}")
            rows = len(pd.read_csv(path))
            if rows != expected_rows:
                raise AssertionError(f"{filename}: expected {expected_rows} rows, found {rows}")
        for filename, expected_rows in expected_node5.items():
            path = node5_output_dir / filename
            if not path.exists():
                raise AssertionError(f"Missing expected node-05 artifact: {filename}")
            rows = len(pd.read_csv(path))
            if rows != expected_rows:
                raise AssertionError(f"{filename}: expected {expected_rows} rows, found {rows}")

        features = pd.read_csv(full_output_dir / "dat_radiomics_features_full.csv")
        required_feature_prefixes = ["semiquant_", "shape_active_", "texture3d_", "stability_"]
        for prefix in required_feature_prefixes:
            if not any(column.startswith(prefix) for column in features.columns):
                raise AssertionError(f"No feature column starts with {prefix!r}")
        required_background_columns = {
            "background_qc_valid",
            "background_floor_applied",
            "background_scale_used",
            "semiquant_log_target_background",
        }
        if required_background_columns - set(features.columns):
            raise AssertionError("Robust background provenance is incomplete.")
        if not np.isfinite(features["semiquant_sbr"]).all():
            raise AssertionError("Robust SBR contains non-finite values.")

        registration = pd.read_csv(full_output_dir / "registration_qc_full.csv")
        required_registration_columns = {
            "registration_backend",
            "gpu_fallback_used",
            "registration_seconds",
        }
        missing_registration_columns = required_registration_columns - set(registration.columns)
        if missing_registration_columns:
            raise AssertionError(
                f"Missing registration provenance: {sorted(missing_registration_columns)}"
            )
        if not np.isfinite(registration["registration_seconds"]).all():
            raise AssertionError("Non-finite registration timing found.")
        backend_counts = registration["registration_backend"].value_counts().to_dict()
        fallback_count = int(registration["gpu_fallback_used"].astype(bool).sum())
        mean_seconds = float(registration["registration_seconds"].mean())
        if torch.cuda.is_available() and "torch_cuda" not in backend_counts:
            raise AssertionError("CUDA is available but no synthetic registration used it.")
        print(
            "Registration performance: "
            f"backends={backend_counts}, fallbacks={fallback_count}, "
            f"mean={mean_seconds:.3f}s/case",
            flush=True,
        )

        oof = pd.read_csv(optuna_output_dir / "optuna_oof_predictions.csv")
        required_probability_columns = {
            "p_logistic_oof", "p_random_forest_oof", "p_xgboost_oof",
            "p_ensemble_oof",
        }
        if oof["uid"].nunique() != 12 or required_probability_columns - set(oof.columns):
            raise AssertionError("Optuna OOF coverage is incomplete.")
        fold_checkpoint = pd.read_csv(optuna_output_dir / "oof_fold_checkpoint.csv")
        if len(fold_checkpoint) != 3 * len(oof):
            raise AssertionError("Fold-level resume checkpoint was not fully restored.")

        maps_path = full_output_dir / "cohort_maps_full.npz"
        if not maps_path.exists():
            raise AssertionError("Missing cohort_maps_full.npz")
        with np.load(maps_path) as maps:
            for key in ["normal_mean", "pathologic_mean", "difference", "cohen_d", "hedges_g"]:
                if key not in maps or not np.isfinite(maps[key]).all():
                    raise AssertionError(f"Invalid map: {key}")

        state_path = full_output_dir / "streaming_state_full.npz"
        with np.load(state_path) as state:
            if "mapped_uids" not in state or len(state["mapped_uids"]) != 12:
                raise AssertionError("Resume state does not track all mapped UIDs.")

        crop_count = len(list((full_output_dir / "registered_crops").glob("*.npz")))
        if crop_count != 12:
            raise AssertionError(f"Expected 12 registered crops, found {crop_count}")
        sample_crop = next((full_output_dir / "registered_crops").glob("*.npz"))
        with np.load(sample_crop) as crop:
            if "volume_intensity01" not in crop:
                raise AssertionError("Registered crop lacks robust intensity normalization.")

        coverage = pd.read_csv(node5_output_dir / "node5_coverage_qc_full.csv")
        if coverage["features"].sum() != 12:
            raise AssertionError("Node 5 coverage does not include the complete cohort.")

        print("Synthetic end-to-end validation passed (3 full-cohort notebooks).")


if __name__ == "__main__":
    main()
