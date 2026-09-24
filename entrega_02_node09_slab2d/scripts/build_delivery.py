from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

DELIVERY_NAME = "entrega_02_node09_slab2d"
RUN_ID = "node09_fine_v1"
PRIMARY_CANDIDATE = "555b81376733bb67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def make_isotropic_reference(image: sitk.Image, spacing_mm: float) -> sitk.Image:
    old_size = np.asarray(image.GetSize(), dtype=float)
    old_spacing = np.asarray(image.GetSpacing(), dtype=float)
    new_spacing = np.repeat(float(spacing_mm), 3)
    new_size = np.maximum(1, np.rint(old_size * old_spacing / new_spacing)).astype(int)
    return sitk.Resample(
        image,
        [int(value) for value in new_size],
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        tuple(float(value) for value in new_spacing),
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )


def load_reference_from_archive(
    archive_path: Path, reference_uid: str, spacing_mm: float
) -> sitk.Image:
    with zipfile.ZipFile(archive_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if Path(name).name == f"{reference_uid}.nii.gz"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one configured registration reference, found {len(matches)}."
            )
        with tempfile.TemporaryDirectory(prefix="dat_node09_template_") as temporary:
            archive.extract(matches[0], temporary)
            image = sitk.ReadImage(str(Path(temporary) / matches[0]), sitk.sitkFloat32)
            return make_isotropic_reference(
                sitk.Image(sitk.DICOMOrient(image, "LPS")), spacing_mm
            )


def write_registration_asset(
    *,
    archive_path: Path,
    masks_path: Path,
    upstream_config: dict,
    output_path: Path,
) -> dict:
    spacing_mm = float(upstream_config["isotropic_spacing_mm"])
    fixed = load_reference_from_archive(
        archive_path, str(upstream_config["reference_uid"]), spacing_mm
    )
    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.float32)
    with np.load(masks_path, allow_pickle=False) as payload:
        masks = {
            name: np.asarray(payload[name], dtype=np.uint8)
            for name in (
                "target",
                "background",
                "brain",
                "right",
                "left",
                "anterior",
                "posterior",
            )
        }
    if any(mask.shape != fixed_array.shape for mask in masks.values()):
        raise RuntimeError("Training masks and registration reference have different grids.")
    coordinates = np.argwhere(masks["target"] > 0)
    if not len(coordinates):
        raise RuntimeError("The frozen target mask is empty.")
    margin = np.ceil(20.0 / np.asarray(fixed.GetSpacing()[::-1])).astype(int)
    crop_start = np.maximum(coordinates.min(axis=0) - margin, 0)
    crop_stop = np.minimum(coordinates.max(axis=0) + margin + 1, masks["target"].shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            fixed=fixed_array,
            target_mask=masks["target"],
            brain_mask=masks["brain"],
            background_mask=masks["background"],
            crop_start_zyx=crop_start.astype(np.int16),
            crop_stop_zyx=crop_stop.astype(np.int16),
            spacing_xyz=np.asarray(fixed.GetSpacing(), dtype=np.float64),
            origin_xyz=np.asarray(fixed.GetOrigin(), dtype=np.float64),
            direction=np.asarray(fixed.GetDirection(), dtype=np.float64),
            **masks,
        )
    return {
        "template_type": "training-derived fixed reference without UID",
        "grid_shape_zyx": list(fixed_array.shape),
        "crop_start_zyx": crop_start.tolist(),
        "crop_stop_zyx": crop_stop.tolist(),
        "spacing_xyz_mm": list(map(float, fixed.GetSpacing())),
    }


def locate_fold_template(project_root: Path, contract: dict) -> Path:
    root = project_root / "outputs" / "private_eda" / "node07_cache" / "v3" / "final"
    matches: list[Path] = []
    for metadata_path in root.glob("fold_registered/*/template_bank.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("fold", -1)) == int(contract["fold"])
            and metadata.get("train_uids") == contract["train_uids"]
            and metadata.get("upstream_hash") == contract["upstream_hash"]
        ):
            matches.append(metadata_path.with_suffix(".npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one fold-specific template bank for fold {contract['fold']}, "
            f"found {len(matches)}."
        )
    return matches[0]


def slim_checkpoint(source: Path, destination: Path, fold: int) -> dict:
    state = torch.load(source, map_location="cpu", weights_only=False)
    contract = state["checkpoint_contract"]
    model = contract["model"]
    if int(contract["fold"]) != fold:
        raise RuntimeError(f"Checkpoint fold mismatch in {source}.")
    if model["architecture"] != "slab2d":
        raise RuntimeError("The node09 submission winner must be slab2d.")
    if model["feature_variant"] != "image_radiomics_sbr":
        raise RuntimeError("The node09 submission winner must include regional and SBR branches.")
    regional = state["regional_transformer"]
    original_indices = np.asarray(regional["retained_indices"], dtype=int)[
        np.asarray(regional["selected_indices"], dtype=int)
    ]
    selected_names = list(map(str, regional["selected_names"]))
    if len(selected_names) != len(original_indices):
        raise RuntimeError("Regional transformer names and indices are inconsistent.")
    slim = {
        "model_state": state["model_state"],
        "model_config": model,
        "data_config": contract["data"],
        "regional_transformer": {
            "selected_names": selected_names,
            "center": np.asarray(regional["center"], dtype=np.float64)[original_indices],
            "scale": np.asarray(regional["scale"], dtype=np.float64)[original_indices],
            "median": np.asarray(regional["median"], dtype=np.float64)[original_indices],
            "pca_mean": regional.get("pca_mean"),
            "pca_components": regional.get("pca_components"),
        },
        "sbr_columns": list(map(str, contract["sbr_columns"])),
        "sbr_transformer": state["sbr_transformer"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, destination)
    return {
        "source_checkpoint_sha256": sha256(source),
        "selected_regional_features": len(selected_names),
        "sbr_features": len(slim["sbr_columns"]),
    }


def package_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
    )


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in package_files(source_dir):
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def copy_runtime_modules(project_root: Path, source_dir: Path) -> None:
    destination = source_dir / "modeling"
    (destination / "cnn").mkdir(parents=True, exist_ok=True)
    (destination / "dat_spect_v2").mkdir(parents=True, exist_ok=True)
    for path in (destination, destination / "cnn", destination / "dat_spect_v2"):
        (path / "__init__.py").write_text("", encoding="utf-8")
    for name in ("io.py",):
        shutil.copy2(project_root / "modeling" / "cnn" / name, destination / "cnn" / name)
    for name in ("config.py", "models.py", "features.py", "preprocessing.py"):
        shutil.copy2(
            project_root / "modeling" / "dat_spect_v2" / name,
            destination / "dat_spect_v2" / name,
        )
    raw_registration = source_dir / "src" / "raw_registration.py"
    if not raw_registration.exists():
        raise FileNotFoundError(
            "Missing the delivery-owned raw registration implementation: "
            f"{raw_registration}"
        )


def build(project_root: Path) -> Path:
    delivery = project_root / DELIVERY_NAME
    source_dir = delivery / "submission_src"
    assets = source_dir / "assets"
    validation_dir = delivery / "validation"
    assets.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    copy_runtime_modules(project_root, source_dir)

    run_dir = (
        project_root / "outputs" / "private_eda" / "node09_runs" / RUN_ID / "final"
    )
    deployment_path = run_dir / "deployment_manifest.json"
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    if deployment["primary_candidate_id"] != PRIMARY_CANDIDATE:
        raise RuntimeError(
            f"Current node09 primary is {deployment['primary_candidate_id']}, "
            f"not {PRIMARY_CANDIDATE}."
        )
    metrics_rows = __import__("csv").DictReader(
        (run_dir / "final_metrics.csv").open("r", encoding="utf-8", newline="")
    )
    metric = next(row for row in metrics_rows if row["candidate_id"] == PRIMARY_CANDIDATE)

    upstream_path = (
        project_root
        / "outputs"
        / "private_eda"
        / "full_cohort_v3"
        / "registration_radiomics_full_config.json"
    )
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    template_metadata = write_registration_asset(
        archive_path=project_root / "data" / "raw" / "niftis.zip",
        masks_path=(
            project_root
            / "outputs"
            / "private_eda"
            / "full_cohort_v3"
            / "analysis_masks_full.npz"
        ),
        upstream_config=upstream,
        output_path=assets / "registration_template.npz",
    )

    folds: list[dict] = []
    source_hashes: dict[str, str] = {}
    for fold, source_text in enumerate(deployment["fold_checkpoints"]):
        source = Path(source_text)
        state = torch.load(source, map_location="cpu", weights_only=False)
        contract = state["checkpoint_contract"]
        checkpoint_name = f"fold_{fold}.pt"
        template_name = f"fold_{fold}_templates.npz"
        details = slim_checkpoint(source, assets / checkpoint_name, fold)
        template_source = locate_fold_template(project_root, contract)
        shutil.copy2(template_source, assets / template_name)
        folds.append(
            {
                "fold": fold,
                "checkpoint_file": checkpoint_name,
                "template_file": template_name,
                **details,
            }
        )
        source_hashes[f"fold_{fold}"] = sha256(source)

    manifest = {
        "schema_version": "dat_node09_submission_v1",
        "candidate_id": PRIMARY_CANDIDATE,
        "architecture": "slab2d",
        "feature_variant": "image_radiomics_sbr",
        "temperature": float(metric["deployment_temperature"]),
        "cross_validated_log_loss": float(metric["calibrated_log_loss"]),
        "n_fold_models": len(folds),
        "folds": folds,
        "registration_template_file": "registration_template.npz",
        "registration": {
            "orientation": "LPS",
            "mode": "rigid then fold-specific bounded affine",
            "backend_primary": "torch_cuda",
            "backend_fallback": "SimpleITK",
            "stages": upstream["torch_registration_stages"],
            "early_stopping_patience": upstream["torch_early_stopping_patience"],
        },
        "template": template_metadata,
        "inference_rule": "temperature-calibrate each fold logit, then mean five probabilities",
        "acquisition_family_used_as_predictor": False,
        "test_samples_processed_independently": True,
    }
    (assets / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    submission_zip = delivery / "submission" / "submission.zip"
    deterministic_zip(source_dir, submission_zip)
    packaged_files = {
        path.relative_to(source_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in package_files(source_dir)
    }
    build_manifest = {
        "delivery": DELIVERY_NAME,
        "source_run": RUN_ID,
        "primary_candidate": PRIMARY_CANDIDATE,
        "selection_metric": "five-fold OOF cross-calibrated log loss",
        "cross_validated_log_loss": float(metric["calibrated_log_loss"]),
        "source_deployment_manifest_sha256": sha256(deployment_path),
        "source_upstream_config_sha256": sha256(upstream_path),
        "source_checkpoint_sha256": source_hashes,
        "submission_zip_bytes": submission_zip.stat().st_size,
        "submission_zip_sha256": sha256(submission_zip),
        "packaged_files": packaged_files,
    }
    (validation_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (validation_dir / "submission.sha256").write_text(
        f"{build_manifest['submission_zip_sha256']}  submission.zip\n", encoding="utf-8"
    )
    return submission_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()
    print(build(args.project_root.resolve()))


if __name__ == "__main__":
    main()
