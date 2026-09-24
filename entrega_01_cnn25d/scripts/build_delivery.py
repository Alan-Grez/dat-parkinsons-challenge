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


DELIVERY_NAME = "entrega_01_cnn25d"
PRIMARY_CANDIDATE = "43765c3f64a985b1"


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
    new_size = np.maximum(
        1, np.rint(old_size * old_spacing / new_spacing)
    ).astype(int)
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
                f"Expected one reference member for the configured UID, found {len(matches)}."
            )
        with tempfile.TemporaryDirectory(prefix="dat_submission_template_") as temporary:
            archive.extract(matches[0], temporary)
            image = sitk.ReadImage(str(Path(temporary) / matches[0]), sitk.sitkFloat32)
            image = sitk.Image(sitk.DICOMOrient(image, "LPS"))
            return make_isotropic_reference(image, spacing_mm)


def write_template_asset(
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
    with np.load(masks_path, allow_pickle=False) as masks:
        target = masks["target"].astype(bool)
        brain = masks["brain"].astype(bool)
        background = masks["background"].astype(bool)
    if target.shape != fixed_array.shape or brain.shape != fixed_array.shape:
        raise RuntimeError("Training masks and registration reference do not share a grid.")
    coordinates = np.argwhere(target)
    if not len(coordinates):
        raise RuntimeError("The selected training target mask is empty.")
    margin = np.ceil(20.0 / np.asarray(fixed.GetSpacing()[::-1])).astype(int)
    crop_start = np.maximum(coordinates.min(axis=0) - margin, 0)
    crop_stop = np.minimum(coordinates.max(axis=0) + margin + 1, target.shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            fixed=fixed_array,
            target_mask=target.astype(np.uint8),
            brain_mask=brain.astype(np.uint8),
            background_mask=background.astype(np.uint8),
            crop_start_zyx=crop_start.astype(np.int16),
            crop_stop_zyx=crop_stop.astype(np.int16),
            spacing_xyz=np.asarray(fixed.GetSpacing(), dtype=np.float64),
            origin_xyz=np.asarray(fixed.GetOrigin(), dtype=np.float64),
            direction=np.asarray(fixed.GetDirection(), dtype=np.float64),
        )
    return {
        "template_type": "training_derived_fixed_reference_without_uid",
        "grid_shape_zyx": list(fixed_array.shape),
        "crop_start_zyx": crop_start.tolist(),
        "crop_stop_zyx": crop_stop.tolist(),
        "spacing_xyz_mm": list(map(float, fixed.GetSpacing())),
    }


def slim_fold_checkpoint(source: Path, destination: Path, expected: dict) -> None:
    state = torch.load(source, map_location="cpu", weights_only=False)
    contract = state["checkpoint_contract"]
    model_config = contract["model_config"]
    if model_config != expected:
        raise RuntimeError(f"Model configuration mismatch in {source}.")
    if model_config["architecture"] != "2.5d" or model_config["feature_variant"] != "image_only":
        raise RuntimeError("The selected delivery must be the primary 2.5D image-only model.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": state["model_state"], "model_config": model_config},
        destination,
    )


def package_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in package_files(source_dir):
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build(project_root: Path) -> Path:
    delivery = project_root / DELIVERY_NAME
    source_dir = delivery / "submission_src"
    assets = source_dir / "assets"
    validation_dir = delivery / "validation"
    assets.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    run_dir = project_root / "outputs" / "private_eda" / "cnn_runs" / "cnn_compact_v1"
    final_dir = run_dir / "final"
    primary_path = final_dir / "primary_ensemble_manifest.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    if primary["candidate_id"] != PRIMARY_CANDIDATE:
        raise RuntimeError(
            f"The current primary candidate is {primary['candidate_id']}, not {PRIMARY_CANDIDATE}."
        )

    upstream_path = (
        project_root
        / "outputs"
        / "private_eda"
        / "full_cohort_v3"
        / "registration_radiomics_full_config.json"
    )
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    template_metadata = write_template_asset(
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

    fold_files = []
    source_hashes: dict[str, str] = {}
    for fold, relative in enumerate(primary["fold_checkpoints"]):
        source = final_dir / relative
        destination = assets / f"fold_{fold}.pt"
        slim_fold_checkpoint(source, destination, primary["model_config"])
        fold_files.append(destination.name)
        source_hashes[f"source_fold_{fold}"] = sha256(source)

    manifest = {
        "schema_version": "dat_submission_v1",
        "candidate_id": primary["candidate_id"],
        "architecture": primary["architecture"],
        "feature_variant": primary["feature_variant"],
        "n_fold_models": 5,
        "fold_files": fold_files,
        "temperature": float(primary["temperature"]),
        "model_config": primary["model_config"],
        "data_config": primary["data_config"],
        "template_file": "registration_template.npz",
        "registration": {
            "orientation": "LPS",
            "mode": "rigid",
            "backend_primary": "torch_cuda",
            "backend_fallback": "SimpleITK",
            "stages": upstream["torch_registration_stages"],
            "early_stopping_patience": upstream["torch_early_stopping_patience"],
        },
        "template": template_metadata,
        "inference_rule": primary["inference_rule"],
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
        "primary_candidate": PRIMARY_CANDIDATE,
        "source_primary_manifest_sha256": sha256(primary_path),
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
        f"{build_manifest['submission_zip_sha256']}  submission.zip\n",
        encoding="utf-8",
    )
    return submission_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()
    path = build(args.project_root.resolve())
    print(path)


if __name__ == "__main__":
    main()
