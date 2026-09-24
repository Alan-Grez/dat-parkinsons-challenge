"""Compact, resumable 2.5D/3D modeling for the private DaT pipeline."""

from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
)
from .data import CropDataset, InvarianceAugment3D, prepare_image_cache
from .features import (
    SBR_FEATURE_COLUMNS,
    build_derived_feature_cache,
    core_feature_columns,
)
from .folds import (
    create_or_load_balanced_group_folds,
    fold_summary,
    validate_fold_manifest,
)
from .inference import EnsemblePredictionResult, predict_fold_ensemble
from .models import HybridDaTClassifier

__all__ = [
    "SBR_FEATURE_COLUMNS",
    "CropDataset",
    "DataConfig",
    "EnsemblePredictionResult",
    "ExperimentConfig",
    "HybridDaTClassifier",
    "InvarianceAugment3D",
    "ModelConfig",
    "SearchConfig",
    "TrainConfig",
    "build_derived_feature_cache",
    "core_feature_columns",
    "create_or_load_balanced_group_folds",
    "fold_summary",
    "predict_fold_ensemble",
    "prepare_image_cache",
    "validate_fold_manifest",
]
