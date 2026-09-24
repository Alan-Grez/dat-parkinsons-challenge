"""Model-selection utilities for the private DaT exploratory pipeline."""

from .embedding_search import (
    EmbeddingSearchResult,
    rank_trial_metrics,
    run_embedding_search,
)
from .optuna_models import (
    FoldData,
    OptunaExperimentResult,
    run_optuna_experiment,
)
from .cnn import (
    DataConfig,
    EnsemblePredictionResult,
    ExperimentConfig,
    HybridDaTClassifier,
    ModelConfig,
    SearchConfig,
    TrainConfig,
    predict_fold_ensemble,
)

__all__ = [
    "EmbeddingSearchResult",
    "FoldData",
    "OptunaExperimentResult",
    "rank_trial_metrics",
    "run_embedding_search",
    "run_optuna_experiment",
    "DataConfig",
    "EnsemblePredictionResult",
    "ExperimentConfig",
    "HybridDaTClassifier",
    "ModelConfig",
    "SearchConfig",
    "TrainConfig",
    "predict_fold_ensemble",
]
