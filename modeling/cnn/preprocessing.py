from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FoldTabularTransformer:
    pca_variance: float | None = None
    median_: np.ndarray | None = None
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    pca_mean_: np.ndarray | None = None
    pca_components_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> FoldTabularTransformer:
        array = np.asarray(values, dtype=float)
        self.median_ = np.nanmedian(array, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(array), array, self.median_)
        self.center_ = np.median(filled, axis=0)
        q25, q75 = np.percentile(filled, [25, 75], axis=0)
        self.scale_ = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
        scaled = (filled - self.center_) / self.scale_
        if self.pca_variance is not None and scaled.shape[1] > 1:
            pca = PCA(n_components=self.pca_variance, svd_solver="full")
            pca.fit(scaled)
            self.pca_mean_ = pca.mean_.astype(np.float64)
            self.pca_components_ = pca.components_.astype(np.float64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("El transformador tabular aun no ha sido ajustado.")
        array = np.asarray(values, dtype=float)
        filled = np.where(np.isfinite(array), array, self.median_)
        scaled = (filled - self.center_) / self.scale_
        if self.pca_components_ is not None and self.pca_mean_ is not None:
            scaled = (scaled - self.pca_mean_) @ self.pca_components_.T
        return scaled.astype(np.float32)

    @property
    def output_dim(self) -> int:
        if self.pca_components_ is not None:
            return int(self.pca_components_.shape[0])
        if self.median_ is None:
            return 0
        return int(self.median_.shape[0])

    def to_state(self) -> dict[str, Any]:
        return {
            "pca_variance": self.pca_variance,
            "median": self.median_,
            "center": self.center_,
            "scale": self.scale_,
            "pca_mean": self.pca_mean_,
            "pca_components": self.pca_components_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FoldTabularTransformer:
        instance = cls(pca_variance=state.get("pca_variance"))
        instance.median_ = state.get("median")
        instance.center_ = state.get("center")
        instance.scale_ = state.get("scale")
        instance.pca_mean_ = state.get("pca_mean")
        instance.pca_components_ = state.get("pca_components")
        return instance


@dataclass
class MaskedSBRTransformer:
    median_: np.ndarray | None = None
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray, valid: np.ndarray) -> MaskedSBRTransformer:
        array = np.asarray(values, dtype=float)
        valid_array = np.asarray(valid, dtype=bool)
        if not valid_array.any():
            raise ValueError("El fold de entrenamiento no tiene ningun fondo SBR valido.")
        valid_values = array[valid_array]
        self.median_ = np.nanmedian(valid_values, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(valid_values), valid_values, self.median_)
        self.center_ = np.median(filled, axis=0)
        q25, q75 = np.percentile(filled, [25, 75], axis=0)
        self.scale_ = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
        return self

    def transform(self, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("El transformador SBR aun no ha sido ajustado.")
        array = np.asarray(values, dtype=float)
        valid_array = np.asarray(valid, dtype=bool)
        filled = np.where(np.isfinite(array), array, self.median_)
        scaled = ((filled - self.center_) / self.scale_).astype(np.float32)
        scaled[~valid_array] = 0.0
        return scaled

    @property
    def output_dim(self) -> int:
        return int(self.median_.shape[0]) if self.median_ is not None else 0

    def to_state(self) -> dict[str, Any]:
        return {"median": self.median_, "center": self.center_, "scale": self.scale_}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MaskedSBRTransformer:
        return cls(
            median_=state.get("median"),
            center_=state.get("center"),
            scale_=state.get("scale"),
        )
