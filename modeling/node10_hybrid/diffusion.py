from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances


@dataclass
class DiffusionMapTransformer:
    """Fold-local diffusion map with a Nyström out-of-sample extension."""

    n_components: int = 8
    epsilon_quantile: float = 0.5
    diffusion_time: float = 1.0
    train_: np.ndarray | None = None
    epsilon_: float | None = None
    eigenvalues_: np.ndarray | None = None
    eigenvectors_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> DiffusionMapTransformer:
        array = np.asarray(values, dtype=np.float64)
        distances = pairwise_distances(array, metric="euclidean", squared=True)
        positive = distances[distances > 0]
        epsilon = float(np.quantile(positive, self.epsilon_quantile)) if positive.size else 1.0
        self.epsilon_ = max(epsilon, 1e-8)
        kernel = np.exp(-distances / self.epsilon_)
        degree = kernel.sum(axis=1).clip(min=1e-12)
        normalized = kernel / np.sqrt(degree[:, None] * degree[None, :])
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        order = np.argsort(eigenvalues)[::-1]
        # First eigenvector is the stationary/trivial component.
        count = min(self.n_components + 1, len(order))
        selected = order[1:count]
        self.train_ = array
        self.eigenvalues_ = np.clip(eigenvalues[selected], 1e-8, None)
        self.eigenvectors_ = eigenvectors[:, selected] / np.sqrt(degree[:, None])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (self.train_, self.epsilon_, self.eigenvalues_, self.eigenvectors_)
        ):
            raise RuntimeError("DiffusionMapTransformer no ajustado.")
        array = np.asarray(values, dtype=np.float64)
        distances = pairwise_distances(array, self.train_, metric="euclidean", squared=True)
        kernel = np.exp(-distances / float(self.epsilon_))
        row_sum = kernel.sum(axis=1, keepdims=True).clip(min=1e-12)
        transition = kernel / row_sum
        coordinates = transition @ self.eigenvectors_
        # Nyström gives psi(x)=P(x, train) psi(train) / lambda. Diffusion
        # coordinates are lambda**t * psi(x), hence the t-1 exponent here.
        scale = np.power(self.eigenvalues_, self.diffusion_time - 1.0)
        return np.asarray(coordinates * scale[None, :], dtype=np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)
