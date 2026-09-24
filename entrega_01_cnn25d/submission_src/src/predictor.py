from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .model import HybridDaTClassifier, ModelConfig
from .preprocessing import ImageConfig, SubmissionPreprocessor


class DaTEnsemblePredictor:
    def __init__(self, assets_dir: Path, device: str | None = None) -> None:
        self.assets_dir = Path(assets_dir)
        self.manifest = json.loads(
            (self.assets_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.temperature = float(self.manifest["temperature"])
        self.image_config = ImageConfig.from_dict(self.manifest["data_config"])
        self.preprocessor = SubmissionPreprocessor(
            self.assets_dir / self.manifest["template_file"],
            self.image_config,
            self.device,
        )
        self.models: list[HybridDaTClassifier] = []
        expected_config = ModelConfig.from_dict(self.manifest["model_config"])
        for filename in self.manifest["fold_files"]:
            state = torch.load(
                self.assets_dir / filename,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_config = ModelConfig.from_dict(state["model_config"])
            if checkpoint_config != expected_config:
                raise RuntimeError("Fold checkpoint model configuration mismatch.")
            model = HybridDaTClassifier(checkpoint_config)
            model.load_state_dict(state["model_state"], strict=True)
            model.to(self.device).eval()
            self.models.append(model)
        if len(self.models) != 5:
            raise RuntimeError("The selected submission must contain exactly five fold models.")

    def preprocess(self, path: Path) -> torch.Tensor:
        return self.preprocessor.transform(Path(path))

    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        batch = images.to(self.device, non_blocking=self.device.type == "cuda")
        probabilities = []
        with torch.inference_mode():
            for model in self.models:
                logits = model(batch)
                probabilities.append(torch.sigmoid(logits / self.temperature))
        result = torch.stack(probabilities, dim=1).mean(dim=1)
        return result.float().cpu().numpy()

    def predict_paths(self, paths: Iterable[Path], batch_size: int = 24) -> np.ndarray:
        results: list[np.ndarray] = []
        pending: list[torch.Tensor] = []
        for path in paths:
            pending.append(self.preprocess(path))
            if len(pending) >= batch_size:
                results.append(self.predict_batch(torch.stack(pending)))
                pending.clear()
        if pending:
            results.append(self.predict_batch(torch.stack(pending)))
        if not results:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(results).clip(1e-5, 1.0 - 1e-5)
