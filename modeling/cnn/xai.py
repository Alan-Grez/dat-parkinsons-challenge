"""Explainability utilities for compact 3D DaT classifiers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "CounterfactualResult3D",
    "GradCAM3D",
    "OcclusionSensitivity3DResult",
    "counterfactual_suite_3d",
    "occlusion_sensitivity_3d",
    "quantify_explanations",
]


@dataclass(frozen=True)
class OcclusionSensitivity3DResult:
    """Occlusion maps and the unperturbed class score."""

    baseline_score: Tensor
    signed_delta: Tensor
    importance: Tensor
    coverage: Tensor

    @property
    def heatmap(self) -> Tensor:
        """Alias for the absolute occlusion importance map."""

        return self.importance


@dataclass(frozen=True)
class CounterfactualResult3D:
    """Probabilities and changes produced by controlled perturbations."""

    baseline_probability: Tensor
    probabilities: dict[str, Tensor]
    deltas: dict[str, Tensor]
    expected_direction: dict[str, str]


def _validate_image(image: Tensor) -> None:
    if not isinstance(image, Tensor) or image.ndim != 5:
        raise ValueError("image must be a tensor with shape BxCxDxHxW.")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image must contain at least one sample and one channel.")
    if not image.is_floating_point():
        raise TypeError("image must use a floating-point dtype.")


def _call_model(
    model: nn.Module,
    image: Tensor,
    *,
    tabular: Tensor | None,
    sbr: Tensor | None,
    sbr_valid: Tensor | None,
) -> Any:
    kwargs: dict[str, Tensor] = {}
    if tabular is not None:
        kwargs["tabular"] = tabular
    if sbr is not None:
        kwargs["sbr"] = sbr
    if sbr_valid is not None:
        kwargs["sbr_valid"] = sbr_valid
    return model(image, **kwargs)


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = None
        for key in ("logits", "logit", "output", "prediction", "pred"):
            candidate = output.get(key)
            if isinstance(candidate, Tensor):
                logits = candidate
                break
        if logits is None:
            raise TypeError("Model mapping output does not contain a tensor logit.")
    elif isinstance(output, (tuple, list)):
        logits = next((value for value in output if isinstance(value, Tensor)), None)
        if logits is None:
            raise TypeError("Model sequence output does not contain a tensor logit.")
    else:
        raise TypeError("Model output must contain logits as a tensor.")

    if logits.ndim == 0:
        logits = logits.reshape(1)
    if logits.ndim > 2:
        raise ValueError("Classifier logits must have shape B, Bx1, or BxC.")
    return logits


def _target_tensor(
    target_class: int | Tensor | None,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if target_class is None:
        return torch.ones(batch_size, dtype=torch.long, device=device)
    if isinstance(target_class, int):
        return torch.full((batch_size,), target_class, dtype=torch.long, device=device)
    target = torch.as_tensor(target_class, dtype=torch.long, device=device).reshape(-1)
    if target.numel() == 1:
        return target.expand(batch_size)
    if target.numel() != batch_size:
        raise ValueError("target_class must be scalar or contain one class per sample.")
    return target


def _select_scores(logits: Tensor, target_class: int | Tensor | None) -> Tensor:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        binary_logits = logits.reshape(-1)
        targets = _target_tensor(target_class, len(binary_logits), logits.device)
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("A one-logit classifier only supports target classes 0 and 1.")
        return torch.where(targets == 1, binary_logits, -binary_logits)

    targets = _target_tensor(target_class, logits.shape[0], logits.device)
    if torch.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("target_class is outside the model output range.")
    return logits.gather(1, targets[:, None]).squeeze(1)


def _select_probabilities(logits: Tensor, target_class: int | Tensor | None) -> Tensor:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return torch.sigmoid(_select_scores(logits, target_class))
    targets = _target_tensor(target_class, logits.shape[0], logits.device)
    return torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)


def _model_probabilities(
    model: nn.Module,
    image: Tensor,
    *,
    tabular: Tensor | None,
    sbr: Tensor | None,
    sbr_valid: Tensor | None,
    target_class: int | Tensor | None,
) -> Tensor:
    output = _call_model(
        model,
        image,
        tabular=tabular,
        sbr=sbr,
        sbr_valid=sbr_valid,
    )
    return _select_probabilities(_extract_logits(output), target_class)


def _normalize_map(values: Tensor, eps: float = 1e-8) -> Tensor:
    flat = values.flatten(start_dim=1)
    minimum = flat.amin(dim=1).view(-1, 1, 1, 1, 1)
    maximum = flat.amax(dim=1).view(-1, 1, 1, 1, 1)
    scale = maximum - minimum
    normalized = (values - minimum) / scale.clamp_min(eps)
    return torch.where(scale > eps, normalized, torch.zeros_like(normalized))


class GradCAM3D:
    """Generate 3D Grad-CAM maps from one convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activation: Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, Tensor):
            self._activation = output
            return
        if isinstance(output, (tuple, list)):
            self._activation = next(
                (value for value in output if isinstance(value, Tensor)), None
            )
            return
        self._activation = None

    def close(self) -> None:
        """Remove the forward hook."""

        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __call__(
        self,
        image: Tensor,
        *,
        tabular: Tensor | None = None,
        sbr: Tensor | None = None,
        sbr_valid: Tensor | None = None,
        target_class: int | Tensor | None = 1,
        normalize: bool = True,
        return_on_cpu: bool = True,
    ) -> Tensor:
        """Return a Bx1xDxHxW Grad-CAM map for ``target_class``."""

        _validate_image(image)
        was_training = self.model.training
        self.model.eval()
        self._activation = None
        try:
            with torch.enable_grad():
                differentiable_image = image.detach().requires_grad_(True)
                output = _call_model(
                    self.model,
                    differentiable_image,
                    tabular=tabular,
                    sbr=sbr,
                    sbr_valid=sbr_valid,
                )
                logits = _extract_logits(output)
                scores = _select_scores(logits, target_class)
                activation = self._activation
                if activation is None:
                    raise RuntimeError("The target layer did not produce a tensor activation.")
                if activation.ndim != 5:
                    raise ValueError("GradCAM3D target activation must have shape BxCxDxHxW.")
                gradients = torch.autograd.grad(
                    scores.sum(), activation, retain_graph=False, create_graph=False
                )[0]
                weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                if cam.shape[2:] != image.shape[2:]:
                    cam = F.interpolate(
                        cam,
                        size=image.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                if normalize:
                    cam = _normalize_map(cam)
                cam = cam.detach()
        finally:
            self.model.train(was_training)
        return cam.cpu() if return_on_cpu else cam


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(item) for item in value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain exactly three values.")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive.")
    return result


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    patch = min(size, patch)
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    last = size - patch
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _index_optional(value: Tensor | None, indices: Tensor, batch_size: int) -> Tensor | None:
    if value is None:
        return None
    if value.ndim == 0:
        return value.expand(len(indices))
    if value.shape[0] == 1 and batch_size > 1:
        return value.expand((batch_size, *value.shape[1:])).index_select(0, indices)
    if value.shape[0] != batch_size:
        raise ValueError("Optional model input does not match the image batch size.")
    return value.index_select(0, indices)


def _apply_occlusion(
    batch: Tensor,
    source_indices: Tensor,
    windows: Sequence[tuple[int, int, int, int, int, int]],
    occlusion_value: float | Tensor,
    original_image: Tensor,
) -> None:
    scalar_value: float | None = None
    tensor_value: Tensor | None = None
    if isinstance(occlusion_value, Tensor):
        tensor_value = occlusion_value.to(device=batch.device, dtype=batch.dtype)
        if tensor_value.numel() == 1:
            scalar_value = float(tensor_value.item())
    else:
        scalar_value = float(occlusion_value)

    for row, ((z0, z1, y0, y1, x0, x1), source_index) in enumerate(
        zip(windows, source_indices.tolist(), strict=True)
    ):
        if scalar_value is not None:
            batch[row, :, z0:z1, y0:y1, x0:x1] = scalar_value
        elif tensor_value is not None and tensor_value.shape == original_image.shape:
            batch[row, :, z0:z1, y0:y1, x0:x1] = tensor_value[
                source_index, :, z0:z1, y0:y1, x0:x1
            ]
        else:
            try:
                batch[row, :, z0:z1, y0:y1, x0:x1] = tensor_value
            except RuntimeError as error:
                raise ValueError(
                    "occlusion_value must be scalar, broadcastable, or match image."
                ) from error


def occlusion_sensitivity_3d(
    model: nn.Module,
    image: Tensor,
    *,
    tabular: Tensor | None = None,
    sbr: Tensor | None = None,
    sbr_valid: Tensor | None = None,
    target_class: int | Tensor | None = 1,
    patch_size: int | Sequence[int] = 16,
    stride: int | Sequence[int] = 8,
    occlusion_value: float | Tensor = 0.0,
    inference_batch_size: int = 16,
    return_on_cpu: bool = True,
) -> OcclusionSensitivity3DResult:
    """Compute batched 3D occlusion sensitivity with overlap averaging."""

    _validate_image(image)
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive.")
    patch = _triple(patch_size, "patch_size")
    step = _triple(stride, "stride")
    batch_size, _, depth, height, width = image.shape
    patch = tuple(min(size, item) for size, item in zip((depth, height, width), patch))
    targets = _target_tensor(target_class, batch_size, image.device)

    tasks: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    for sample in range(batch_size):
        for z0 in _window_starts(depth, patch[0], step[0]):
            for y0 in _window_starts(height, patch[1], step[1]):
                for x0 in _window_starts(width, patch[2], step[2]):
                    tasks.append(
                        (
                            sample,
                            (z0, z0 + patch[0], y0, y0 + patch[1], x0, x0 + patch[2]),
                        )
                    )

    signed_sum = torch.zeros(
        (batch_size, 1, depth, height, width),
        dtype=torch.float32,
        device=image.device,
    )
    coverage = torch.zeros_like(signed_sum)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline_logits = _extract_logits(
                _call_model(
                    model,
                    image,
                    tabular=tabular,
                    sbr=sbr,
                    sbr_valid=sbr_valid,
                )
            )
            baseline_score = _select_scores(baseline_logits, targets)

            for start in range(0, len(tasks), inference_batch_size):
                task_batch = tasks[start : start + inference_batch_size]
                source_indices = torch.as_tensor(
                    [sample for sample, _window in task_batch],
                    dtype=torch.long,
                    device=image.device,
                )
                windows = [window for _sample, window in task_batch]
                occluded = image.index_select(0, source_indices).clone()
                _apply_occlusion(
                    occluded,
                    source_indices,
                    windows,
                    occlusion_value,
                    image,
                )
                logits = _extract_logits(
                    _call_model(
                        model,
                        occluded,
                        tabular=_index_optional(tabular, source_indices, batch_size),
                        sbr=_index_optional(sbr, source_indices, batch_size),
                        sbr_valid=_index_optional(sbr_valid, source_indices, batch_size),
                    )
                )
                selected_targets = targets.index_select(0, source_indices)
                occluded_score = _select_scores(logits, selected_targets)
                delta = baseline_score.index_select(0, source_indices) - occluded_score

                for row, (sample, window) in enumerate(task_batch):
                    z0, z1, y0, y1, x0, x1 = window
                    signed_sum[sample, :, z0:z1, y0:y1, x0:x1] += delta[row].float()
                    coverage[sample, :, z0:z1, y0:y1, x0:x1] += 1.0
    finally:
        model.train(was_training)

    signed_delta = signed_sum / coverage.clamp_min(1.0)
    result = OcclusionSensitivity3DResult(
        baseline_score=baseline_score.detach(),
        signed_delta=signed_delta.detach(),
        importance=signed_delta.abs().detach(),
        coverage=coverage.detach(),
    )
    if not return_on_cpu:
        return result
    return OcclusionSensitivity3DResult(
        baseline_score=result.baseline_score.cpu(),
        signed_delta=result.signed_delta.cpu(),
        importance=result.importance.cpu(),
        coverage=result.coverage.cpu(),
    )


def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    radius = max(1, round(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_blur3d(image: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return image.clone()
    kernel = _gaussian_kernel1d(sigma, image.dtype, image.device)
    channels = image.shape[1]
    radius = kernel.numel() // 2
    result = image
    for dimension in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[dimension + 2] = kernel.numel()
        weight = kernel.reshape(shape).expand(channels, 1, *shape[2:]).contiguous()
        padding = [0, 0, 0]
        padding[dimension] = radius
        result = F.conv3d(result, weight, padding=tuple(padding), groups=channels)
    return result


def _prepare_mask(mask: Tensor | None, image: Tensor, name: str) -> Tensor | None:
    if mask is None:
        return None
    prepared = torch.as_tensor(mask, device=image.device)
    if prepared.ndim == 3:
        prepared = prepared[None, None]
    elif prepared.ndim == 4:
        prepared = prepared[:, None]
    if prepared.ndim != 5 or prepared.shape[2:] != image.shape[2:]:
        raise ValueError(f"{name} must be broadcastable to Bx1xDxHxW.")
    if prepared.shape[0] == 1 and image.shape[0] > 1:
        prepared = prepared.expand(image.shape[0], -1, -1, -1, -1)
    if prepared.shape[0] != image.shape[0] or prepared.shape[1] not in (1, image.shape[1]):
        raise ValueError(f"{name} does not match the image batch or channel count.")
    return prepared.to(dtype=image.dtype).clamp(0, 1)


def counterfactual_suite_3d(
    model: nn.Module,
    image: Tensor,
    *,
    tabular: Tensor | None = None,
    sbr: Tensor | None = None,
    sbr_valid: Tensor | None = None,
    target_class: int | Tensor | None = 1,
    target_mask: Tensor | None = None,
    background_mask: Tensor | None = None,
    gain_factors: Sequence[float] = (0.85, 1.15),
    gamma_values: Sequence[float] = (0.85, 1.15),
    blur_sigmas: Sequence[float] = (0.6,),
    noise_std: float = 0.03,
    target_attenuation: float = 0.35,
    lr_axis: int = -1,
    random_seed: int = 20260821,
    return_on_cpu: bool = True,
) -> CounterfactualResult3D:
    """Evaluate invariant and biologically directed 3D counterfactuals."""

    _validate_image(image)
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")
    if not 0 <= target_attenuation <= 1:
        raise ValueError("target_attenuation must lie in [0, 1].")
    if lr_axis < 0:
        lr_axis += image.ndim
    if lr_axis not in (2, 3, 4):
        raise ValueError("lr_axis must identify one of the three spatial dimensions.")

    target_region = _prepare_mask(target_mask, image, "target_mask")
    background_region = _prepare_mask(background_mask, image, "background_mask")
    transforms: dict[str, Tensor] = {}
    expected: dict[str, str] = {}

    for factor in gain_factors:
        if factor <= 0:
            raise ValueError("gain factors must be positive.")
        name = f"gain_{factor:g}"
        transforms[name] = image * float(factor)
        expected[name] = "stable"
    for gamma in gamma_values:
        if gamma <= 0:
            raise ValueError("gamma values must be positive.")
        name = f"gamma_{gamma:g}"
        transforms[name] = image.sign() * image.abs().pow(float(gamma))
        expected[name] = "stable"
    for sigma in blur_sigmas:
        if sigma < 0:
            raise ValueError("blur sigmas cannot be negative.")
        name = f"blur_{sigma:g}"
        transforms[name] = _gaussian_blur3d(image, float(sigma))
        expected[name] = "stable"
    if noise_std > 0:
        generator = torch.Generator(device=image.device)
        generator.manual_seed(random_seed)
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device=image.device,
            generator=generator,
        )
        transforms[f"noise_{noise_std:g}"] = image + noise_std * noise
        expected[f"noise_{noise_std:g}"] = "stable"

    transforms["flip_lr"] = torch.flip(image, dims=(lr_axis,))
    expected["flip_lr"] = "stable"
    if target_region is not None:
        transforms[f"target_attenuation_{target_attenuation:g}"] = image * (
            1.0 - target_attenuation * target_region
        )
        expected[f"target_attenuation_{target_attenuation:g}"] = "increase"
    if background_region is not None:
        transforms["background_zero"] = image * (1.0 - background_region)
        expected["background_zero"] = "stable"

    probabilities: dict[str, Tensor] = {}
    deltas: dict[str, Tensor] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = _model_probabilities(
                model,
                image,
                tabular=tabular,
                sbr=sbr,
                sbr_valid=sbr_valid,
                target_class=target_class,
            )
            for name, transformed in transforms.items():
                probability = _model_probabilities(
                    model,
                    transformed,
                    tabular=tabular,
                    sbr=sbr,
                    sbr_valid=sbr_valid,
                    target_class=target_class,
                )
                probabilities[name] = probability.detach()
                deltas[name] = (probability - baseline).detach()
    finally:
        model.train(was_training)

    baseline = baseline.detach()
    if return_on_cpu:
        baseline = baseline.cpu()
        probabilities = {name: value.cpu() for name, value in probabilities.items()}
        deltas = {name: value.cpu() for name, value in deltas.items()}
    return CounterfactualResult3D(
        baseline_probability=baseline,
        probabilities=probabilities,
        deltas=deltas,
        expected_direction=expected,
    )


def _batch_size_from_explanations(
    gradcam: Tensor | None,
    occlusion: OcclusionSensitivity3DResult | Tensor | None,
    counterfactuals: CounterfactualResult3D | None,
) -> int:
    if gradcam is not None:
        return gradcam.shape[0]
    if isinstance(occlusion, OcclusionSensitivity3DResult):
        return occlusion.importance.shape[0]
    if isinstance(occlusion, Tensor):
        return occlusion.shape[0]
    if counterfactuals is not None:
        return counterfactuals.baseline_probability.numel()
    raise ValueError("At least one explanation source is required.")


def _explanation_map(value: Tensor, batch_size: int, name: str) -> Tensor:
    result = value.detach().float().cpu()
    if result.ndim == 4:
        result = result[:, None]
    if result.ndim != 5 or result.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape Bx1xDxHxW or BxDxHxW.")
    if result.shape[1] != 1:
        result = result.abs().mean(dim=1, keepdim=True)
    return result


def _mask_for_map(mask: Tensor | None, reference: Tensor, name: str) -> Tensor | None:
    if mask is None:
        return None
    result = torch.as_tensor(mask).detach().float().cpu()
    if result.ndim == 3:
        result = result[None, None]
    elif result.ndim == 4:
        result = result[:, None]
    if result.ndim != 5:
        raise ValueError(f"{name} must have 3, 4, or 5 dimensions.")
    if result.shape[0] == 1 and reference.shape[0] > 1:
        result = result.expand(reference.shape[0], -1, -1, -1, -1)
    if result.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} batch size does not match explanations.")
    if result.shape[2:] != reference.shape[2:]:
        result = F.interpolate(result, size=reference.shape[2:], mode="nearest")
    if result.shape[1] != 1:
        result = result.amax(dim=1, keepdim=True)
    return result.clamp(0, 1)


def _roi_metrics(values: Tensor, mask: Tensor | None, index: int) -> tuple[float, float]:
    if mask is None:
        return float("nan"), float("nan")
    saliency = values[index].abs()
    region = mask[index]
    saliency_fraction = float((saliency * region).sum() / saliency.sum().clamp_min(1e-8))
    volume_fraction = float(region.mean())
    enrichment = saliency_fraction / max(volume_fraction, 1e-8)
    return saliency_fraction, enrichment


def _pearson(first: Tensor, second: Tensor) -> float:
    x = first.flatten().float()
    y = second.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denominator)


def quantify_explanations(
    gradcam: Tensor | None = None,
    occlusion: OcclusionSensitivity3DResult | Tensor | None = None,
    counterfactuals: CounterfactualResult3D | None = None,
    *,
    target_mask: Tensor | None = None,
    background_mask: Tensor | None = None,
    minimum_spatial_agreement: float = 0.10,
    invariance_tolerance: float = 0.10,
    direction_tolerance: float = 0.01,
) -> list[dict[str, float | bool | str]]:
    """Quantify XAI evidence, prioritizing occlusion and counterfactual tests."""

    batch_size = _batch_size_from_explanations(gradcam, occlusion, counterfactuals)
    gradcam_map = (
        _explanation_map(gradcam, batch_size, "gradcam") if gradcam is not None else None
    )
    if isinstance(occlusion, OcclusionSensitivity3DResult):
        occlusion_map = _explanation_map(occlusion.importance, batch_size, "occlusion")
    elif isinstance(occlusion, Tensor):
        occlusion_map = _explanation_map(occlusion, batch_size, "occlusion")
    else:
        occlusion_map = None

    if (
        gradcam_map is not None
        and occlusion_map is not None
        and gradcam_map.shape[2:] != occlusion_map.shape[2:]
    ):
        gradcam_map = F.interpolate(
            gradcam_map,
            size=occlusion_map.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    if (
        counterfactuals is not None
        and counterfactuals.baseline_probability.numel() != batch_size
    ):
        raise ValueError("counterfactual batch size does not match explanations.")

    reference = occlusion_map if occlusion_map is not None else gradcam_map
    if reference is None:
        reference = torch.zeros((batch_size, 1, 1, 1, 1))
    target_region = _mask_for_map(target_mask, reference, "target_mask")
    background_region = _mask_for_map(background_mask, reference, "background_mask")

    rows: list[dict[str, float | bool | str]] = []
    for index in range(batch_size):
        row: dict[str, float | bool | str] = {}
        grad_target, grad_enrichment = (
            _roi_metrics(gradcam_map, target_region, index)
            if gradcam_map is not None
            else (float("nan"), float("nan"))
        )
        occ_target, occ_enrichment = (
            _roi_metrics(occlusion_map, target_region, index)
            if occlusion_map is not None
            else (float("nan"), float("nan"))
        )
        grad_background, _ = (
            _roi_metrics(gradcam_map, background_region, index)
            if gradcam_map is not None
            else (float("nan"), float("nan"))
        )
        occ_background, _ = (
            _roi_metrics(occlusion_map, background_region, index)
            if occlusion_map is not None
            else (float("nan"), float("nan"))
        )
        row.update(
            {
                "gradcam_target_fraction": grad_target,
                "gradcam_target_enrichment": grad_enrichment,
                "gradcam_background_fraction": grad_background,
                "occlusion_target_fraction": occ_target,
                "occlusion_target_enrichment": occ_enrichment,
                "occlusion_background_fraction": occ_background,
            }
        )

        agreement = float("nan")
        spatial_contradiction = False
        if gradcam_map is not None and occlusion_map is not None:
            agreement = _pearson(gradcam_map[index], occlusion_map[index])
            spatial_contradiction = (
                math.isfinite(agreement) and agreement < minimum_spatial_agreement
            )
        row["gradcam_occlusion_correlation"] = agreement

        invariant_deltas: list[float] = []
        direction_contradiction = False
        target_attenuation_delta = float("nan")
        if counterfactuals is not None:
            for name, delta_tensor in counterfactuals.deltas.items():
                delta = float(delta_tensor.reshape(-1)[index])
                direction = counterfactuals.expected_direction.get(name, "stable")
                if direction == "stable":
                    invariant_deltas.append(abs(delta))
                elif direction == "increase":
                    target_attenuation_delta = delta
                    direction_contradiction |= delta < -direction_tolerance
                elif direction == "decrease":
                    target_attenuation_delta = delta
                    direction_contradiction |= delta > direction_tolerance

        maximum_invariant_delta = max(invariant_deltas, default=float("nan"))
        invariance_failed = (
            math.isfinite(maximum_invariant_delta)
            and maximum_invariant_delta > invariance_tolerance
        )
        row.update(
            {
                "counterfactual_target_attenuation_delta": target_attenuation_delta,
                "counterfactual_max_invariant_abs_delta": maximum_invariant_delta,
                "counterfactual_invariance_failed": invariance_failed,
                "spatial_contradiction": spatial_contradiction,
                "direction_contradiction": direction_contradiction,
                "contradiction": spatial_contradiction or direction_contradiction,
            }
        )

        if occlusion_map is not None and counterfactuals is not None:
            row["evidence_priority"] = "occlusion+counterfactual"
        elif occlusion_map is not None:
            row["evidence_priority"] = "occlusion"
        elif counterfactuals is not None:
            row["evidence_priority"] = "counterfactual"
        elif gradcam_map is not None:
            row["evidence_priority"] = "gradcam"
        else:
            row["evidence_priority"] = "none"
        rows.append(row)
    return rows
