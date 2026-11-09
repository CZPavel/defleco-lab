"""Seven bounded, single-frame Surface Fusion experiments.

Responses are relative signed feature maps, not confidence or probabilities.
Only the source passes through normalize_input; signed intermediate arrays go
directly into the same float-domain kernels as the existing Surface methods.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy

import cv2
import numpy as np

from ..base import MethodInfo, ProcessingMethod, ProcessingResult
from ..registry import MethodRegistry
from . import kernels
from .methods import _gray, normalize_input

FUSION_VERSION = "gpixel-surface-fusion.v1"
NORMALIZATION = "p99(abs(response)), max fallback, clip [-1,1]; K06 raw unsharp delta"
fusion_registry = MethodRegistry()


def _number(default, minimum, maximum, *, step=None, visible=None, units=""):
    result = {"default": default, "minimum": minimum, "maximum": maximum}
    if step is not None:
        result["step"] = step
    if visible:
        result["visible_if"] = {visible: True}
    if units:
        result["units"] = units
    return result


def _choice(default, choices, visible=None):
    result = {"default": default, "choices": choices}
    if visible:
        result["visible_if"] = {visible: True}
    return result


CLAHE = {
    "clahe_enabled": {"default": True},
    "clahe_clip": _number(2.0, 0.1, 10.0, step=0.1, visible="clahe_enabled"),
    "clahe_grid": _number(8, 2, 32, visible="clahe_enabled"),
}
MORPH = {
    "morphology_enabled": {"default": True},
    "morphology_size": _number(31, 3, 63, step=2, visible="morphology_enabled", units="px"),
    "morphology_angle": _number(0.0, 0, 179, step=5, visible="morphology_enabled", units="deg"),
}
HESSIAN = {
    "hessian_enabled": {"default": True},
    "hessian_sigmas": {
        "default": [2.0, 4.0],
        "type": "list",
        "item_type": "float",
        "visible_if": {"hessian_enabled": True},
        "units": "px",
    },
    "hessian_view": _choice(
        "fused", ["fused", "scale_1", "scale_2", "scale_3", "scale_4"], "hessian_enabled"
    ),
}
MEDIAN = {
    "median_enabled": {"default": True},
    "median_size": _number(31, 3, 63, step=2, visible="median_enabled", units="px"),
}
LOCAL = {
    "local_enabled": {"default": True},
    "local_sigma": _number(4.0, 0.2, 20, step=0.2, visible="local_enabled", units="px"),
    "local_strength": _number(1.0, 0, 2, step=0.1, visible="local_enabled"),
}
FFT = {
    "fft_enabled": {"default": True},
    "fft_sigma": _number(4.0, 0.2, 20, step=0.2, visible="fft_enabled", units="px"),
    "fft_polarity": _choice("signed", ["signed", "absolute"], "fft_enabled"),
}
UNSHARP = {
    "unsharp_enabled": {"default": True},
    "unsharp_amount": _number(2.0, 0, 4, step=0.1, visible="unsharp_enabled"),
    "unsharp_sigma": _number(4.0, 0.2, 20, step=0.2, visible="unsharp_enabled", units="px"),
}
POLARITY = {"polarity": _choice("both", ["bright", "dark", "both"])}
INTENSITY = {"intensity": _number(0.5, 0, 2, step=0.1)}
RECIPES = {
    "fusion_k01": (
        "K01 – CLAHE + Directional Top-hat",
        "Directional structures and scratches",
        CLAHE | MORPH | POLARITY | INTENSITY,
    ),
    "fusion_k02": (
        "K02 – CLAHE + Multi-scale Hessian",
        "Thin contrasting lines at multiple scales",
        CLAHE | HESSIAN | POLARITY | INTENSITY,
    ),
    "fusion_k03": (
        "K03 – Median Residual + Hessian",
        "Local background suppression then ridges",
        MEDIAN
        | {"correction_strength": _number(1.0, 0, 2, step=0.1, visible="median_enabled")}
        | HESSIAN
        | POLARITY
        | INTENSITY,
    ),
    "fusion_k04": (
        "K04 – CLAHE + FFT High-pass",
        "Fine high-frequency structures",
        CLAHE | FFT | INTENSITY,
    ),
    "fusion_k05": (
        "K05 – Local Residual + Directional Top-hat",
        "Shading correction and directional features",
        LOCAL
        | MORPH
        | {"morphology_strength": _number(1.0, 0, 2, step=0.1, visible="morphology_enabled")}
        | POLARITY
        | INTENSITY,
    ),
    "fusion_k06": (
        "K06 – CLAHE + Unsharp Mask",
        "Readable enhanced-image baseline",
        CLAHE | UNSHARP | INTENSITY,
    ),
    "fusion_k07": (
        "K07 – GPixel Surface Fusion",
        "Combine three normalized feature branches",
        CLAHE
        | HESSIAN
        | MORPH
        | MEDIAN
        | {
            "hessian_weight": _number(0.40, 0, 2, step=0.05, visible="hessian_enabled"),
            "morphology_weight": _number(0.35, 0, 2, step=0.05, visible="morphology_enabled"),
            "median_weight": _number(0.25, 0, 2, step=0.05, visible="median_enabled"),
            "fusion_mode": _choice("Weighted Sum", ["Weighted Sum", "Maximum"]),
        }
        | POLARITY
        | INTENSITY,
    ),
}


def _selected(a, polarity):
    if polarity == "bright":
        return np.maximum(a, 0)
    if polarity == "dark":
        return np.minimum(a, 0)
    return a


def _normalize_response(a):
    """Bound response without centering or flipping its meaningful sign."""
    values = np.asarray(a, np.float32)
    magnitude = np.abs(values)
    percentile = float(np.percentile(magnitude, 99))
    peak = float(np.max(magnitude))
    scale = percentile if percentile > 1e-6 else peak
    if scale <= 1e-6:
        return np.zeros(values.shape, np.float32), 0.0
    return np.clip(values / scale, -1, 1).astype(np.float32), scale


def _max_abs(arrays):
    stack = np.stack(arrays)
    return np.take_along_axis(stack, np.argmax(np.abs(stack), axis=0)[None, ...], axis=0)[0]


def _scaled_size(size, scale):
    return max(3, min(63, round(size * scale) | 1))


def _scaled_sigma(sigma, scale):
    return max(0.2, float(sigma) * scale)


class _FusionMethod(ProcessingMethod):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        for key, value in self.parameters.items():
            spec = self.info.parameters[key]
            default = spec["default"]
            if isinstance(default, bool):
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"{key}: expected boolean")
            elif isinstance(default, list):
                if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 4:
                    raise ValueError(f"{key}: expected 1–4 scales")
                if any(
                    isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or not 1 <= v <= 8
                    for v in value
                ):
                    raise ValueError(f"{key}: scales must be finite values 1–8 px")
            elif "choices" in spec:
                if value not in spec["choices"]:
                    raise ValueError(f"{key}: invalid choice")
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not spec["minimum"] <= value <= spec["maximum"]
            ):
                raise ValueError(f"{key}: out of range")
            elif isinstance(default, int) and (
                int(value) != value or (key.endswith("_size") and int(value) % 2 != 1)
            ):
                raise ValueError(
                    f"{key}: expected odd integer"
                    if key.endswith("_size")
                    else f"{key}: expected integer"
                )
        view = self.parameters.get("hessian_view", "fused")
        if view != "fused" and int(view[-1]) > len(self.parameters["hessian_sigmas"]):
            raise ValueError("Selected Hessian scale is not present")

    def _process(self, frames, **context):
        original, metadata = normalize_input(frames[-1])
        p = self.parameters
        scale = float(context.get("spatial_scale", 1.0))
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("spatial_scale must be in (0,1]")
        gray = _gray(original)
        shape = gray.shape
        base = original.copy()
        base_gray = gray.copy()
        response = np.zeros(shape, np.float32)
        intermediates = {}
        diagnostics = metadata | {
            "recipe_id": self.info.id,
            "recipe_version": FUSION_VERSION,
            "normalization": NORMALIZATION,
            "spatial_scale": scale,
        }
        if p.get("clahe_enabled", False):
            base_gray = kernels.clahe_luminance(gray, p["clahe_clip"], p["clahe_grid"])
            base = original + (
                (base_gray - gray)[..., None] if original.ndim == 3 else (base_gray - gray)
            )

        def norm(a, name):
            out, limit = _normalize_response(a)
            diagnostics[name + "_normalization_scale"] = limit
            diagnostics[name + "_normalized_clip_fraction"] = (
                float(np.mean(np.abs(a) > limit)) if limit > 0 else 0.0
            )
            return out

        def hessian(source):
            sigmas = [_scaled_sigma(s, scale) for s in p["hessian_sigmas"]]
            diagnostics["effective_hessian_sigmas"] = json.dumps(sigmas)
            view = p["hessian_view"]
            if view != "fused":
                index = int(view[-1]) - 1
                return norm(
                    _selected(kernels.hessian_signed(source, sigmas[index]), p["polarity"]),
                    f"hessian_scale_{index + 1}",
                )
            responses = [
                norm(
                    _selected(kernels.hessian_signed(source, sigma), p["polarity"]),
                    f"hessian_scale_{i + 1}",
                )
                for i, sigma in enumerate(sigmas)
            ]
            return _max_abs(responses)

        def morph(source):
            size = _scaled_size(p["morphology_size"], scale)
            diagnostics["effective_morphology_size"] = size
            bright = dark = None
            if p["polarity"] in {"bright", "both"}:
                bright = kernels.directional_morph(
                    source, size, p["morphology_angle"], cv2.MORPH_TOPHAT
                )
            if p["polarity"] in {"dark", "both"}:
                dark = kernels.directional_morph(
                    source, size, p["morphology_angle"], cv2.MORPH_BLACKHAT
                )
            raw = bright if dark is None else -dark if bright is None else bright - dark
            return norm(raw, "morphology")

        def median(source):
            size = _scaled_size(p["median_size"], scale)
            diagnostics["effective_median_size"] = size
            return kernels.median_residual(source, size, float_precision=True)

        rid = self.info.id
        if rid == "fusion_k01":
            if p["morphology_enabled"]:
                response = morph(base_gray)
        elif rid in {"fusion_k02", "fusion_k03"}:
            source = base_gray
            if rid == "fusion_k03" and p["median_enabled"]:
                residual = median(gray)
                source = gray + p["correction_strength"] * (residual - gray)
                intermediates["median_residual_signed"] = residual.astype(np.float32)
            if p["hessian_enabled"]:
                response = hessian(source)
            elif rid == "fusion_k03" and p["median_enabled"]:
                response = np.clip(
                    norm(_selected(residual, p["polarity"]), "median") * p["correction_strength"],
                    -1,
                    1,
                ).astype(np.float32)
        elif rid == "fusion_k04":
            if p["fft_enabled"]:
                sigma = _scaled_sigma(p["fft_sigma"], scale)
                diagnostics["effective_fft_sigma"] = sigma
                raw = kernels.fft_highpass(base_gray, sigma)
                response = norm(np.abs(raw) if p["fft_polarity"] == "absolute" else raw, "fft")
        elif rid == "fusion_k05":
            source = gray
            if p["local_enabled"]:
                sigma = _scaled_sigma(p["local_sigma"], scale)
                diagnostics["effective_local_sigma"] = sigma
                residual = kernels.local_residual(gray, sigma)
                source = gray + p["local_strength"] * (residual - gray)
                intermediates["local_residual_signed"] = residual.astype(np.float32)
            if p["morphology_enabled"]:
                response = np.clip(morph(source) * p["morphology_strength"], -1, 1).astype(
                    np.float32
                )
            elif p["local_enabled"]:
                response = np.clip(
                    norm(_selected(residual, p["polarity"]), "local") * p["local_strength"], -1, 1
                ).astype(np.float32)
        elif rid == "fusion_k06":
            if p["unsharp_enabled"]:
                sigma = _scaled_sigma(p["unsharp_sigma"], scale)
                diagnostics["effective_unsharp_sigma"] = sigma
                sharpened_raw = kernels.unsharp_raw(base, sigma, p["unsharp_amount"])
                diagnostics["unsharp_clip_fraction"] = float(
                    np.mean((sharpened_raw < 0) | (sharpened_raw > 1))
                )
                delta = sharpened_raw - base
                response = _gray(delta).astype(np.float32)
                base = sharpened_raw
        elif rid == "fusion_k07":
            branches = []
            for branch in ("hessian", "morphology", "median"):
                weight = float(p[branch + "_weight"])
                if not p[branch + "_enabled"] or weight <= 0:
                    continue
                values = (
                    hessian(base_gray)
                    if branch == "hessian"
                    else morph(base_gray)
                    if branch == "morphology"
                    else norm(_selected(median(base_gray), p["polarity"]), "median")
                )
                branches.append((branch, weight, values))
                intermediates["branch_" + branch] = values
            if branches:
                total = sum(weight for _, weight, _ in branches)
                weighted = [values * (weight / total) for _, weight, values in branches]
                response = (
                    sum(weighted) if p["fusion_mode"] == "Weighted Sum" else _max_abs(weighted)
                ).astype(np.float32)
            diagnostics["active_branches"] = ",".join(name for name, _, _ in branches)
        else:
            raise AssertionError(rid)

        response = np.asarray(response, np.float32)
        # K06's unsharp delta is already included in base; don't add it twice.
        response_for_image = np.zeros_like(response) if rid == "fusion_k06" else response
        delta = response_for_image[..., None] if original.ndim == 3 else response_for_image
        enhanced_unclipped = original + float(p["intensity"]) * (base - original + delta)
        diagnostics["enhanced_clip_fraction"] = float(
            np.mean((enhanced_unclipped < 0) | (enhanced_unclipped > 1))
        )
        enhanced = np.clip(enhanced_unclipped, 0, 1).astype(np.float32)
        if not np.isfinite(response).all() or not np.isfinite(enhanced).all():
            raise ValueError("Fusion produced non-finite output")
        keep = context.get("keep_intermediates", True)
        if keep:
            intermediates["enhanced"] = enhanced
        else:
            intermediates = {}
        return ProcessingResult(response, intermediates, diagnostics=diagnostics)


def recipe_metadata(method_id, params):
    """Validate and describe a reproducible preset or exported experiment."""
    method = fusion_registry.create(method_id, **params)
    values = deepcopy(method.parameters)
    enabled = [
        name.removesuffix("_enabled")
        for name, enabled in values.items()
        if name.endswith("_enabled") and enabled
    ]
    active = [
        name
        for name in enabled
        if method_id != "fusion_k07"
        or name not in {"hessian", "morphology", "median"}
        or values[name + "_weight"] > 0
    ]
    return {
        "recipe_id": method_id,
        "recipe_version": FUSION_VERSION,
        "active_steps": active,
        "enabled_steps": enabled,
        "parameters": values,
        "step_parameters": deepcopy(values),
        "weights": {k: v for k, v in values.items() if k.endswith("_weight")},
        "normalization": NORMALIZATION,
        "intensity": values["intensity"],
    }


for _id, (_name, _description, _schema) in RECIPES.items():
    _info = MethodInfo(
        _id,
        _name,
        "GPixel Surface Fusion",
        1,
        False,
        _description,
        recommended_use="Experimental visual enhancement of static-light images.",
        limitations="Relative responses; not calibrated defect scores or confidence.",
        parameters=deepcopy(_schema),
    )
    fusion_registry.register(
        type("Fusion_" + _id, (_FusionMethod,), {"info": _info, "__module__": __name__})
    )

__all__ = ["FUSION_VERSION", "fusion_registry", "recipe_metadata"]
