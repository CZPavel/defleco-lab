"""Surface LAB display and lossless, non-overwriting export of a result snapshot.

Images are display RGB8; quantitative float32 response is saved separately as NPY.
Display mapping never changes the response or the input raster.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from defleco_lab.processing.base import ProcessingResult

from .visualization import COLORMAPS, RangeMode, VisualizationSettings, VisualizationTransform


def _settings(settings: dict) -> VisualizationSettings:
    mode = settings.get("range", "percentile")
    if mode not in {"auto", "fixed", "percentile"}:
        raise ValueError("Unknown display range")
    low, high = float(settings.get("low", 0)), float(settings.get("high", 1))
    pl, ph = float(settings.get("percentile_low", 1)), float(settings.get("percentile_high", 99))
    opacity = float(settings.get("opacity", 0.5))
    enhancement_gain = float(settings.get("enhancement_gain", 1.0))
    if settings.get("overlay_base", "original") not in {"original", "enhanced"}:
        raise ValueError("Unknown overlay base")
    if settings.get("colormap", "Turbo") not in COLORMAPS:
        raise ValueError("Unknown heatmap palette")
    if settings.get("enhanced_polarity", "signed") not in {"signed", "bright", "dark", "absolute"}:
        raise ValueError("Unknown enhanced polarity")
    if not np.isfinite(enhancement_gain) or not 0 <= enhancement_gain <= 4:
        raise ValueError("Enhancement gain must be finite and between 0 and 4")
    if not np.isfinite([low, high, pl, ph, opacity]).all():
        raise ValueError("Display settings must be finite")
    if not 0 <= pl < ph <= 100 or not 0 <= opacity <= 1:
        raise ValueError("Invalid percentile range or opacity")
    if mode == "fixed" and low >= high:
        raise ValueError("Fixed low must be below high")
    return VisualizationSettings(
        range_mode={
            "auto": RangeMode.AUTO,
            "fixed": RangeMode.MANUAL,
            "percentile": RangeMode.PERCENTILE,
        }[mode],
        percentile_low=pl,
        percentile_high=ph,
        manual_minimum=low,
        manual_maximum=high,
        heatmap=settings.get("mode", "grayscale") in {"heatmap", "overlay"},
        colormap=settings.get("colormap", "Turbo"),
        overlay_alpha=1.0,
    )


def original_rgb(image: np.ndarray) -> np.ndarray:
    from defleco_lab.processing.surface import normalize_input

    normalized, _ = normalize_input(image)
    values = np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(values, cv2.COLOR_GRAY2RGB if values.ndim == 2 else cv2.COLOR_BGR2RGB)


def _normalized_source(image, result):
    """Prefer the worker's full-source normalization, then ROI/preview transform."""
    source = result.intermediates.get("_source_normalized")
    if source is not None:
        if source.shape != image.shape:
            raise ValueError("Normalized source coordinates must match input")
        return source
    from defleco_lab.processing.surface import normalize_input

    return normalize_input(image)[0]


def _response(result: ProcessingResult, settings: dict) -> np.ndarray:
    response = np.asarray(result.primary)
    if response.ndim != 2 or not np.isfinite(response).all():
        raise ValueError("Response must be finite HxW")
    polarity = settings.get("polarity", "signed")
    if polarity == "signed":
        return response
    if polarity == "bright":
        return np.maximum(response, 0)
    if polarity == "dark":
        return np.maximum(-response, 0)
    if polarity == "absolute":
        return np.abs(response)
    raise ValueError("Unknown response polarity")


def _enhanced_rgb(image, result, response, settings):
    source = _normalized_source(image, result)
    enhanced = result.intermediates.get("enhanced")
    if enhanced is None:
        if "enhanced_polarity" in settings:
            response = _response(result, {"polarity": settings["enhanced_polarity"]})
        # Signed feature maps are added to normalized luminance, display only.
        enhanced = source + (response[..., None] if source.ndim == 3 else response)
    gain = float(settings.get("enhancement_gain", 1.0))
    if gain != 1.0:
        baseline = source
        if enhanced.ndim == 2 and baseline.ndim == 3:
            baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        enhanced = baseline + gain * (enhanced - baseline)
    a = np.rint(np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB if a.ndim == 2 else cv2.COLOR_BGR2RGB)


def render_surface(image: np.ndarray, result: ProcessingResult, settings: dict) -> np.ndarray:
    display = _settings(settings)
    response = _response(result, settings)
    if response.shape != image.shape[:2]:
        raise ValueError("Input and response coordinates must match")
    mode = settings.get("mode", "grayscale")
    if mode not in {"enhanced", "grayscale", "heatmap", "overlay"}:
        raise ValueError("Unknown Surface LAB display mode")
    if mode == "enhanced":
        return _enhanced_rgb(image, result, response, settings)
    rgb = VisualizationTransform().render(response, display).image
    if mode == "overlay":
        # Response-dependent alpha leaves zero feature strength transparent.
        gray = (
            VisualizationTransform()
            .render(
                response,
                VisualizationSettings(
                    range_mode=display.range_mode,
                    percentile_low=display.percentile_low,
                    percentile_high=display.percentile_high,
                    manual_minimum=display.manual_minimum,
                    manual_maximum=display.manual_maximum,
                    heatmap=False,
                    overlay_alpha=1.0,
                ),
            )
            .image[..., 0]
        )
        strength = gray.astype(np.float32) / 255
        if settings.get("polarity", "signed") == "signed":
            # Signed zero is transparent even when the display range straddles it.
            values = response[result.valid_mask] if result.valid_mask is not None else response
            if display.range_mode == RangeMode.MANUAL:
                maximum = max(abs(display.manual_minimum), abs(display.manual_maximum))
            elif display.range_mode == RangeMode.PERCENTILE:
                maximum = float(np.percentile(np.abs(values), display.percentile_high))
            else:
                maximum = float(np.max(np.abs(values)))
            strength = np.clip(np.abs(response) / max(maximum, 1e-12), 0, 1)
        alpha = strength[..., None] * float(settings.get("opacity", 0.5))
        base = (
            _enhanced_rgb(image, result, response, settings)
            if settings.get("overlay_base", "original") == "enhanced"
            else original_rgb(_normalized_source(image, result))
        )
        rgb = np.rint(base * (1 - alpha) + rgb * alpha).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def export_surface(path, image, result, metadata, settings, save_response=True) -> dict:
    """Save RGB8 PNG/TIFF + JSON + optional unmodified float32 NPY.

    Refuse every pre-existing destination, including the source file. Roll back only
    files exclusively created by this call if encoding/writing fails.
    """
    path = Path(path)
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValueError("Choose PNG or TIFF")
    source = metadata.get("source_filename")
    if source and Path(source).resolve() == path.resolve():
        raise ValueError("The original file cannot be overwritten")
    rgb = render_surface(image, result, settings)
    ok, encoded = cv2.imencode(path.suffix, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Image encoding failed")
    from defleco_lab.processing.surface import normalize_input

    _, input_metadata = normalize_input(image)
    paths = {"image": path, "sidecar": path.with_suffix(".json")}
    if save_response:
        paths["response"] = path.with_suffix(".npy")
    document = {
        "schema": "defleco.surface.result.v1",
        **metadata,
        "input": input_metadata,
        "diagnostics": result.diagnostics,
        "display_settings": settings,
        "export_image": "RGB uint8 display rendering",
        "response_dtype": "float32",
        "response_shape": list(result.primary.shape),
        "response_min": float(np.min(result.primary)),
        "response_max": float(np.max(result.primary)),
        "files": {key: value.name for key, value in paths.items()},
    }
    serialized = json.dumps(
        document, indent=2, ensure_ascii=False, default=_json_value, allow_nan=False
    ).encode("utf-8")
    created = []
    try:
        for key, target in paths.items():
            with target.open("xb") as handle:
                created.append(target)
                if key == "image":
                    handle.write(encoded.tobytes())
                elif key == "sidecar":
                    handle.write(serialized)
                else:
                    np.save(
                        handle, np.asarray(result.primary, dtype=np.float32), allow_pickle=False
                    )
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return paths


def save_preset(path, config, *, method_registry=None):
    validated = _validate_preset(config, method_registry=method_registry)
    Path(path).write_text(
        json.dumps(validated, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def load_preset(path, *, method_registry=None) -> dict:
    return _validate_preset(
        json.loads(Path(path).read_text(encoding="utf-8")), method_registry=method_registry
    )


def _validate_preset(config, *, method_registry=None):
    from defleco_lab.processing.surface import surface_registry

    selected_registry = method_registry if method_registry is not None else surface_registry
    if not isinstance(config, dict):
        raise TypeError("Preset must be an object")
    method_id = config.get("method_id")
    if method_id not in {info.id for info in selected_registry.infos()}:
        raise ValueError("Unknown Surface LAB method")
    params = config.get("parameters", {})
    if not isinstance(params, dict):
        raise TypeError("Preset parameters must be an object")
    # Parameter validation must not trigger expensive image processing.
    selected_registry.create(method_id, **params)
    settings = config.get("display_settings", {})
    _settings(settings)
    if settings.get("mode", "grayscale") not in {"enhanced", "grayscale", "heatmap", "overlay"}:
        raise ValueError("Invalid display mode")
    if settings.get("polarity", "signed") not in {"signed", "bright", "dark", "absolute"}:
        raise ValueError("Invalid polarity")
    return {"schema": "defleco.surface.preset.v1", **config}
