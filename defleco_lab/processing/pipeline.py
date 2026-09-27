"""Reusable numeric pre/post-processing for Defleco LAB.

These transforms are deliberately separate from the image-analysis method and from
display-only colour/alpha mapping.  The defaults are mathematical no-ops.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .preprocessing import gray32


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    contrast: float = 1.0
    brightness: float = 0.0
    gamma: float = 1.0
    blur_sigma: float = 0.0
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PostprocessingSettings:
    local_background_sigma: float = 0.0
    absolute: bool = False
    smooth_sigma: float = 0.0
    gain: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_preprocessing(
    image: np.ndarray, settings: PreprocessingSettings | dict[str, Any] | None
) -> np.ndarray:
    """Prepare one frame for a processing method.

    Contrast is centred around mid-grey.  Brightness is an additive 8-bit-level
    offset.  Gamma is applied in the bounded 0..255 intensity domain.  CLAHE is
    optional and therefore never silently changes the baseline.
    """

    cfg = _pre(settings)
    a = gray32(image).copy()
    if cfg.contrast != 1.0 or cfg.brightness != 0.0:
        a = (a - 127.5) * float(cfg.contrast) + 127.5 + float(cfg.brightness)
    a = np.clip(a, 0.0, 255.0)

    if cfg.gamma != 1.0:
        gamma = max(float(cfg.gamma), 1e-6)
        a = np.power(a / 255.0, 1.0 / gamma, dtype=np.float32) * 255.0

    if cfg.clahe:
        grid = max(2, int(cfg.clahe_grid))
        clahe = cv2.createCLAHE(
            clipLimit=max(0.01, float(cfg.clahe_clip)),
            tileGridSize=(grid, grid),
        )
        a = clahe.apply(np.clip(a, 0, 255).astype(np.uint8)).astype(np.float32)

    if cfg.blur_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(cfg.blur_sigma))

    return a.astype(np.float32, copy=False)


def apply_postprocessing(
    response: np.ndarray, settings: PostprocessingSettings | dict[str, Any] | None
) -> np.ndarray:
    """Apply optional numeric response cleanup without presentation mapping."""

    cfg = _post(settings)
    a = np.asarray(response, dtype=np.float32).copy()

    if cfg.local_background_sigma > 0:
        background = cv2.GaussianBlur(a, (0, 0), float(cfg.local_background_sigma))
        a = a - background

    if cfg.absolute:
        a = np.abs(a)

    if cfg.smooth_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(cfg.smooth_sigma))

    if cfg.gain != 1.0:
        a = a * float(cfg.gain)

    return a.astype(np.float32, copy=False)


def _pre(settings: PreprocessingSettings | dict[str, Any] | None) -> PreprocessingSettings:
    if settings is None:
        return PreprocessingSettings()
    if isinstance(settings, PreprocessingSettings):
        return settings
    allowed = PreprocessingSettings.__dataclass_fields__
    return PreprocessingSettings(**{key: value for key, value in settings.items() if key in allowed})


def _post(
    settings: PostprocessingSettings | dict[str, Any] | None,
) -> PostprocessingSettings:
    if settings is None:
        return PostprocessingSettings()
    if isinstance(settings, PostprocessingSettings):
        return settings
    allowed = PostprocessingSettings.__dataclass_fields__
    return PostprocessingSettings(**{key: value for key, value in settings.items() if key in allowed})
