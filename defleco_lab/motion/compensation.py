"""Translation registration with explicit valid pixels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CompensationResult:
    aligned: np.ndarray
    valid_mask: np.ndarray
    transform: np.ndarray


def compensate_translation(
    historical, historical_position, current_position, axis="x", crop_common=False
):
    shift = float(current_position) - float(historical_position)
    dx, dy = (shift, 0.0) if axis == "x" else (0.0, shift)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    h, w = historical.shape[:2]
    aligned = cv2.warpAffine(
        historical,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpAffine(
        np.ones((h, w), np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if crop_common and valid.any():
        ys, xs = np.where(valid)
        aligned = aligned[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        valid = valid[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return CompensationResult(aligned, valid, matrix)


def compensate_ecc(historical, current, iterations=50, epsilon=1e-5):
    a = np.asarray(historical, np.float32)
    b = np.asarray(current, np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(epsilon))
    quality, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_TRANSLATION, criteria)
    aligned = cv2.warpAffine(
        historical,
        warp,
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.warpAffine(
        np.ones(a.shape, np.uint8),
        warp,
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    return CompensationResult(aligned, valid, warp), float(quality)
