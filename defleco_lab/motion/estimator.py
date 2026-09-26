"""One-axis, multi-ROI image-motion estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MotionROI:
    x: int
    y: int
    width: int
    height: int
    name: str = "ROI"
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ROIMotion:
    roi: MotionROI
    dx: float
    dy: float
    selected_shift: float
    texture: float
    quality: float
    valid: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    shift_px: float
    quality: float
    valid_count: int
    valid: bool
    roi_results: tuple[ROIMotion, ...]


class PhaseMotionEstimator:
    def __init__(
        self,
        axis="x",
        minimum_texture=3.0,
        minimum_q=0.08,
        deadband=0.0,
        max_shift=50.0,
        preprocessing="none",
    ):
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        self.axis = axis
        self.minimum_texture = float(minimum_texture)
        self.minimum_q = float(minimum_q)
        self.deadband = float(deadband)
        self.max_shift = float(max_shift)
        self.preprocessing = preprocessing
        self._windows = {}

    def _prepare(self, a):
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        a = a.astype(np.float32)
        if self.preprocessing == "gaussian":
            a = cv2.GaussianBlur(a, (0, 0), 2)
        elif self.preprocessing == "gradient":
            a = cv2.magnitude(cv2.Scharr(a, cv2.CV_32F, 1, 0), cv2.Scharr(a, cv2.CV_32F, 0, 1))
        return a

    def estimate(self, previous, current, rois):
        a, b = self._prepare(previous), self._prepare(current)
        results = []
        for roi in rois:
            if not roi.enabled:
                continue
            p = a[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            c = b[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            if p.shape != c.shape or min(p.shape, default=0) < 16:
                results.append(ROIMotion(roi, 0, 0, 0, 0, 0, False, "ROI outside image"))
                continue
            texture = float(min(p.std(), c.std()))
            if texture < self.minimum_texture:
                results.append(ROIMotion(roi, 0, 0, 0, texture, 0, False, "low texture"))
                continue
            key = (p.shape[1], p.shape[0])
            win = self._windows.setdefault(key, cv2.createHanningWindow(key, cv2.CV_32F))
            shift, q = cv2.phaseCorrelate(p - p.mean(), c - c.mean(), win)
            dx, dy = map(float, shift)
            selected = dx if self.axis == "x" else dy
            valid = bool(
                np.isfinite(selected)
                and np.isfinite(q)
                and q >= self.minimum_q
                and abs(selected) <= self.max_shift
            )
            reason = "" if valid else "quality or displacement limit"
            if valid and abs(selected) < self.deadband:
                selected = 0.0
            results.append(ROIMotion(roi, dx, dy, selected, texture, float(q), valid, reason))
        valid = [r for r in results if r.valid]
        shift = float(np.median([r.selected_shift for r in valid])) if valid else 0.0
        quality = float(np.median([r.quality for r in valid])) if valid else 0.0
        return MotionEstimate(shift, quality, len(valid), bool(valid), tuple(results))


class CumulativePosition:
    def __init__(self, initial=0.0):
        self.position = float(initial)

    def reset(self, value=0.0):
        self.position = float(value)

    def update(self, shift_px, valid=True):
        if valid:
            self.position += float(shift_px)
        return self.position
