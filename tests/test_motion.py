import cv2
import numpy as np

from defleco_lab.motion import (
    CumulativePosition,
    MotionROI,
    PhaseMotionEstimator,
    compensate_translation,
)


def texture(seed=4):
    return np.random.default_rng(seed).normal(127, 35, (128, 192)).astype(np.float32)


def shift(a, dx=0, dy=0):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]))


def test_phase_correlation_recovers_known_positive_x_shift():
    a = texture()
    r = PhaseMotionEstimator().estimate(a, shift(a, 5), [MotionROI(16, 16, 144, 96)])
    assert r.shift_px == pytest.approx(5, abs=0.3)


def test_phase_correlation_recovers_negative_x_shift():
    a = texture()
    r = PhaseMotionEstimator().estimate(a, shift(a, -4), [MotionROI(16, 16, 144, 96)])
    assert r.shift_px == pytest.approx(-4, abs=0.3)


def test_selected_y_component_does_not_corrupt_x_only_mode():
    a = texture()
    r = PhaseMotionEstimator(axis="x").estimate(a, shift(a, 3, 7), [MotionROI(20, 20, 120, 80)])
    assert r.shift_px == pytest.approx(3, abs=0.4)


def test_multiple_roi_median_rejects_one_bad_roi(monkeypatch):
    a = texture()
    b = shift(a, 4)
    b[16:96, 96:144] = np.random.default_rng(9).normal(127, 35, (80, 48))
    rois = [MotionROI(i * 48, 16, 48, 80, str(i)) for i in range(3)]
    r = PhaseMotionEstimator(minimum_q=-1).estimate(a, b, rois)
    assert r.shift_px == pytest.approx(4, abs=0.5)


def test_motion_compensation_reduces_error_between_translated_images():
    a = texture()
    b = shift(a, 6)
    result = compensate_translation(a, 0, 6)
    raw = np.mean(np.abs(a - b))
    aligned = np.mean(np.abs(result.aligned[result.valid_mask] - b[result.valid_mask]))
    assert aligned < raw * 0.2


def test_valid_mask_borders_are_excluded_after_translation():
    r = compensate_translation(np.ones((20, 30), np.uint8), 0, 5)
    assert not r.valid_mask[:, :5].any() and r.valid_mask[:, 5:].all()


def test_cumulative_position_works_over_multiple_frames():
    p = CumulativePosition()
    assert [p.update(x) for x in (2, -1, 3)] == [2, 1, 4]


import pytest
