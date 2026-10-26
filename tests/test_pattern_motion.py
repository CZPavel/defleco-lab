from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from defleco_lab.gui.pattern_motion import frame_motion_samples
from defleco_lab.gui.pattern_output import PatternSettings


def test_static_step_and_disabled_modes_keep_sharp_frames():
    settings = PatternSettings(mode="continuous", breathing_depth_percent=20)
    for changed in (
        replace(settings, mode="static"), replace(settings, mode="step"),
        replace(settings, motion_smoothing=False), replace(settings, family="speckle"),
        replace(settings, family="solid"),
        replace(settings, speed_deg_s=0, breathing_depth_percent=0),
    ):
        assert frame_motion_samples(changed, 10, 20, 1, 60) == ()


def test_motion_samples_preserve_breathing_and_wrap_periods():
    settings = PatternSettings(
        family="starburst", mode="continuous", speed_deg_s=20,
        phase_speed_cycles_s=1, breathing_depth_percent=20, breathing_hz=0.5,
    )
    samples = frame_motion_samples(settings, 0.05, 360.05, 0, 60)
    assert len(samples) == 4
    assert any(s.angle_deg > 359 for s in samples)
    assert any(s.phase_deg < 360 for s in samples)
    assert all(354 < s.phase_deg < 361 for s in samples)
    assert all(40 <= s.period_px <= 60 for s in samples)
    assert samples[0].period_px < samples[-1].period_px < 50
    assert frame_motion_samples(settings, 0, 0, 0, float("nan")) == frame_motion_samples(
        settings, 0, 0, 0, 60
    )


def test_temporal_samples_reduce_error_against_dense_exposure_reference():
    # Exact reported motion at a 4K edge. The dense oracle evaluates the original
    # timeline independently; it does not call the production sampler.
    settings = PatternSettings(mode="continuous", breathing_depth_percent=20, breathing_hz=0.5)
    t, rate = 1.137, 60.0
    x, y = np.linspace(-1080, 1080, 12000), 1800.0

    def coverage(angle, period):
        radians = math.radians(angle)
        cycles = (np.cos(radians) * x + np.sin(radians) * y) / period
        distance = np.abs((cycles - 0.25 + 0.5) % 1 - 0.5)
        footprint = (abs(math.cos(radians)) + abs(math.sin(radians))) / period
        return np.clip((0.25 - distance) / footprint + 0.5, 0, 1)

    dense = np.zeros_like(x)
    for index in range(128):
        time = t + ((index + 0.5) / 128 - 1) / rate
        dense += coverage(20 * time, 50 * (1 + 0.2 * math.sin(math.pi * time))) / 128
    samples = frame_motion_samples(settings, 20 * t, 0, math.pi * t, rate)
    smooth = np.mean([coverage(s.angle_deg, s.period_px) for s in samples], axis=0)
    sharp = coverage(20 * t, 50 * (1 + 0.2 * math.sin(math.pi * t)))
    assert np.mean(abs(smooth - dense)) < np.mean(abs(sharp - dense)) / 4
    assert np.mean(abs(smooth - dense)) < 0.035
