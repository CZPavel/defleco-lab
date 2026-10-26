"""One-display-frame temporal sampling; no camera or GPU dependencies."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pattern_output import PatternSettings


def frame_motion_samples(
    settings: PatternSettings,
    angle: float,
    phase: float,
    breathing_phase: float,
    refresh_hz: float,
    count: int = 4,
) -> tuple[PatternSettings, ...]:
    """Midpoint integration over the preceding refresh interval, not a queued frame.

    Speed and breathing use the original timeline even if a previous frame was
    late. Static/stepped experiments retain their original sharp snapshots.
    """
    if (
        settings.mode != "continuous" or not settings.motion_smoothing
        or settings.family in {"solid", "speckle"}
        or not (settings.speed_deg_s or settings.phase_speed_cycles_s
                or settings.breathing_depth_percent)
    ):
        return ()
    rate = refresh_hz if math.isfinite(refresh_hz) and refresh_hz > 1 else 60.0
    count = min(8, max(1, int(count)))
    depth = settings.breathing_depth_percent / 100.0
    phase_rate = settings.phase_speed_cycles_s * 360.0
    rotation_rate = settings.speed_deg_s
    if settings.family == "rings" and not phase_rate:
        phase_rate, rotation_rate = rotation_rate, 0.0
    period = 720.0 if settings.family == "starburst" else 360.0
    result = []
    for index in range(count):
        offset = ((index + 0.5) / count - 1.0) / rate
        breathing = breathing_phase + 2.0 * math.pi * settings.breathing_hz * offset
        result.append(replace(
            settings,
            angle_deg=(angle + rotation_rate * offset) % 360.0,
            phase_deg=(phase + phase_rate * offset) % period,
            period_px=settings.period_px * max(0.05, 1.0 + depth * math.sin(breathing)),
        ))
    return tuple(result)
