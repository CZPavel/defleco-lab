"""Camera-independent frame and derived motion metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FramePacket:
    image: np.ndarray
    frame_id: int
    host_timestamp_ns: int
    camera_timestamp: int | None = None
    block_id: int | None = None
    exposure_us: float | None = None
    gain: float | None = None
    pixel_format: str | None = None
    source: str = "unknown"
    estimated_motion_px: float | None = None
    motion_quality: float | None = None
    cumulative_position_px: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_owned(self) -> FramePacket:
        return FramePacket(
            image=self.image.copy(),
            frame_id=self.frame_id,
            host_timestamp_ns=self.host_timestamp_ns,
            camera_timestamp=self.camera_timestamp,
            block_id=self.block_id,
            exposure_us=self.exposure_us,
            gain=self.gain,
            pixel_format=self.pixel_format,
            source=self.source,
            estimated_motion_px=self.estimated_motion_px,
            motion_quality=self.motion_quality,
            cumulative_position_px=self.cumulative_position_px,
            metadata=dict(self.metadata),
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return session-safe metadata without unique device identifiers."""
        values = {
            "frame_id": self.frame_id,
            "host_timestamp_ns": self.host_timestamp_ns,
            "camera_timestamp": self.camera_timestamp,
            "block_id": self.block_id,
            "exposure_us": self.exposure_us,
            "gain": self.gain,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "pixel_format": self.pixel_format,
            "source": self.source,
            "estimated_motion_px": self.estimated_motion_px,
            "motion_quality": self.motion_quality,
            "cumulative_position_px": self.cumulative_position_px,
        }
        return {key: value for key, value in values.items() if value is not None}

    @classmethod
    def from_metadata(
        cls, image: np.ndarray, row: dict[str, Any], source: str | None = None
    ) -> FramePacket:
        def number(name: str, cast):
            value = row.get(name)
            if value in (None, "", "None"):
                return None
            try:
                return cast(float(value)) if cast is int else cast(value)
            except (TypeError, ValueError):
                return None

        return cls(
            image=image,
            frame_id=number("frame_id", int) or 0,
            host_timestamp_ns=number("host_timestamp_ns", int) or 0,
            camera_timestamp=number("camera_timestamp", int),
            block_id=number("block_id", int),
            exposure_us=number("exposure_us", float),
            gain=number("gain", float),
            pixel_format=row.get("pixel_format") or None,
            source=source or row.get("source") or "replay",
            estimated_motion_px=number("estimated_motion_px", float),
            motion_quality=number("motion_quality", float),
            cumulative_position_px=number("cumulative_position_px", float),
        )
