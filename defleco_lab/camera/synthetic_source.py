from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import cv2
import numpy as np

from .frame_packet import FramePacket


@dataclass(slots=True)
class SyntheticConfig:
    width: int = 960
    height: int = 640
    pattern: str = "fringes"
    period_px: float = 32.0
    shift_per_frame_px: float = 1.5
    axis: str = "x"
    local_deformation: bool = True
    seed: int = 7


class SyntheticSource:
    """Deterministic hardware-free source for software validation."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.frame_id = 0
        self._rng = np.random.default_rng(self.config.seed)
        self._base = self._make_base()

    def _make_base(self) -> np.ndarray:
        c = self.config
        yy, xx = np.mgrid[: c.height, : c.width].astype(np.float32)
        if c.pattern == "checker":
            image = (((xx // c.period_px) + (yy // c.period_px)) % 2) * 210 + 22
        elif c.pattern == "speckle":
            image = self._rng.normal(127, 45, (c.height, c.width))
            image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
        elif c.pattern == "grid":
            image = 127 + 55 * np.sin(2 * np.pi * xx / c.period_px)
            image += 55 * np.sin(2 * np.pi * yy / c.period_px)
        else:
            image = 127 + 100 * np.sin(2 * np.pi * xx / c.period_px)
        texture = self._rng.normal(0, 7, image.shape)
        return np.clip(image + texture, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self.frame_id = 0

    def next_frame(self) -> FramePacket:
        c = self.config
        shift = self.frame_id * c.shift_per_frame_px
        dx, dy = (shift, 0.0) if c.axis == "x" else (0.0, shift)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(
            self._base,
            matrix,
            (c.width, c.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        if c.local_deformation and self.frame_id:
            yy, xx = np.mgrid[: c.height, : c.width].astype(np.float32)
            cx, cy = c.width * 0.62, c.height * 0.48
            radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
            bump = 5.0 * np.exp(-radius2 / (2 * (c.width * 0.07) ** 2))
            map_x = xx + bump.astype(np.float32)
            image = cv2.remap(image, map_x, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        packet = FramePacket(
            image=image.copy(),
            frame_id=self.frame_id,
            host_timestamp_ns=perf_counter_ns(),
            pixel_format="Mono8",
            source="synthetic",
        )
        self.frame_id += 1
        return packet
