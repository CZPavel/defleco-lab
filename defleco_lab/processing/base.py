"""Stable processing plug-in contract."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MethodInfo:
    id: str
    name: str
    category: str
    required_frames: int
    supports_motion_compensation: bool
    description: str
    recommended_use: str = ""
    limitations: str = ""
    references: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingResult:
    primary: np.ndarray
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)
    overlays: list[dict[str, Any]] = field(default_factory=list)
    valid_mask: np.ndarray | None = None
    processing_time_ms: float = 0.0


class ProcessingMethod(ABC):
    info: MethodInfo

    def __init__(self, **parameters: Any):
        defaults = {
            k: (v.get("default") if isinstance(v, dict) else v)
            for k, v in self.info.parameters.items()
        }
        self.parameters = defaults | parameters

    def process(self, frames: Sequence[np.ndarray], **context: Any) -> ProcessingResult:
        required = self.history_requirement()
        if len(frames) < required:
            raise ValueError(f"{self.info.id} requires {required} frames")
        started = time.perf_counter_ns()
        result = self._process(frames, **context)
        result.processing_time_ms = (time.perf_counter_ns() - started) / 1e6
        return result

    def history_requirement(self) -> int:
        """Return source-history span, including method-local frame stride."""
        stride = max(1, int(self.parameters.get("stride", 1)))
        if "window" in self.parameters:
            return 1 + (max(1, int(self.parameters["window"])) - 1) * stride
        if "pairs" in self.parameters:
            return 1 + max(1, int(self.parameters["pairs"])) * stride
        return 1 + (self.info.required_frames - 1) * stride

    @abstractmethod
    def _process(self, frames: Sequence[np.ndarray], **context: Any) -> ProcessingResult: ...
