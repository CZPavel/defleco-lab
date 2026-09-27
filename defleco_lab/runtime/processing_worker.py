from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from threading import Condition
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PySide6 import QtCore

from defleco_lab.motion import MotionROI, PhaseMotionEstimator, compensate_translation
from defleco_lab.processing.pipeline import apply_postprocessing, apply_preprocessing
from defleco_lab.processing.registry import registry


@dataclass(slots=True)
class ProcessingRequest:
    method_id: str
    parameters: dict[str, Any]
    frames: list
    scale: float
    stride: int
    preprocessing: dict[str, Any] | None = None
    postprocessing: dict[str, Any] | None = None
    motion_compensation: bool = False
    motion_axis: str = "x"
    motion_rois: list[dict[str, Any]] | None = None
    analysis_rois: list[dict[str, Any]] | None = None
    motion_mode: str = "auto"
    manual_shift_px: float = 0.0
    motion_preprocessing: str = "none"
    minimum_texture: float = 3.0
    minimum_q: float = 0.08
    deadband: float = 0.0
    max_shift: float = 50.0
    frame_id: int = -1
    motion_frames: list[Any] | None = None
    keep_intermediates: bool = False
    generation: int = 0


class ProcessingWorker(QtCore.QThread):
    """Latest-wins processor: stale display work is dropped, never queued indefinitely."""

    # Result payloads can be very large at multi-megapixel resolutions. Do not
    # queue them directly through Qt; signal only that a newest result is ready.
    resultAvailable = QtCore.Signal()
    completed = QtCore.Signal(object, float, float, int, object, int)  # legacy API
    failed = QtCore.Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._condition = Condition()
        self._request: ProcessingRequest | None = None
        self._stopping = False
        self.dropped = 0
        self._cumulative_position = 0.0
        self._last_motion_frame_id = -1
        self._last_motion_image: np.ndarray | None = None
        self._generation = 0
        self._latest_result: tuple | None = None
        self._result_notification_pending = False
        self.result_dropped = 0

    def submit(self, request: ProcessingRequest) -> None:
        with self._condition:
            if self._request is not None:
                self.dropped += 1
            request.generation = self._generation
            self._request = request
            self._condition.notify()

    def reset_state(self) -> None:
        """Start a new experiment and invalidate queued results from the old one."""
        with self._condition:
            self._generation += 1
            self._request = None
            self._cumulative_position = 0.0
            self._last_motion_frame_id = -1
            self._last_motion_image = None
            self._latest_result = None
            self._result_notification_pending = False

    @staticmethod
    def _calculate_cumulative_motion(
        request: ProcessingRequest,
        motion: dict[str, Any],
        cumulative_position: float,
        last_motion_frame_id: int,
        last_motion_image: np.ndarray | None,
    ) -> tuple[float, int, np.ndarray | None]:
        """Integrate every available source-frame pair, including skipped display work."""
        packets = sorted(request.motion_frames or [], key=lambda packet: packet.frame_id)
        if not request.motion_compensation or not packets:
            motion["cumulative"] = cumulative_position
            return cumulative_position, last_motion_frame_id, last_motion_image
        packets = [packet for packet in packets if packet.frame_id > last_motion_frame_id]
        previous = last_motion_image
        if previous is None:
            first = packets.pop(0) if packets else None
            if first is not None:
                previous = first.image
                last_motion_frame_id = first.frame_id
                last_motion_image = first.image.copy()
        if request.motion_mode == "auto":
            roi_specs = request.motion_rois or []
            rois = [MotionROI(**roi) for roi in roi_specs]
            estimator = PhaseMotionEstimator(
                axis=request.motion_axis,
                preprocessing=request.motion_preprocessing,
                minimum_texture=request.minimum_texture,
                minimum_q=request.minimum_q,
                deadband=request.deadband,
                max_shift=request.max_shift,
            )
        for packet in packets:
            if previous is None:
                previous = packet.image
                continue
            if request.motion_mode == "manual":
                shift = request.manual_shift_px
            else:
                estimate = estimator.estimate(previous, packet.image, rois)
                if not estimate.valid:
                    raise ValueError("Motion estimate invalid in skipped-frame history")
                shift = estimate.shift_px
                motion.update(
                    shift=shift,
                    quality=estimate.quality,
                    valid=estimate.valid_count,
                    roi_results=estimate.roi_results,
                )
            cumulative_position += float(shift)
            previous = packet.image
            last_motion_frame_id = packet.frame_id
            last_motion_image = packet.image.copy()
        motion["cumulative"] = cumulative_position
        return cumulative_position, last_motion_frame_id, last_motion_image

    def _accumulate_motion(self, request: ProcessingRequest, motion: dict[str, Any]) -> None:
        """Synchronous helper retained for focused tests and non-threaded callers."""
        state = self._calculate_cumulative_motion(
            request,
            motion,
            self._cumulative_position,
            self._last_motion_frame_id,
            self._last_motion_image,
        )
        self._cumulative_position, self._last_motion_frame_id, self._last_motion_image = state


    def take_latest_result(self) -> tuple | None:
        """Take the newest completed result and release the notification gate."""
        with self._condition:
            payload = self._latest_result
            self._latest_result = None
            self._result_notification_pending = False
            return payload

    def _publish_result(self, payload: tuple) -> None:
        """Keep at most one heavy completed result waiting for the GUI."""
        notify = False
        with self._condition:
            if self._latest_result is not None:
                self.result_dropped += 1
            self._latest_result = payload
            if not self._result_notification_pending:
                self._result_notification_pending = True
                notify = True
        if notify:
            self.resultAvailable.emit()

    def stop(self) -> bool:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        return self.wait(30000)

    def run(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request, self._request = self._request, None
            assert request is not None
            try:
                frames = request.frames
                motion = {"shift": 0.0, "quality": 0.0, "valid": 0, "mask": None}
                valid_mask = None
                if request.motion_compensation:
                    positions = [0.0]
                    estimates = []
                    if request.motion_mode == "manual":
                        for _ in range(len(frames) - 1):
                            positions.append(positions[-1] + request.manual_shift_px)
                        motion.update({"shift": request.manual_shift_px, "valid": 1})
                    else:
                        roi_specs = request.motion_rois or []
                        if not roi_specs:
                            raise ValueError(
                                "Draw and enable at least one Motion ROI before compensation"
                            )
                        rois = [MotionROI(**roi) for roi in roi_specs]
                        estimator = PhaseMotionEstimator(
                            axis=request.motion_axis,
                            preprocessing=request.motion_preprocessing,
                            minimum_texture=request.minimum_texture,
                            minimum_q=request.minimum_q,
                            deadband=request.deadband,
                            max_shift=request.max_shift,
                        )
                        for previous, current in pairwise(frames):
                            estimate = estimator.estimate(previous, current, rois)
                            if not estimate.valid:
                                raise ValueError(
                                    "Motion estimate invalid; inspect Motion ROI texture and Q"
                                )
                            positions.append(positions[-1] + estimate.shift_px)
                            estimates.append(estimate)
                    aligned = []
                    masks = []
                    current_position = positions[-1]
                    for frame, position in zip(frames, positions, strict=True):
                        compensated = compensate_translation(
                            frame, position, current_position, axis=request.motion_axis
                        )
                        aligned.append(compensated.aligned)
                        masks.append(compensated.valid_mask)
                    frames = aligned
                    valid_mask = np.logical_and.reduce(masks)
                    motion["mask"] = valid_mask
                    if estimates:
                        latest = estimates[-1]
                        motion.update(
                            {
                                "shift": latest.shift_px,
                                "quality": latest.quality,
                                "valid": latest.valid_count,
                                "roi_results": latest.roi_results,
                            }
                        )
                if request.scale != 1.0:
                    frames = [
                        cv2.resize(
                            frame,
                            None,
                            fx=request.scale,
                            fy=request.scale,
                            interpolation=cv2.INTER_AREA,
                        )
                        for frame in frames
                    ]
                    if valid_mask is not None:
                        valid_mask = cv2.resize(
                            valid_mask.astype(np.uint8),
                            (frames[-1].shape[1], frames[-1].shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                frames = [
                    apply_preprocessing(frame, request.preprocessing) for frame in frames
                ]
                if request.analysis_rois:
                    height, width = frames[-1].shape[:2]
                    analysis_mask = np.zeros((height, width), dtype=bool)
                    for roi in request.analysis_rois:
                        if not roi.get("enabled", True):
                            continue
                        factor = request.scale
                        x = max(0, round(roi["x"] * factor))
                        y = max(0, round(roi["y"] * factor))
                        x2 = min(width, round((roi["x"] + roi["width"]) * factor))
                        y2 = min(height, round((roi["y"] + roi["height"]) * factor))
                        analysis_mask[y:y2, x:x2] = True
                    valid_mask = analysis_mask if valid_mask is None else valid_mask & analysis_mask
                started = perf_counter()
                result = registry.create(request.method_id, **request.parameters).process(
                    frames,
                    valid_mask=valid_mask,
                    keep_intermediates=request.keep_intermediates,
                )
                result.primary = apply_postprocessing(result.primary, request.postprocessing)
                if not request.keep_intermediates:
                    result.intermediates.clear()
                if valid_mask is not None:
                    result.valid_mask = valid_mask
                    if result.primary.shape[:2] == valid_mask.shape:
                        result.primary = np.where(valid_mask, result.primary, 0)
                    for name, value in tuple(result.intermediates.items()):
                        if isinstance(value, np.ndarray) and value.shape[:2] == valid_mask.shape:
                            if value.ndim == 2:
                                result.intermediates[name] = np.where(valid_mask, value, 0)
                            else:
                                result.intermediates[name] = np.where(
                                    valid_mask[..., None], value, 0
                                )
                elapsed = (perf_counter() - started) * 1000.0
                with self._condition:
                    if request.generation != self._generation:
                        continue
                    state = (
                        self._cumulative_position,
                        self._last_motion_frame_id,
                        self._last_motion_image,
                    )
                state = self._calculate_cumulative_motion(request, motion, *state)
                with self._condition:
                    if request.generation != self._generation:
                        continue
                    (
                        self._cumulative_position,
                        self._last_motion_frame_id,
                        self._last_motion_image,
                    ) = state
                self._publish_result(
                    (result, elapsed, request.scale, request.stride, motion, request.frame_id)
                )
            except Exception as exc:  # isolate plug-in failures from thread lifecycle
                self.failed.emit(str(exc))
