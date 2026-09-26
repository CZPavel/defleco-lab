import cv2
import numpy as np
import pytest

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.runtime.processing_worker import ProcessingRequest, ProcessingWorker


def _shift(image, dx):
    return cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, 0]]), (image.shape[1], image.shape[0])
    )


def _request(packets):
    return ProcessingRequest(
        method_id="frame_difference",
        parameters={"stride": 1},
        frames=[packet.image for packet in packets[-2:]],
        scale=1.0,
        stride=1,
        motion_compensation=True,
        motion_rois=[{"x": 16, "y": 16, "width": 128, "height": 80, "name": "stable"}],
        frame_id=packets[-1].frame_id,
        motion_frames=packets,
    )


def test_cumulative_motion_includes_available_frames_skipped_by_latest_wins():
    base = np.random.default_rng(7).normal(127, 35, (112, 176)).astype(np.float32)
    packets = [FramePacket(_shift(base, index), 100 + index, index) for index in range(4)]
    worker = ProcessingWorker()
    first = {"shift": 0.0, "quality": 0.0, "valid": 0}
    worker._accumulate_motion(_request(packets[:1]), first)
    latest = {"shift": 0.0, "quality": 0.0, "valid": 0}
    worker._accumulate_motion(_request(packets), latest)
    assert latest["cumulative"] == pytest.approx(3.0, abs=0.25)
    assert worker._last_motion_frame_id == 103


def test_processing_motion_state_reset_starts_new_experiment():
    worker = ProcessingWorker()
    worker._cumulative_position = 12.5
    worker._last_motion_frame_id = 103
    worker._last_motion_image = np.ones((8, 8), np.uint8)
    worker.reset_state()
    assert worker._cumulative_position == 0.0
    assert worker._last_motion_frame_id == -1
    assert worker._last_motion_image is None
