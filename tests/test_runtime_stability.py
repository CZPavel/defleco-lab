from __future__ import annotations

import numpy as np

from defleco_lab.app import configure_opencv_runtime
from defleco_lab.camera.basler_source import BaslerSource, CameraDescriptor
from defleco_lab.processing.single_frame import GaborBank, Gradient, StructureTensor
from defleco_lab.runtime.acquisition_worker import BaslerAcquisitionWorker
from defleco_lab.runtime.processing_worker import ProcessingWorker


class _FakeEnumNode:
    def __init__(self, value: str, values: list[str]) -> None:
        self.Value = value
        self.Symbolics = values

    def IsReadable(self) -> bool:
        return True

    def IsWritable(self) -> bool:
        return True


def test_camera_mailbox_is_bounded_and_drains_chronologically() -> None:
    descriptor = CameraDescriptor("MODEL", "", "BaslerGigE", serial="hidden")
    worker = BaslerAcquisitionWorker(descriptor, buffer_capacity=3)
    for item in range(5):
        worker._store_frame(item)
    assert worker.buffered_frames == 3
    assert worker.buffer_drops == 2
    assert worker.drain_frames() == [2, 3, 4]
    assert worker.buffered_frames == 0


def test_processing_result_mailbox_keeps_only_latest_payload() -> None:
    worker = ProcessingWorker()
    worker._publish_result(("old",))
    worker._publish_result(("new",))
    assert worker.result_dropped == 1
    assert worker.take_latest_result() == ("new",)
    assert worker.take_latest_result() is None


def test_free_run_state_is_temporary_and_restored(monkeypatch) -> None:
    descriptor = CameraDescriptor("MODEL", "", "BaslerGigE", serial="hidden")
    source = BaslerSource(descriptor)
    source.camera = object()
    acquisition = _FakeEnumNode("SingleFrame", ["Continuous", "SingleFrame"])
    selector = _FakeEnumNode("AcquisitionStart", ["AcquisitionStart", "FrameStart"])
    trigger = _FakeEnumNode("On", ["On", "Off"])
    nodes = {
        "AcquisitionMode": acquisition,
        "TriggerSelector": selector,
        "TriggerMode": trigger,
    }
    monkeypatch.setattr(source, "_node", lambda name: nodes.get(name))

    source.prepare_free_run()
    assert acquisition.Value == "Continuous"
    assert selector.Value == "FrameStart"
    assert trigger.Value == "Off"

    failures: list[str] = []
    source._restore_free_run(failures)
    assert not failures
    assert acquisition.Value == "SingleFrame"
    assert selector.Value == "AcquisitionStart"
    assert trigger.Value == "On"


def test_single_frame_live_path_can_skip_large_intermediate_maps() -> None:
    _y, x = np.mgrid[:96, :128]
    image = (127 + 60 * np.sin(2 * np.pi * x / 12)).astype(np.float32)

    gradient = Gradient().process([image], keep_intermediates=False)
    assert gradient.primary.shape == image.shape
    assert not gradient.intermediates

    tensor = StructureTensor().process([image], keep_intermediates=False)
    assert tensor.primary.shape == image.shape
    assert not tensor.intermediates

    gabor = GaborBank(periods=[12], orientations=4, sigma=5).process(
        [image], keep_intermediates=False
    )
    assert gabor.primary.shape == image.shape
    assert np.isfinite(gabor.primary).all()
    assert not gabor.intermediates


def test_opencv_runtime_thread_limit_is_positive(monkeypatch) -> None:
    monkeypatch.setenv("DEFLECO_OPENCV_THREADS", "2")
    assert configure_opencv_runtime() == 2
