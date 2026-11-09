import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.surface_lab import SurfaceLab


def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(application, condition, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        application.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_surface_async_preview_error_recovery_and_native_roi():
    application = app()
    lab = SurfaceLab()
    try:
        image = np.tile(np.arange(1300, dtype=np.uint16), (900, 1))
        lab.set_active(True)
        lab.set_image(image, {"source_filename": "source.tif"})
        assert wait_for(application, lambda: lab.snapshot is not None)
        request, processed, result = lab.snapshot
        assert processed.shape[0] * processed.shape[1] <= 1_000_000
        assert request["source_metadata"]["source_filename"] == "source.tif"
        assert result.primary.shape == processed.shape[:2]
        assert np.array_equal(image, lab.source)
        lab.range_mode.setCurrentText("fixed")
        lab.low.setValue(2)
        assert wait_for(application, lambda: "Processing error" in lab.status.text())
        lab.high.setValue(3)
        assert wait_for(
            application,
            lambda: lab.snapshot is not None and lab.snapshot[0]["settings"]["high"] == 3,
        )
        lab.submit(native=True, roi={"x": 10, "y": 20, "width": 100, "height": 80})
        assert wait_for(
            application, lambda: lab.snapshot is not None and lab.snapshot[0]["roi"] is not None
        )
        assert lab.snapshot[1].shape == (80, 100)
        assert lab.result_view._image.shape[:2] == image.shape
    finally:
        assert lab.shutdown()


def test_surface_tab_does_not_submit_legacy_processing():
    application = app()
    window = MainWindow()
    try:
        window.tabs.setCurrentWidget(window.surface_lab)
        called = []
        window.processor.submit = lambda request: called.append(request)
        from defleco_lab.camera.frame_packet import FramePacket

        window._receive_packet(FramePacket(np.zeros((64, 64), dtype=np.uint8), 1, 1))
        assert wait_for(application, lambda: window.surface_lab.snapshot is not None)
        assert not called
    finally:
        window.close()
        application.processEvents()


def test_surface_tab_restores_analysis_dock_visibility():
    application = app()
    window = MainWindow()
    try:
        window.analysis_dock.show()
        window.tabs.setCurrentWidget(window.surface_lab)
        assert window.analysis_dock.isHidden()
        window.tabs.setCurrentIndex(0)
        assert not window.analysis_dock.isHidden()
        window.analysis_dock.hide()
        window.tabs.setCurrentWidget(window.surface_lab)
        window.tabs.setCurrentIndex(0)
        assert window.analysis_dock.isHidden()
    finally:
        window.close()
        application.processEvents()


def test_open_invalid_and_bgra_image_reports_error(tmp_path, monkeypatch):
    import cv2

    application = app()
    lab = SurfaceLab()
    try:
        image = np.zeros((32, 32, 4), np.uint8)
        path = tmp_path / "alpha.png"
        assert cv2.imwrite(str(path), image)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
        lab.open_image()
        assert "three-channel" in lab.status.text()
        assert lab.source is None
        path.unlink()
        lab.open_image()
        assert "Cannot open image" in lab.status.text()
        assert lab.source is None
    finally:
        assert lab.shutdown()
        application.processEvents()


def test_open_replaces_inflight_native_image(tmp_path, monkeypatch):
    import threading

    import cv2

    from defleco_lab.processing.surface import surface_registry

    application = app()
    lab = SurfaceLab()
    started, release = threading.Event(), threading.Event()
    try:
        info = next(iter(surface_registry.infos()))
        method_type = type(surface_registry.create(info.id))
        original_process = method_type.process
        calls = []

        def blocked_process(self, frames, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                assert release.wait(10)
            return original_process(self, frames, **kwargs)

        monkeypatch.setattr(method_type, "process", blocked_process)
        lab.set_active(True)
        lab.source = np.zeros((64, 64), dtype=np.uint8)
        lab.source_metadata = {"source_filename": "old.png"}
        lab.process_full()
        assert started.wait(5)
        assert lab.native_inflight
        path = tmp_path / "new.png"
        assert cv2.imwrite(str(path), np.full((64, 64), 120, dtype=np.uint8))
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
        lab.open_image()
        release.set()
        assert wait_for(application, lambda: lab.snapshot is not None)
        assert lab.snapshot[0]["source_metadata"]["source_filename"] == str(path)
        assert len(calls) >= 2
        assert np.all(lab.snapshot[1] == 120)
    finally:
        release.set()
        assert lab.shutdown()
        application.processEvents()
