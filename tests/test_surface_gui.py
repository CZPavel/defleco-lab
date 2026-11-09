import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.surface_lab import SurfaceLab


def test_manual_and_screening_share_enhanced_background_for_absolute_heatmap():
    from defleco_lab.gui.surface_io import render_surface
    from defleco_lab.processing.base import ProcessingResult

    application = app()
    lab = SurfaceLab()
    try:
        lab.mode.setCurrentText("overlay")
        lab.polarity.setCurrentText("absolute")
        lab.range_mode.setCurrentText("fixed")
        source = np.full((1, 3), 0.5, np.float32)
        result = ProcessingResult(np.array([[-0.1, 0.0, 0.1]], np.float32))
        settings = lab.settings()
        assert settings["enhanced_polarity"] == "signed"
        manual = render_surface(source, result, settings)
        screening = render_surface(source, result, settings | {"enhanced_polarity": "signed"})
        np.testing.assert_array_equal(manual, screening)
    finally:
        assert lab.shutdown()
        application.processEvents()


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
        window.tabs.setCurrentWidget(window.surface_compare_page)
        called = []
        window.processor.submit = lambda request: called.append(request)
        from defleco_lab.camera.frame_packet import FramePacket

        window._receive_packet(FramePacket(np.zeros((64, 64), dtype=np.uint8), 1, 1))
        assert wait_for(application, lambda: window.surface_lab.snapshot is not None)
        assert not called
    finally:
        window.close()
        application.processEvents()


def test_surface_heatmap_controls_and_old_preset_compatibility(tmp_path, monkeypatch):
    application = app()
    lab = SurfaceLab()
    try:
        assert lab.settings()["overlay_base"] == "enhanced"
        assert lab.settings()["colormap"] == "Turbo"
        lab.overlay_base.setCurrentText("original")
        lab.colormap.setCurrentText("Viridis")
        lab.opacity.setValue(0.35)
        assert (
            lab.settings()["overlay_base"],
            lab.settings()["colormap"],
            lab.settings()["opacity"],
        ) == ("original", "Viridis", 0.35)

        new_path = tmp_path / "new-preset.json"
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(new_path), "")
        )
        lab.save_preset()
        saved = json.loads(new_path.read_text(encoding="utf-8"))
        assert saved["display_settings"]["overlay_base"] == "original"
        assert saved["display_settings"]["colormap"] == "Viridis"
        lab.overlay_base.setCurrentText("enhanced")
        lab.colormap.setCurrentText("Magma")
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(new_path), "")
        )
        lab.load_preset()
        assert lab.overlay_base.currentText() == "original"
        assert lab.colormap.currentText() == "Viridis"

        old_settings = lab.settings()
        del old_settings["overlay_base"]
        del old_settings["colormap"]
        path = tmp_path / "old-preset.json"
        path.write_text(
            json.dumps(
                {
                    "method_id": lab.method.currentData(),
                    "parameters": lab.params.values(),
                    "display_settings": old_settings,
                }
            ),
            encoding="utf-8",
        )
        lab.overlay_base.setCurrentText("enhanced")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
        lab.load_preset()
        assert lab.overlay_base.currentText() == "original"
        assert lab.colormap.currentText() == "Turbo"
        assert lab.opacity.value() == 0.35
    finally:
        assert lab.shutdown()
        application.processEvents()


def test_surface_tab_restores_analysis_dock_visibility():
    application = app()
    window = MainWindow()
    try:
        window.analysis_dock.show()
        window.tabs.setCurrentWidget(window.surface_compare_page)
        assert window.analysis_dock.isHidden()
        window.tabs.setCurrentIndex(0)
        assert not window.analysis_dock.isHidden()
        window.analysis_dock.hide()
        window.tabs.setCurrentWidget(window.surface_compare_page)
        window.tabs.setCurrentIndex(0)
        assert window.analysis_dock.isHidden()
    finally:
        window.close()
        application.processEvents()


def test_surface_single_and_compare_share_result_without_reprocessing():
    application = app()
    window = MainWindow()
    try:
        window.tabs.setCurrentWidget(window.surface_compare_page)
        lab = window.surface_lab
        assert wait_for(application, lambda: lab.snapshot is not None)
        snapshot = lab.snapshot
        lab.worker.submit = lambda request: (_ for _ in ()).throw(
            AssertionError("Changing the surface view must not recompute the result")
        )
        window.tabs.setCurrentWidget(window.surface_single_page)
        assert window.tabs.tabText(window.tabs.currentIndex()) == "Surface LAB"
        assert lab.parentWidget() is window.surface_single_page
        assert lab.original_panel.isHidden()
        assert not lab.result_view.isHidden()
        assert lab.roi_view() is lab.result_view
        assert lab.snapshot is snapshot
        assert window.analysis_dock.isHidden()
        window.tabs.setCurrentWidget(window.surface_compare_page)
        assert window.tabs.tabText(window.tabs.currentIndex()) == "Surface Compare"
        assert not lab.original_panel.isHidden()
        assert lab.roi_view() is lab.original
        assert lab.snapshot is snapshot
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
