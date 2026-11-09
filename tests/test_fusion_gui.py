import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.gui.fusion_lab import FusionLab
from defleco_lab.gui.main_window import MainWindow
from defleco_lab.processing.surface.fusion import FUSION_VERSION, fusion_registry


def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def source():
    image = np.full((72, 96), 0.5, np.float32)
    image[25:28, 10:80] = 0.7
    image[45:49, 10:80] = 0.3
    return image


def test_fusion_tab_routes_source_without_running_other_workspaces():
    app = application()
    window = MainWindow()
    try:
        lab = window.fusion_lab
        assert window.tabs.tabText(window.tabs.indexOf(lab)) == "GPixel Surface Fusion"
        assert lab.method.count() == 7
        assert lab.family.isHidden()
        assert lab.mode.currentText() == "enhanced"
        assert not lab.active and not window.surface_lab.active
        called = []
        window.processor.submit = lambda request: called.append(request)
        window.tabs.setCurrentWidget(lab)
        window._receive_packet(FramePacket(source(), 1, 1, source="test"))
        assert wait_for(
            app, lambda: lab.snapshot is not None and lab.snapshot[1].shape == source().shape
        )
        assert lab.active and not window.surface_lab.active
        assert window.analysis_dock.isHidden()
        assert not called
        assert lab.original._image.shape == source().shape
        clipped = float(lab.snapshot[2].diagnostics["enhanced_clip_fraction"])
        assert f"Clipped pixels: {clipped:.1%}" in lab.status.text()
        window.tabs.setCurrentIndex(0)
        assert not lab.active
        assert lab.worker.pending is None
    finally:
        window.close()
        app.processEvents()
        assert not window.fusion_lab.worker.isRunning()


def test_explicit_file_survives_tab_switch_and_stream_until_use_input(tmp_path, monkeypatch):
    import cv2

    app = application()
    window = MainWindow()
    file_image = np.full((40, 48), 170, np.uint8)
    selected_file = tmp_path / "chosen.png"
    assert cv2.imwrite(str(selected_file), file_image)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(selected_file), "")
    )
    try:
        for page, lab in (
            (window.fusion_lab, window.fusion_lab),
            (window.surface_compare_page, window.surface_lab),
        ):
            window.tabs.setCurrentWidget(page)
            lab.open_image()
            assert not lab.follow_input
            assert wait_for(app, lambda selected=lab: selected.snapshot is not None)
            window._receive_packet(FramePacket(np.full((40, 48), 30, np.uint8), 2, 2))
            np.testing.assert_array_equal(lab.source, file_image)
            window.tabs.setCurrentIndex(0)
            window._receive_packet(FramePacket(np.full((40, 48), 60, np.uint8), 3, 3))
            window.tabs.setCurrentWidget(page)
            np.testing.assert_array_equal(lab.source, file_image)
            assert lab.source_metadata["source_filename"] == str(selected_file)
            lab.use_input_source()
            assert lab.follow_input
            np.testing.assert_array_equal(lab.source, np.full((40, 48), 60, np.uint8))
            window._receive_packet(FramePacket(np.full((40, 48), 90, np.uint8), 4, 4))
            np.testing.assert_array_equal(lab.source, np.full((40, 48), 90, np.uint8))
            assert "source_filename" not in lab.source_metadata
    finally:
        window.close()
        app.processEvents()


def test_preview_native_scale_and_export_recipe_metadata(tmp_path, monkeypatch):
    app = application()
    lab = FusionLab()
    try:
        lab.set_active(True)
        image = np.tile(np.arange(1300, dtype=np.uint16), (900, 1))
        lab.set_image(image, {"source_filename": "source.tiff", "frame_id": 71})
        assert wait_for(app, lambda: lab.snapshot is not None)
        assert lab.snapshot[0]["spatial_scale"] < 1
        assert lab.snapshot[1].size <= 1_000_000
        lab.submit(True, {"x": 10, "y": 20, "width": 100, "height": 80})
        assert wait_for(
            app, lambda: lab.snapshot is not None and lab.snapshot[0]["roi"] is not None
        )
        assert lab.snapshot[0]["spatial_scale"] == 1
        assert lab.snapshot[1].shape == (80, 100)
        target = tmp_path / "K01.png"
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(target), "")
        )
        lab.export()
        assert target.exists() and target.with_suffix(".npy").exists()
        saved = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert saved["recipe_version"] == FUSION_VERSION
        assert saved["method_id"] == "fusion_k01"
        assert saved["roi"] == {"x": 10, "y": 20, "width": 100, "height": 80}
        assert saved["source_filename"] == "source.tiff"
        assert saved["native_resolution"] and saved["spatial_scale"] == 1
        assert saved["active_steps"]
        np.testing.assert_array_equal(np.load(target.with_suffix(".npy")), lab.snapshot[2].primary)
    finally:
        assert lab.shutdown()
        app.processEvents()


def test_fusion_preset_roundtrip_reproduces_result_and_reset(tmp_path, monkeypatch):
    app = application()
    lab = FusionLab()
    try:
        lab.method.setCurrentIndex(lab.method.findData("fusion_k07"))
        lab.params.editors["intensity"].setValue(0.7)
        config = lab.params.values()
        expected = fusion_registry.create("fusion_k07", **config).process([source()]).primary
        target = tmp_path / "K07.json"
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(target), "")
        )
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(target), "")
        )
        lab.save_preset()
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["recipe_version"] == FUSION_VERSION
        lab.restore_defaults()
        assert lab.params.values()["intensity"] == 0.5
        lab.load_preset()
        assert lab.params.values() == config
        actual = fusion_registry.create(lab.method.currentData(), **lab.params.values()).process(
            [source()]
        )
        np.testing.assert_array_equal(actual.primary, expected)
        assert "clahe_enabled" in lab.params.editors
        assert "hessian_enabled" in lab.params.editors
        assert "morphology_enabled" in lab.params.editors
        assert "median_enabled" in lab.params.editors
        assert "fft_enabled" not in lab.params.editors
    finally:
        assert lab.shutdown()
        app.processEvents()


def test_parameter_change_discards_inflight_previous_result(monkeypatch):
    app = application()
    lab = FusionLab()
    started, release = threading.Event(), threading.Event()
    try:
        method = type(fusion_registry.create("fusion_k01"))
        original = method.process
        calls = []

        def blocked(self, frames, **context):
            calls.append(self.parameters["intensity"])
            if len(calls) == 1:
                started.set()
                assert release.wait(10)
            return original(self, frames, **context)

        monkeypatch.setattr(method, "process", blocked)
        lab.set_active(True)
        lab.set_image(source())
        assert started.wait(5)
        lab.params.editors["intensity"].setValue(0.8)
        release.set()
        assert wait_for(app, lambda: lab.snapshot is not None)
        assert lab.snapshot[0]["parameters"]["intensity"] == 0.8
        assert calls[:2] == [0.5, 0.8]
        assert lab.worker.completed is None
    finally:
        release.set()
        assert lab.shutdown()
        app.processEvents()
