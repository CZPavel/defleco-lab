import json
import os
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.gui.experiment_panel import ExperimentPanel
from defleco_lab.gui.main_window import MainWindow


def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeCamera:
    def __init__(self):
        self.packets = []

    def isRunning(self):
        return True

    def drain_frames(self):
        packets, self.packets = self.packets, []
        return packets

    def stop(self):
        return True

    def deleteLater(self):
        pass


def test_surface_profiles_update_controls_and_preserve_pattern_profiles():
    app = application()
    panel = ExperimentPanel()
    try:
        assert panel.profile.currentData() == "quick"
        assert not panel.settle_ms.isHidden()
        panel.profile.setCurrentIndex(panel.profile.findData("surface"))
        assert panel.case_count.text() == "50"
        assert panel.settle_ms.isHidden()
        assert panel.use_pattern_colors.isHidden()
        assert panel.auto_process.isHidden()
        assert not panel.surface_npy.isHidden()
        assert not panel.config()["save_response"]
        assert panel.start_capture.text() == "Run Surface screening"
        panel.profile.setCurrentIndex(panel.profile.findData("surface_extended"))
        assert panel.case_count.text() == "100"
        panel.profile.setCurrentIndex(panel.profile.findData("extended"))
        assert not panel.settle_ms.isHidden()
        assert panel.surface_npy.isHidden()
        assert panel.start_capture.text() == "Run pattern capture sweep"
    finally:
        panel.close()
        app.processEvents()


def test_surface_capture_requires_basler_and_fresh_owned_packet(tmp_path, monkeypatch):
    app = application()
    window = MainWindow()
    try:
        config = {"profile": "surface", "parent": str(tmp_path), "save_response": True}
        window._run_capture_experiment(config)
        assert "Basler" in window.experiment.status.text()
        window.source_combo.setCurrentText("Basler")
        camera = window.camera_worker = FakeCamera()
        captured = []
        monkeypatch.setattr(window, "_start_surface_worker", captured.append)
        window.surface_lab.colormap.setCurrentText("Viridis")
        window.surface_lab.opacity.setValue(0.25)
        window.surface_lab.overlay_base.setCurrentText("enhanced")
        window._run_capture_experiment(config)
        assert window.surface_capture_timer.isActive()
        assert window.tabs.currentIndex() == 0
        assert not window.tabs.isTabEnabled(1)
        after = window._surface_capture_request["after_ns"]
        window.surface_lab.opacity.setValue(0.8)
        window.surface_lab.overlay_base.setCurrentText("original")
        raw = np.full((16, 17), 32000, np.uint16)
        camera.packets.append(FramePacket(raw, 3, after - 1))
        window._poll_camera_frames()
        assert not captured
        camera.packets.append(FramePacket(raw, 4, max(time.perf_counter_ns(), after + 1)))
        window._poll_camera_frames()
        assert len(captured) == 1
        parent, packet, profile, save_response = captured[0].capture
        assert parent == tmp_path and profile == "surface" and save_response
        assert captured[0].display_settings["opacity"] == 0.25
        assert captured[0].display_settings["overlay_base"] == "enhanced"
        assert captured[0].display_settings["colormap"] == "Viridis"
        assert packet.frame_id == 4
        raw[:] = 0
        assert np.all(packet.image == 32000)
        assert not window.surface_capture_timer.isActive()
    finally:
        window._set_surface_screening_lock(False)
        window.close()
        app.processEvents()


def test_surface_timeout_and_stop_preserve_pattern_output(tmp_path, monkeypatch):
    app = application()
    window = MainWindow()
    try:
        window.source_combo.setCurrentText("Basler")
        window.camera_worker = FakeCamera()
        output = window.pattern_output.output
        calls = []
        monkeypatch.setattr(output, "hide_output", lambda: calls.append("hide"))
        config = {"profile": "surface", "parent": str(tmp_path)}
        window._run_capture_experiment(config)
        window._stop_experiment()
        assert "cancelled" in window.experiment.status.text().lower()
        assert window.tabs.isTabEnabled(1)
        assert not calls
        window._run_capture_experiment(config)
        window._surface_capture_timeout()
        assert "Timed out" in window.experiment.status.text()
        assert window.tabs.isTabEnabled(1)
        assert not calls
        window.surface_capture_timer.stop()
    finally:
        window.close()
        app.processEvents()


def test_surface_offline_dispatch_uses_selected_options(tmp_path, monkeypatch):
    app = application()
    window = MainWindow()
    try:
        (tmp_path / "capture_manifest.json").write_text(
            json.dumps({"schema": "defleco.surface.screening.capture.v1", "profile": "surface"})
        )
        panel = window.experiment
        panel.profile.setCurrentIndex(panel.profile.findData("surface_extended"))
        panel.surface_npy.setChecked(True)
        window.surface_lab.colormap.setCurrentText("Magma")
        window.surface_lab.opacity.setValue(0.37)
        window.surface_lab.overlay_base.setCurrentText("original")
        captured = []
        monkeypatch.setattr(window, "_start_surface_worker", captured.append)
        window._start_offline_processing(str(tmp_path))
        assert len(captured) == 1
        assert captured[0].root == tmp_path
        assert captured[0].profile == "surface_extended"
        assert captured[0].save_response is True
        assert captured[0].display_settings["colormap"] == "Magma"
        assert captured[0].display_settings["opacity"] == 0.37
        assert captured[0].display_settings["overlay_base"] == "original"
    finally:
        window.close()
        app.processEvents()


def test_surface_live_frame_runs_worker_and_enables_gallery(tmp_path, monkeypatch):
    from defleco_lab.experiments import surface_screening

    app = application()
    window = MainWindow()
    try:
        recipes = surface_screening.build_surface_recipes("surface")
        monkeypatch.setattr(
            surface_screening, "build_surface_recipes", lambda _profile: recipes[:1]
        )
        window.source_combo.setCurrentText("Basler")
        camera = window.camera_worker = FakeCamera()
        window._run_capture_experiment({"profile": "surface", "parent": str(tmp_path)})
        after = window._surface_capture_request["after_ns"]
        camera.packets.append(FramePacket(np.full((32, 32), 4000, np.uint16), 5, after + 1))
        window._poll_camera_frames()
        assert window._experiment_processing_worker is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not window.experiment.open_results.isEnabled():
            app.processEvents()
            time.sleep(0.01)
        assert window.experiment.open_results.isEnabled()
        assert "completed" in window.experiment.status.text()
        assert window.tabs.isTabEnabled(1)
        assert list(tmp_path.glob("Surface_screening_*/capture_manifest.json"))
    finally:
        window.close()
        app.processEvents()


def test_window_close_cancels_and_joins_surface_worker():
    import threading

    from PySide6 import QtCore

    app = application()
    window = MainWindow()
    started = threading.Event()
    stopped = threading.Event()

    class WaitingWorker(QtCore.QThread):
        progressChanged = QtCore.Signal(int, int, str)
        completed = QtCore.Signal(str)
        failed = QtCore.Signal(str)

        def cancel(self):
            self.requestInterruption()

        def run(self):
            started.set()
            while not self.isInterruptionRequested():
                time.sleep(0.01)
            stopped.set()

    worker = WaitingWorker(window)
    window._start_surface_worker(worker)
    assert started.wait(5)
    window.close()
    assert stopped.is_set()
    assert not worker.isRunning()
    app.processEvents()
