import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.gui.main_window import MainWindow


def _packets(count):
    return [
        FramePacket(np.full((32, 32), index, np.uint8), index, index, source="replay")
        for index in range(count)
    ]


def _select_method(window, method_id):
    window.method_combo.setCurrentIndex(window.method_combo.findData(method_id))


def _set_editor(editor, value):
    if hasattr(editor, "setValue"):
        editor.setValue(value)
    else:
        editor.setText(str(value))


def test_replay_tick_keeps_history_chronological():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.history.clear()
    window.replay = [
        FramePacket(np.zeros((32, 32), np.uint8), index, index, source="replay")
        for index in range(3)
    ]
    window.replay_index = 0
    window._next_frame()
    window._next_frame()
    assert [packet.frame_id for packet in window.history] == [0, 1]
    window.close()
    app.processEvents()


def test_replay_seek_reconstructs_frame_difference_stride_history(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window.processor, "submit", lambda request: None)
    window.replay = _packets(9)
    _select_method(window, "frame_difference")
    window.stride.setValue(4)
    window._seek_absolute(7)
    assert [packet.frame_id for packet in window.history] == [3, 4, 5, 6, 7]
    window.close()
    app.processEvents()


def test_replay_seek_reconstructs_temporal_window_and_farneback(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window.processor, "submit", lambda request: None)
    window.replay = _packets(12)
    _select_method(window, "temporal_statistics")
    _set_editor(window.params.editors["window"], 4)
    window.stride.setValue(2)
    window._seek_absolute(9)
    assert [packet.frame_id for packet in window.history] == [3, 4, 5, 6, 7, 8, 9]
    _select_method(window, "farneback")
    window.stride.setValue(3)
    window._seek_absolute(8)
    assert [packet.frame_id for packet in window.history] == [5, 6, 7, 8]
    window.close()
    app.processEvents()


def test_replay_seek_reports_insufficient_history_without_stale_result(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window.processor, "submit", lambda request: None)
    window.replay = _packets(6)
    _select_method(window, "frame_difference")
    window.stride.setValue(4)
    window.last_result = object()
    window.processed.set_array(np.ones((8, 8), np.uint8))
    window._seek_absolute(2)
    assert not window.history
    assert window.last_result is None
    assert "Insufficient history" in window.statusBar().currentMessage()
    assert window.processed._pixmap.pixmap().isNull()
    window.close()
    app.processEvents()


def test_method_switch_rebuilds_loaded_replay_history(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window.processor, "submit", lambda request: None)
    window.replay = _packets(12)
    _select_method(window, "gradient")
    window._seek_absolute(9)
    assert len(window.history) == 1
    _select_method(window, "temporal_statistics")
    assert [packet.frame_id for packet in window.history] == list(range(2, 10))
    window.close()
    app.processEvents()


def test_play_from_replay_end_restarts_at_zero_and_resets_motion(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.replay = _packets(5)
    window.replay_index = 4
    reset_calls = []
    monkeypatch.setattr(window.processor, "reset_state", lambda: reset_calls.append(True))
    window._toggle_replay()
    assert window.timer.isActive()
    assert window.replay_index == 0
    assert reset_calls
    window.timer.stop()
    window.close()
    app.processEvents()


def test_previous_next_use_displayed_slider_frame(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    monkeypatch.setattr(window.processor, "submit", lambda request: None)
    window.replay = _packets(5)
    window.slider.setRange(0, 4)
    window._seek_absolute(2)
    window._seek(1)
    assert window.slider.value() == 3
    assert window.replay_index == 3
    window._seek(-1)
    assert window.slider.value() == 2
    window.close()
    app.processEvents()
