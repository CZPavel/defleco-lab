import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.gui.main_window import MainWindow


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
