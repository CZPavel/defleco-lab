import pytest

from defleco_lab.processing.history import FrameHistory
from defleco_lab.processing.multi_frame import TemporalStatistics


def test_frame_stride_selects_correct_history_frame():
    h = FrameHistory(10)
    for i in range(8):
        h.append(i)
    assert h.pair(4) == (3, 7) and h.last(3, 2) == [3, 5, 7]


def test_history_is_bounded():
    h = FrameHistory(2)
    h.append(1)
    h.append(2)
    h.append(3)
    assert len(h) == 2 and h.latest(1) == 2
    with pytest.raises(IndexError):
        h.latest(2)


def test_dynamic_history_requirement_includes_window_and_stride():
    assert TemporalStatistics(window=4, stride=3).history_requirement() == 10
