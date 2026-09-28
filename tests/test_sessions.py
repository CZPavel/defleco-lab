from pathlib import Path

import numpy as np

from defleco_lab.camera.frame_packet import FramePacket
from defleco_lab.sessions.session_io import SessionRecorder, load_session


def test_record_replay_round_trip(tmp_path: Path):
    recorder = SessionRecorder(tmp_path, "test")
    packet = FramePacket(
        np.arange(120, dtype=np.uint8).reshape(10, 12),
        7,
        123456,
        pixel_format="Mono8",
        source="synthetic",
        estimated_motion_px=1.25,
        motion_quality=0.8,
        cumulative_position_px=4.5,
    )
    recorder.append(packet)
    path = recorder.close()
    loaded = load_session(path)
    assert len(loaded) == 1 and loaded[0].frame_id == 7
    assert np.array_equal(loaded[0].image, packet.image)
    assert loaded[0].estimated_motion_px == 1.25


def test_late_motion_metadata_update_round_trip(tmp_path: Path):
    recorder = SessionRecorder(tmp_path, "motion")
    packet = FramePacket(np.zeros((8, 8), np.uint8), 3, 44, source="synthetic")
    recorder.append(packet)
    recorder.update_motion(3, 2.5, 0.91, 8.0)
    loaded = load_session(recorder.close())
    assert loaded[0].estimated_motion_px == 2.5
    assert loaded[0].motion_quality == 0.91
    assert loaded[0].cumulative_position_px == 8.0



def test_empty_session_is_structurally_valid_and_unique(tmp_path: Path):
    first = SessionRecorder(tmp_path, "empty")
    first_path = first.close()
    second = SessionRecorder(tmp_path, "empty2")
    second_path = second.close()

    assert first_path != second_path
    assert (first_path / "metadata.csv").exists()
    assert load_session(first_path) == []
