from __future__ import annotations

from pathlib import Path
from queue import Full, Queue
from threading import Thread

from defleco_lab.camera.frame_packet import FramePacket

from .session_io import SessionRecorder


class AsyncSessionRecorder:
    """Bounded raw recorder; acquisition/display never waits for disk I/O."""

    def __init__(self, root: Path, notes: str = "", capacity: int = 64) -> None:
        self._recorder = SessionRecorder(root, notes)
        self._queue: Queue[object] = Queue(maxsize=capacity)
        self.dropped = 0
        self.error: str | None = None
        self._thread = Thread(target=self._run, name="defleco-recorder", daemon=True)
        self._thread.start()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def append(self, packet: FramePacket) -> bool:
        try:
            self._queue.put_nowait(packet.copy_owned())
            return True
        except Full:
            self.dropped += 1
            return False

    def update_motion(
        self,
        frame_id: int,
        estimated_motion_px: float,
        motion_quality: float,
        cumulative_position_px: float,
    ) -> None:
        try:
            self._queue.put_nowait(
                ("motion", frame_id, estimated_motion_px, motion_quality, cumulative_position_px)
            )
        except Full:
            self.dropped += 1

    def close(self) -> Path:
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except Full:
                continue
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise TimeoutError("Recorder did not stop within 30 seconds")
        if self.error:
            raise OSError(self.error)
        return self._recorder.close()

    def _run(self) -> None:
        try:
            while True:
                packet = self._queue.get()
                if packet is None:
                    return
                if isinstance(packet, tuple) and packet[0] == "motion":
                    self._recorder.update_motion(*packet[1:])
                else:
                    self._recorder.append(packet)
        except Exception as exc:  # recorder error is reported to the GUI on close
            self.error = str(exc)
