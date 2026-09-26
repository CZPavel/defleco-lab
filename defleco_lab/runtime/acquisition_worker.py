from __future__ import annotations

from collections import deque
from threading import Event, Lock
from time import perf_counter

from PySide6 import QtCore

from defleco_lab.camera.basler_source import BaslerSource, CameraDescriptor


class BaslerAcquisitionWorker(QtCore.QThread):
    """Own all pypylon objects in one thread and expose a bounded frame mailbox.

    Camera frames are not emitted as payload-carrying Qt signals. A fast camera can
    otherwise enqueue many full-resolution NumPy arrays in the GUI event queue and
    make the application appear frozen. The GUI polls drain_frames() instead.
    """

    # Kept for compatibility with older callers; intentionally not emitted.
    frameReady = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    restoreReport = QtCore.Signal(list)
    applyReport = QtCore.Signal(dict)
    metricsReady = QtCore.Signal(dict)

    def __init__(
        self,
        descriptor: CameraDescriptor,
        temporary_settings=None,
        parent=None,
        buffer_capacity: int = 16,
    ) -> None:
        super().__init__(parent)
        self.descriptor = descriptor
        self.temporary_settings = temporary_settings or {}
        self._stop = Event()
        self._frames = deque(maxlen=max(2, int(buffer_capacity)))
        self._frames_lock = Lock()
        self._buffer_drops = 0

    def stop(self) -> bool:
        self._stop.set()
        return bool(self.wait(5000))

    def drain_frames(self) -> list:
        """Return all currently buffered frames in chronological order."""
        with self._frames_lock:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    @property
    def buffered_frames(self) -> int:
        with self._frames_lock:
            return len(self._frames)

    @property
    def buffer_drops(self) -> int:
        with self._frames_lock:
            return self._buffer_drops

    def _store_frame(self, packet) -> None:
        with self._frames_lock:
            if len(self._frames) == self._frames.maxlen:
                self._buffer_drops += 1
            self._frames.append(packet)

    def run(self) -> None:
        source = BaslerSource(self.descriptor)
        received = 0
        timeouts = 0
        errors = 0
        started = perf_counter()
        last_report = started
        try:
            source.open()
            report = source.prepare_free_run()
            if self.temporary_settings:
                report.update(source.apply_temporary(self.temporary_settings))
            if report:
                self.applyReport.emit(report)
            source.start()
            while not self._stop.is_set():
                try:
                    packet = source.grab(timeout_ms=250)
                    self._store_frame(packet)
                    received += 1
                except TimeoutError:
                    timeouts += 1
                now = perf_counter()
                if now - last_report >= 0.5:
                    self.metricsReady.emit(
                        {
                            "received_fps": received / max(now - started, 1e-9),
                            "received_frames": received,
                            "timeouts": timeouts,
                            "errors": errors,
                            "buffered_frames": self.buffered_frames,
                            "buffer_drops": self.buffer_drops,
                        }
                    )
                    last_report = now
        except Exception as exc:  # pypylon exceptions are runtime-specific
            errors += 1
            self.failed.emit(str(exc))
        finally:
            elapsed = perf_counter() - started
            self.metricsReady.emit(
                {
                    "received_fps": received / max(elapsed, 1e-9),
                    "received_frames": received,
                    "timeouts": timeouts,
                    "errors": errors,
                    "buffered_frames": self.buffered_frames,
                    "buffer_drops": self.buffer_drops,
                }
            )
            try:
                self.restoreReport.emit(source.close())
            except Exception as exc:
                self.restoreReport.emit([f"Camera close/restore failed: {exc}"])
