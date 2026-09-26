from __future__ import annotations

from threading import Event
from time import perf_counter

from PySide6 import QtCore

from defleco_lab.camera.basler_source import BaslerSource, CameraDescriptor


class BaslerAcquisitionWorker(QtCore.QThread):
    """Owns all pypylon camera objects inside one bounded acquisition thread."""

    frameReady = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    restoreReport = QtCore.Signal(list)
    applyReport = QtCore.Signal(dict)
    metricsReady = QtCore.Signal(dict)

    def __init__(self, descriptor: CameraDescriptor, temporary_settings=None, parent=None) -> None:
        super().__init__(parent)
        self.descriptor = descriptor
        self.temporary_settings = temporary_settings or {}
        self._stop = Event()

    def stop(self) -> None:
        self._stop.set()
        self.wait(5000)

    def run(self) -> None:
        source = BaslerSource(self.descriptor)
        received = 0
        timeouts = 0
        errors = 0
        started = perf_counter()
        last_report = started
        try:
            source.open()
            if self.temporary_settings:
                self.applyReport.emit(source.apply_temporary(self.temporary_settings))
            source.start()
            while not self._stop.is_set():
                try:
                    self.frameReady.emit(source.grab(timeout_ms=500))
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
                }
            )
            try:
                self.restoreReport.emit(source.close())
            except Exception as exc:
                self.restoreReport.emit([f"Camera close/restore failed: {exc}"])
