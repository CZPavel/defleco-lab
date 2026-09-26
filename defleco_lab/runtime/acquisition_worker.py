from __future__ import annotations

from threading import Event

from PySide6 import QtCore

from defleco_lab.camera.basler_source import BaslerSource, CameraDescriptor


class BaslerAcquisitionWorker(QtCore.QThread):
    """Owns all pypylon camera objects inside one bounded acquisition thread."""

    frameReady = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    restoreReport = QtCore.Signal(list)
    applyReport = QtCore.Signal(dict)

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
        try:
            source.open()
            if self.temporary_settings:
                self.applyReport.emit(source.apply_temporary(self.temporary_settings))
            source.start()
            while not self._stop.is_set():
                try:
                    self.frameReady.emit(source.grab(timeout_ms=500))
                except TimeoutError:
                    continue
        except Exception as exc:  # pypylon exceptions are runtime-specific
            self.failed.emit(str(exc))
        finally:
            try:
                self.restoreReport.emit(source.close())
            except Exception as exc:
                self.restoreReport.emit([f"Camera close/restore failed: {exc}"])
