from __future__ import annotations

import os
import sys

import cv2
from PySide6 import QtWidgets

from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.theme import ThemeInfo, apply_theme
from defleco_lab.processing import load_builtin_methods


def configure_application(application: QtWidgets.QApplication) -> ThemeInfo:
    application.setApplicationName("Defleco LAB")
    application.setOrganizationName("Defleco LAB")
    return apply_theme(application)


def configure_opencv_runtime() -> int:
    """Reserve CPU headroom for the Qt event loop during heavy image processing."""
    default_threads = max(1, min(4, (os.cpu_count() or 4) - 1))
    raw = os.environ.get("DEFLECO_OPENCV_THREADS", str(default_threads))
    try:
        threads = max(1, int(raw))
    except ValueError:
        threads = default_threads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(threads)
    return threads


def main() -> int:
    configure_opencv_runtime()
    load_builtin_methods()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
