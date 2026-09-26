from __future__ import annotations

import sys

from PySide6 import QtWidgets

from defleco_lab.gui.main_window import MainWindow
from defleco_lab.processing import load_builtin_methods


def main() -> int:
    load_builtin_methods()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Defleco LAB")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
