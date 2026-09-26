from __future__ import annotations

import sys

from PySide6 import QtWidgets

from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.theme import ThemeInfo, apply_theme
from defleco_lab.processing import load_builtin_methods


def configure_application(application: QtWidgets.QApplication) -> ThemeInfo:
    application.setApplicationName("Defleco LAB")
    application.setOrganizationName("Defleco LAB")
    return apply_theme(application)


def main() -> int:
    load_builtin_methods()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
