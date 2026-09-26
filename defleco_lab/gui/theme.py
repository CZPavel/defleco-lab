"""Centralized Qt application theme and system-font selection."""

from __future__ import annotations

import platform
from dataclasses import dataclass

from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True, slots=True)
class ThemeInfo:
    style: str
    font_family: str
    font_point_size: float


def select_system_font() -> QtGui.QFont:
    """Return a readable installed UI font without bundling font files."""

    system_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    installed = {family.casefold(): family for family in QtGui.QFontDatabase.families()}
    if platform.system() == "Windows" and "segoe ui" in installed:
        font = QtGui.QFont(installed["segoe ui"])
        point_size = system_font.pointSizeF()
        font.setPointSizeF(point_size if point_size > 0 else 9.0)
        return font
    if system_font.family() and system_font.family().casefold() in installed:
        return system_font
    for fallback in ("Arial", "Noto Sans", "DejaVu Sans"):
        if fallback.casefold() in installed:
            font = QtGui.QFont(installed[fallback.casefold()])
            font.setPointSizeF(9.0)
            return font
    return QtGui.QFont("Sans Serif", 9)


def dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    active = QtGui.QPalette.ColorGroup.Active
    inactive = QtGui.QPalette.ColorGroup.Inactive
    disabled = QtGui.QPalette.ColorGroup.Disabled
    colors = {
        QtGui.QPalette.ColorRole.Window: "#16191d",
        QtGui.QPalette.ColorRole.WindowText: "#e6edf3",
        QtGui.QPalette.ColorRole.Base: "#20252b",
        QtGui.QPalette.ColorRole.AlternateBase: "#252b31",
        QtGui.QPalette.ColorRole.ToolTipBase: "#2a3037",
        QtGui.QPalette.ColorRole.ToolTipText: "#e6edf3",
        QtGui.QPalette.ColorRole.Text: "#e6edf3",
        QtGui.QPalette.ColorRole.Button: "#2a3037",
        QtGui.QPalette.ColorRole.ButtonText: "#e6edf3",
        QtGui.QPalette.ColorRole.BrightText: "#ffffff",
        QtGui.QPalette.ColorRole.Highlight: "#287c8a",
        QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
        QtGui.QPalette.ColorRole.Link: "#4dd0e1",
        QtGui.QPalette.ColorRole.PlaceholderText: "#77818a",
    }
    for group in (active, inactive):
        for role, value in colors.items():
            palette.setColor(group, role, QtGui.QColor(value))
    for role in (
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QPalette.ColorRole.Text,
        QtGui.QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QtGui.QColor("#707980"))
    palette.setColor(disabled, QtGui.QPalette.ColorRole.Base, QtGui.QColor("#1b1f24"))
    palette.setColor(disabled, QtGui.QPalette.ColorRole.Button, QtGui.QColor("#22272d"))
    return palette


DARK_STYLESHEET = """
QMainWindow, QDialog { background: #16191d; color: #e6edf3; }
QWidget { color: #e6edf3; }
QDockWidget { color: #e6edf3; font-weight: 600; }
QDockWidget::title {
    background: #20252b; border: 1px solid #3b4650;
    padding: 7px 9px; text-align: left;
}
QDockWidget > QWidget { background: #20252b; }
QToolBar {
    background: #20252b; border: 0; border-bottom: 1px solid #3b4650;
    spacing: 5px; padding: 4px;
}
QToolBar::separator { background: #3b4650; width: 1px; margin: 4px 6px; }
QTabWidget::pane { border: 1px solid #3b4650; background: #11151a; }
QTabBar::tab {
    background: #20252b; color: #9aa4ad; border: 1px solid #3b4650;
    border-bottom: 0; padding: 7px 13px; margin-right: 1px;
}
QTabBar::tab:selected { background: #2a3037; color: #e6edf3; }
QTabBar::tab:hover:!selected { color: #c8d1d9; }
QPushButton, QToolButton {
    background: #2a3037; border: 1px solid #3b4650; border-radius: 3px;
    padding: 5px 9px; min-height: 18px;
}
QPushButton:hover, QToolButton:hover { border-color: #4d9eaa; background: #303840; }
QPushButton:pressed, QToolButton:pressed { background: #1e5962; }
QPushButton:disabled, QToolButton:disabled { color: #707980; background: #22272d; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #2a3037; border: 1px solid #3b4650; border-radius: 3px;
    padding: 4px 6px; selection-background-color: #287c8a;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #52616d;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #4dd0e1;
}
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox QAbstractItemView {
    background: #2a3037; color: #e6edf3; border: 1px solid #3b4650;
    selection-background-color: #287c8a;
}
QLineEdit[invalid="true"] { border-color: #df5b5b; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid #52616d;
    border-radius: 2px; background: #2a3037;
}
QCheckBox::indicator:checked { background: #287c8a; border-color: #4dd0e1; }
QTextBrowser, QPlainTextEdit, QTextEdit, QScrollArea {
    background: #1b2025; border: 1px solid #3b4650; selection-background-color: #287c8a;
}
QSlider::groove:horizontal { height: 4px; background: #3b4650; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0; border-radius: 7px; background: #4dd0e1;
}
QStatusBar { background: #20252b; border-top: 1px solid #3b4650; color: #9aa4ad; }
QMenu { background: #20252b; border: 1px solid #3b4650; padding: 3px; }
QMenu::item { padding: 5px 24px 5px 10px; }
QMenu::item:selected { background: #287c8a; }
QToolTip { background: #2a3037; color: #e6edf3; border: 1px solid #52616d; padding: 4px; }
QScrollBar:vertical { background: #1b2025; width: 12px; margin: 0; }
QScrollBar:horizontal { background: #1b2025; height: 12px; margin: 0; }
QScrollBar::handle { background: #46515b; border-radius: 5px; min-height: 24px; min-width: 24px; }
QScrollBar::handle:hover { background: #596773; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QGroupBox { border: 1px solid #3b4650; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9aa4ad; }
"""


def apply_theme(application: QtWidgets.QApplication) -> ThemeInfo:
    application.setStyle("Fusion")
    font = select_system_font()
    application.setFont(font)
    application.setPalette(dark_palette())
    application.setStyleSheet(DARK_STYLESHEET)
    return ThemeInfo("Fusion", font.family(), font.pointSizeF())
