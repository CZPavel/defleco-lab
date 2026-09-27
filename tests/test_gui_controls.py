from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtGui, QtWidgets

from defleco_lab.app import configure_application
from defleco_lab.gui.parameter_panel import ListEditor, ParameterPanel
from defleco_lab.gui.theme import select_system_font
from defleco_lab.gui.visualization import (
    RangeMode,
    VisualizationSettings,
    VisualizationTransform,
)


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dark_theme_uses_fusion_and_installed_system_font() -> None:
    app = _app()
    info = configure_application(app)
    installed = {family.casefold() for family in QtGui.QFontDatabase.families()}
    assert info.style.casefold() == "fusion"
    assert info.font_family
    assert info.font_family.casefold() in installed or info.font_family == "Sans Serif"
    assert app.palette().color(QtGui.QPalette.ColorRole.Window).name() == "#16191d"
    assert "QDockWidget" in app.styleSheet()


def test_system_font_is_valid_without_bundled_font() -> None:
    _app()
    font = select_system_font()
    assert font.family()
    assert font.pointSizeF() > 0


def test_parameter_panel_uses_typed_editors_and_values() -> None:
    _app()
    panel = ParameterPanel()
    panel.set_schema(
        {
            "enabled": {"type": "bool", "default": True, "tooltip": "Use correction"},
            "count": {"type": "int", "default": 3, "minimum": 1, "maximum": 9},
            "sigma": {
                "type": "float",
                "default": 1.25,
                "minimum": 0.0,
                "maximum": 5.0,
                "step": 0.05,
                "decimals": 2,
                "units": "px",
            },
            "mode": {
                "type": "enum",
                "default": "fast",
                "choices": ["fast", "quality"],
            },
            "periods": {
                "type": "list",
                "default": [8.0, 12.0],
                "item_type": "float",
            },
        }
    )
    assert isinstance(panel.editors["enabled"], QtWidgets.QCheckBox)
    assert isinstance(panel.editors["count"], QtWidgets.QSpinBox)
    assert isinstance(panel.editors["sigma"], QtWidgets.QDoubleSpinBox)
    assert isinstance(panel.editors["mode"], QtWidgets.QComboBox)
    assert isinstance(panel.editors["periods"], ListEditor)

    panel.set_values(
        {"enabled": False, "count": 7, "sigma": 2.5, "mode": "quality", "periods": [6, 9]}
    )
    assert panel.values() == {
        "enabled": False,
        "count": 7,
        "sigma": 2.5,
        "mode": "quality",
        "periods": [6.0, 9.0],
    }


def test_legacy_parameter_defaults_are_inferred() -> None:
    _app()
    panel = ParameterPanel()
    panel.set_schema({"iterations": 3, "normalize": True, "periods": [12.0]})
    assert isinstance(panel.editors["iterations"], QtWidgets.QSpinBox)
    assert isinstance(panel.editors["normalize"], QtWidgets.QCheckBox)
    assert isinstance(panel.editors["periods"], ListEditor)
    assert panel.values() == {"iterations": 3, "normalize": True, "periods": [12.0]}


def test_manual_and_percentile_visualization_do_not_modify_result() -> None:
    source = np.array([[0.0, 1.0], [2.0, 100.0]], np.float32)
    before = source.copy()
    transform = VisualizationTransform()
    manual = VisualizationSettings(
        range_mode=RangeMode.MANUAL,
        manual_minimum=0.0,
        manual_maximum=2.0,
        heatmap=False,
    )
    rendered = transform.render(source, manual)
    assert rendered.display_range == (0.0, 2.0)
    assert rendered.image.shape == (2, 2, 3)
    assert np.array_equal(source, before)

    percentile = VisualizationSettings(
        range_mode=RangeMode.PERCENTILE,
        percentile_low=0.0,
        percentile_high=75.0,
    )
    transform.render(source, percentile)
    assert np.array_equal(source, before)


def test_locked_range_is_shared_between_comparison_results() -> None:
    transform = VisualizationTransform()
    settings = VisualizationSettings(range_mode=RangeMode.AUTO, lock_range=True)
    first = transform.render(np.array([[0.0, 10.0]], np.float32), settings)
    second = transform.render(np.array([[100.0, 200.0]], np.float32), settings)
    assert first.display_range == (0.0, 10.0)
    assert second.display_range == first.display_range
    transform.reset_range()
    third = transform.render(np.array([[100.0, 200.0]], np.float32), settings)
    assert third.display_range == (100.0, 200.0)


def test_visualization_range_excludes_invalid_mask_pixels() -> None:
    transform = VisualizationTransform()
    settings = VisualizationSettings(range_mode=RangeMode.AUTO, heatmap=False)
    response = np.array([[0.0, 0.0], [10.0, 20.0]], np.float32)
    valid = np.array([[False, False], [True, True]])
    rendered = transform.render(response, settings, valid_mask=valid)
    assert rendered.display_range == (10.0, 20.0)



def test_original_tab_does_not_submit_processing(monkeypatch) -> None:
    from defleco_lab.camera.frame_packet import FramePacket
    from defleco_lab.gui.main_window import MainWindow

    app = _app()
    window = MainWindow()
    calls = []
    monkeypatch.setattr(window.processor, "submit", lambda request: calls.append(request))
    packet = FramePacket(np.zeros((32, 32), np.uint8), 1, 1, source="synthetic")

    window.tabs.setCurrentIndex(0)
    window._receive_packet(packet)
    assert calls == []

    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert calls

    window.close()
    app.processEvents()


def test_mono8_overlay_preserves_native_brightness():
    original = np.array([[10, 20], [30, 40]], np.uint8)
    response = np.zeros((2, 2), np.float32)
    transform = VisualizationTransform()
    settings = VisualizationSettings(
        range_mode=RangeMode.MANUAL,
        manual_minimum=0.0,
        manual_maximum=1.0,
        heatmap=False,
        overlay_alpha=0.0,
    )
    rendered = transform.render(response, settings, original=original)
    assert np.array_equal(rendered.image[..., 0], original)
    assert np.array_equal(rendered.image[..., 1], original)
    assert np.array_equal(rendered.image[..., 2], original)



def test_parameter_panel_hides_irrelevant_conditional_controls() -> None:
    app = _app()
    panel = ParameterPanel()
    panel.set_schema(
        {
            "output": {"default": "magnitude", "choices": ["magnitude", "residual"]},
            "residual_sigma": {
                "default": 8.0,
                "visible_if": {"output": "residual"},
            },
        }
    )
    app.processEvents()
    assert not panel.editors["residual_sigma"].isVisible()
    panel.set_values({"output": "residual"})
    panel.show()
    app.processEvents()
    assert panel.editors["residual_sigma"].isVisible()
    panel.close()


def test_camera_stop_failure_keeps_worker_and_blocks_restart(monkeypatch) -> None:
    from defleco_lab.gui.main_window import MainWindow

    app = _app()
    window = MainWindow()

    class StuckWorker:
        def stop(self) -> bool:
            return False

    worker = StuckWorker()
    window.camera_worker = worker
    window.source_combo.setCurrentText("Basler")
    window._start()

    assert window.camera_worker is worker
    assert window.state_label.text() == "ERROR"
    assert "reconnect blocked" in window.statusBar().currentMessage()

    window.camera_worker = None
    window.close()
    app.processEvents()
