"""Display-only response mapping and controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets


class RangeMode(StrEnum):
    AUTO = "Auto min/max"
    PERCENTILE = "Percentile"
    MANUAL = "Manual"


COLORMAPS = {
    "Turbo": cv2.COLORMAP_TURBO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Jet": cv2.COLORMAP_JET,
}


@dataclass(frozen=True, slots=True)
class VisualizationSettings:
    range_mode: RangeMode = RangeMode.PERCENTILE
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    manual_minimum: float = 0.0
    manual_maximum: float = 1.0
    heatmap: bool = True
    colormap: str = "Turbo"
    overlay_alpha: float = 1.0
    lock_range: bool = False


@dataclass(frozen=True, slots=True)
class VisualizationFrame:
    image: np.ndarray
    display_range: tuple[float, float]


class VisualizationTransform:
    """Stateful only for an optional shared locked display range."""

    def __init__(self) -> None:
        self._locked_range: tuple[float, float] | None = None

    @property
    def locked_range(self) -> tuple[float, float] | None:
        return self._locked_range

    def reset_range(self) -> None:
        self._locked_range = None

    def render(
        self,
        response: np.ndarray,
        settings: VisualizationSettings,
        original: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> VisualizationFrame:
        if not settings.lock_range:
            self._locked_range = None
        source = np.asarray(response)
        finite = np.asarray(source, dtype=np.float32).copy()
        valid = np.isfinite(finite)
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != finite.shape[:2]:
                raise ValueError("valid mask shape must match the response")
            valid &= mask
        finite[~valid] = 0.0
        display_range = self._resolve_range(finite, valid, settings)
        lo, hi = display_range
        normalized = np.clip((finite - lo) * (255.0 / max(hi - lo, 1e-12)), 0, 255)
        gray = normalized.astype(np.uint8)
        if settings.heatmap:
            colormap = COLORMAPS.get(settings.colormap, cv2.COLORMAP_TURBO)
            response_rgb = cv2.cvtColor(cv2.applyColorMap(gray, colormap), cv2.COLOR_BGR2RGB)
        else:
            response_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        response_rgb[~valid] = 0

        alpha = float(np.clip(settings.overlay_alpha, 0.0, 1.0))
        if original is not None and alpha < 1.0:
            base = _original_rgb(original, response_rgb.shape[:2])
            response_rgb = cv2.addWeighted(base, 1.0 - alpha, response_rgb, alpha, 0.0)
        return VisualizationFrame(np.ascontiguousarray(response_rgb), display_range)

    def _resolve_range(
        self,
        finite: np.ndarray,
        valid: np.ndarray,
        settings: VisualizationSettings,
    ) -> tuple[float, float]:
        if settings.lock_range and self._locked_range is not None:
            return self._locked_range
        values = finite[valid]
        if settings.range_mode == RangeMode.MANUAL:
            display_range = (float(settings.manual_minimum), float(settings.manual_maximum))
        elif values.size == 0:
            display_range = (0.0, 1.0)
        elif settings.range_mode == RangeMode.PERCENTILE:
            display_range = (
                float(np.percentile(values, settings.percentile_low)),
                float(np.percentile(values, settings.percentile_high)),
            )
        else:
            display_range = (float(values.min()), float(values.max()))
        if not np.isfinite(display_range).all() or display_range[1] <= display_range[0]:
            center = display_range[0] if np.isfinite(display_range[0]) else 0.0
            display_range = (center, center + 1.0)
        if settings.lock_range:
            self._locked_range = display_range
        return display_range


def _original_rgb(original: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(original)
    if source.ndim == 2:
        # Preserve native Mono8 levels. Per-frame min/max stretching makes live
        # overlays flicker and destroys brightness comparability between patterns.
        if source.dtype == np.uint8:
            values = source
        else:
            values32 = np.nan_to_num(source.astype(np.float32))
            if np.issubdtype(source.dtype, np.integer):
                info = np.iinfo(source.dtype)
                values = np.clip(
                    (values32 - info.min) * 255.0 / max(info.max - info.min, 1),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                lo, hi = float(values32.min()), float(values32.max())
                values = np.clip(
                    (values32 - lo) * 255.0 / max(hi - lo, 1e-12),
                    0,
                    255,
                ).astype(np.uint8)
        source = cv2.cvtColor(values, cv2.COLOR_GRAY2RGB)
    elif source.ndim == 3 and source.shape[2] == 3:
        source = (
            np.clip(source, 0, 255).astype(np.uint8) if source.dtype != np.uint8 else source.copy()
        )
    else:
        raise ValueError("original image must be grayscale or RGB")
    if source.shape[:2] != shape:
        source = cv2.resize(source, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(source)


class VisualizationPanel(QtWidgets.QGroupBox):
    settingsChanged = QtCore.Signal(object)
    rangeReset = QtCore.Signal()
    compareReferenceRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Response visualization", parent)
        form = QtWidgets.QFormLayout(self)
        self.range_mode = QtWidgets.QComboBox()
        self.range_mode.addItems([mode.value for mode in RangeMode])
        self.range_mode.setCurrentText(RangeMode.PERCENTILE.value)
        self.low_percentile = _double_spin(0.0, 100.0, 1.0, 2, 1.0)
        self.high_percentile = _double_spin(0.0, 100.0, 1.0, 2, 99.0)
        self.manual_minimum = _double_spin(-1e12, 1e12, 0.1, 6, 0.0)
        self.manual_maximum = _double_spin(-1e12, 1e12, 0.1, 6, 1.0)
        self.heatmap = QtWidgets.QCheckBox("Enabled")
        self.heatmap.setChecked(True)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(COLORMAPS)
        self.alpha = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(100)
        self.alpha.setToolTip("0% shows the original; 100% shows the response")
        self.lock_range = QtWidgets.QCheckBox("Use the same range for comparisons")
        self.reset_button = QtWidgets.QPushButton("Reset display range")
        self.compare_button = QtWidgets.QPushButton("Set current as compare reference")
        form.addRow("Range", self.range_mode)
        form.addRow("Low percentile", self.low_percentile)
        form.addRow("High percentile", self.high_percentile)
        form.addRow("Manual minimum", self.manual_minimum)
        form.addRow("Manual maximum", self.manual_maximum)
        form.addRow("Heatmap", self.heatmap)
        form.addRow("Colormap", self.colormap)
        form.addRow("Response alpha", self.alpha)
        form.addRow(self.lock_range)
        form.addRow(self.reset_button)
        form.addRow(self.compare_button)
        for widget in (
            self.range_mode,
            self.low_percentile,
            self.high_percentile,
            self.manual_minimum,
            self.manual_maximum,
            self.heatmap,
            self.colormap,
            self.alpha,
            self.lock_range,
        ):
            _connect_change(widget, self._changed)
        self.reset_button.clicked.connect(self._reset)
        self.compare_button.clicked.connect(self.compareReferenceRequested)
        self._update_enabled()

    def settings(self) -> VisualizationSettings:
        return VisualizationSettings(
            range_mode=RangeMode(self.range_mode.currentText()),
            percentile_low=self.low_percentile.value(),
            percentile_high=self.high_percentile.value(),
            manual_minimum=self.manual_minimum.value(),
            manual_maximum=self.manual_maximum.value(),
            heatmap=self.heatmap.isChecked(),
            colormap=self.colormap.currentText(),
            overlay_alpha=self.alpha.value() / 100.0,
            lock_range=self.lock_range.isChecked(),
        )

    def _changed(self, *_args: object) -> None:
        self._update_enabled()
        settings = self.settings()
        if settings.percentile_low >= settings.percentile_high:
            if settings.percentile_low >= 100.0:
                settings = replace(settings, percentile_low=99.99, percentile_high=100.0)
            else:
                settings = replace(settings, percentile_high=settings.percentile_low + 0.01)
        self.settingsChanged.emit(settings)

    def _reset(self) -> None:
        self.rangeReset.emit()
        self.settingsChanged.emit(self.settings())

    def _update_enabled(self) -> None:
        mode = RangeMode(self.range_mode.currentText())
        percentile = mode == RangeMode.PERCENTILE
        manual = mode == RangeMode.MANUAL
        self.low_percentile.setEnabled(percentile)
        self.high_percentile.setEnabled(percentile)
        self.manual_minimum.setEnabled(manual)
        self.manual_maximum.setEnabled(manual)
        self.colormap.setEnabled(self.heatmap.isChecked())


def _double_spin(
    minimum: float, maximum: float, step: float, decimals: int, value: float
) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setSingleStep(step)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def _connect_change(widget: QtWidgets.QWidget, callback: object) -> None:
    if isinstance(widget, QtWidgets.QComboBox):
        widget.currentIndexChanged.connect(callback)
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.valueChanged.connect(callback)
    elif isinstance(widget, QtWidgets.QCheckBox):
        widget.toggled.connect(callback)
    elif isinstance(widget, QtWidgets.QSlider):
        widget.valueChanged.connect(callback)
