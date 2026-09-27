"""Contextual pipeline controls used by the main laboratory window."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def _double(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    decimals: int = 2,
    tooltip: str = "",
) -> QtWidgets.QDoubleSpinBox:
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setValue(value)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setToolTip(tooltip)
    return w


class PreprocessingPanel(QtWidgets.QGroupBox):
    settingsChanged = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("1. Pre-processing", parent)
        form = QtWidgets.QFormLayout(self)

        hint = QtWidgets.QLabel(
            "Changes the camera image before the selected analysis method. "
            "Leave values at their defaults for an unmodified baseline."
        )
        hint.setWordWrap(True)
        hint.setToolTip(
            "Pre-processing changes the numerical input. It is different from "
            "display colour, transparency and display-range controls."
        )
        form.addRow(hint)

        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(["100%", "50%", "25%"])
        self.scale.setCurrentText("50%")
        self.scale.setToolTip(
            "Processing resolution. 50% is a practical live starting point. "
            "Pixel-valued method parameters refer to this processed grid."
        )
        self.contrast = _double(
            0.10,
            5.0,
            1.0,
            0.05,
            tooltip="Input contrast around mid-grey. 1.0 leaves contrast unchanged.",
        )
        self.brightness = _double(
            -127.0,
            127.0,
            0.0,
            1.0,
            1,
            "Additive brightness offset in approximately 8-bit intensity levels.",
        )
        self.gamma = _double(
            0.20,
            5.0,
            1.0,
            0.05,
            tooltip="Input gamma. 1.0 leaves the tonal curve unchanged.",
        )
        self.blur = _double(
            0.0,
            20.0,
            0.0,
            0.1,
            tooltip="Optional Gaussian denoising before analysis; 0 disables it.",
        )
        self.clahe = QtWidgets.QCheckBox("Enable")
        self.clahe.setToolTip(
            "Optional local contrast equalisation. Useful to test, but keep it off "
            "when comparing raw pattern physics."
        )
        self.clahe_clip = _double(
            0.1, 20.0, 2.0, 0.1, tooltip="CLAHE clipping strength."
        )
        self.clahe_grid = QtWidgets.QSpinBox()
        self.clahe_grid.setRange(2, 32)
        self.clahe_grid.setValue(8)
        self.clahe_grid.setToolTip("CLAHE tile grid size.")

        form.addRow("Processing scale", self.scale)
        form.addRow("Contrast", self.contrast)
        form.addRow("Brightness", self.brightness)
        form.addRow("Gamma", self.gamma)
        form.addRow("Gaussian blur sigma", self.blur)
        form.addRow("CLAHE", self.clahe)
        form.addRow("CLAHE clip", self.clahe_clip)
        form.addRow("CLAHE grid", self.clahe_grid)

        for widget in (
            self.scale,
            self.contrast,
            self.brightness,
            self.gamma,
            self.blur,
            self.clahe,
            self.clahe_clip,
            self.clahe_grid,
        ):
            _connect(widget, self._changed)
        self._update_context()

    def values(self) -> dict:
        return {
            "contrast": self.contrast.value(),
            "brightness": self.brightness.value(),
            "gamma": self.gamma.value(),
            "blur_sigma": self.blur.value(),
            "clahe": self.clahe.isChecked(),
            "clahe_clip": self.clahe_clip.value(),
            "clahe_grid": self.clahe_grid.value(),
        }

    def processing_scale(self) -> float:
        return {"100%": 1.0, "50%": 0.5, "25%": 0.25}[self.scale.currentText()]

    def set_processing_scale(self, scale: float) -> None:
        label = f"{int(round(float(scale) * 100))}%"
        if self.scale.findText(label) >= 0:
            self.scale.setCurrentText(label)

    def _changed(self, *_args: object) -> None:
        self._update_context()
        self.settingsChanged.emit(self.values())

    def _update_context(self) -> None:
        enabled = self.clahe.isChecked()
        self.clahe_clip.setVisible(enabled)
        self.clahe_grid.setVisible(enabled)
        layout = self.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            for editor in (self.clahe_clip, self.clahe_grid):
                label = layout.labelForField(editor)
                if label is not None:
                    label.setVisible(enabled)


class PostprocessingPanel(QtWidgets.QGroupBox):
    settingsChanged = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("3. Response post-processing", parent)
        form = QtWidgets.QFormLayout(self)

        hint = QtWidgets.QLabel(
            "Optional cleanup of the numerical response after the analysis method. "
            "This remains separate from colour and transparency used only for viewing."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        self.local_background = _double(
            0.0,
            500.0,
            0.0,
            0.5,
            tooltip=(
                "Subtract a locally smooth response background. 0 disables it. "
                "Useful for suppressing slowly varying predictable structure."
            ),
        )
        self.absolute = QtWidgets.QCheckBox("Use absolute response")
        self.absolute.setToolTip(
            "Discard the sign after local-background subtraction. Often useful for "
            "defect saliency, but keep off when response direction matters."
        )
        self.smooth = _double(
            0.0,
            50.0,
            0.0,
            0.1,
            tooltip="Optional Gaussian smoothing of the response; 0 disables it.",
        )
        self.gain = _double(
            0.01,
            100.0,
            1.0,
            0.1,
            tooltip=(
                "Numeric response gain. This affects exported/next-stage data; "
                "display contrast can instead be changed in section 4."
            ),
        )

        form.addRow("Local background sigma", self.local_background)
        form.addRow(self.absolute)
        form.addRow("Response smoothing sigma", self.smooth)
        form.addRow("Response gain", self.gain)

        for widget in (self.local_background, self.absolute, self.smooth, self.gain):
            _connect(widget, self._changed)

    def values(self) -> dict:
        return {
            "local_background_sigma": self.local_background.value(),
            "absolute": self.absolute.isChecked(),
            "smooth_sigma": self.smooth.value(),
            "gain": self.gain.value(),
        }

    def _changed(self, *_args: object) -> None:
        self.settingsChanged.emit(self.values())


def _connect(widget: QtWidgets.QWidget, callback: object) -> None:
    if isinstance(widget, QtWidgets.QComboBox):
        widget.currentIndexChanged.connect(callback)
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.valueChanged.connect(callback)
    elif isinstance(widget, QtWidgets.QCheckBox):
        widget.toggled.connect(callback)
