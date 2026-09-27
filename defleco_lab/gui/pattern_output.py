"""Integrated display-pattern generator for physical deflectometry experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True, slots=True)
class PatternSettings:
    family: str = "stripes"
    period_px: float = 24.0
    angle_deg: float = 0.0
    phase_deg: float = 0.0
    brightness: float = 100.0
    invert: bool = False
    waveform: str = "binary"
    mode: str = "static"
    speed_deg_s: float = 20.0
    step_deg: float = 15.0
    render_scale: float = 0.5
    animation_fps: int = 20

    def as_dict(self) -> dict:
        return asdict(self)


def render_pattern(width: int, height: int, settings: PatternSettings) -> np.ndarray:
    """Render one deterministic Mono8 pattern frame."""

    width = max(2, int(width))
    height = max(2, int(height))
    period = max(2.0, float(settings.period_px))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x = xx - (width - 1) / 2.0
    y = yy - (height - 1) / 2.0
    angle = math.radians(float(settings.angle_deg))
    phase = math.radians(float(settings.phase_deg))
    c, s = math.cos(angle), math.sin(angle)
    xr = c * x + s * y
    yr = -s * x + c * y
    family = settings.family

    if family == "checker":
        sx = np.sin(np.pi * xr / period)
        sy = np.sin(np.pi * yr / period)
        signal = (sx * sy >= 0).astype(np.float32)
    elif family == "rings":
        radius = np.hypot(x, y)
        wave = np.sin(2.0 * np.pi * radius / period + phase)
        signal = _wave_to_level(wave, settings.waveform)
    elif family == "spiral":
        radius = np.hypot(x, y)
        theta = np.arctan2(y, x) - angle
        # One-arm Archimedean-like carrier.  The radial period controls fineness.
        wave = np.sin(2.0 * np.pi * radius / period - theta + phase)
        signal = _wave_to_level(wave, settings.waveform)
    else:
        wave = np.sin(2.0 * np.pi * xr / period + phase)
        signal = _wave_to_level(wave, settings.waveform)

    if settings.invert:
        signal = 1.0 - signal
    peak = round(np.clip(float(settings.brightness), 0.0, 100.0) * 255.0 / 100.0)
    return np.rint(np.clip(signal * peak, 0, 255)).astype(np.uint8)


def _wave_to_level(wave: np.ndarray, waveform: str) -> np.ndarray:
    if waveform == "sinusoidal":
        return (wave + 1.0) * 0.5
    return (wave >= 0).astype(np.float32)


class PatternOutputWindow(QtWidgets.QLabel):
    """Borderless full-screen pattern surface intended for the external display."""

    stateChanged = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Defleco LAB - Pattern Output")
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: black;")
        self._settings = PatternSettings()
        self._effective_angle = self._settings.angle_deg
        self._effective_phase = self._settings.phase_deg
        self._elapsed = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._animate)

    @property
    def effective_settings(self) -> PatternSettings:
        return replace(
            self._settings,
            angle_deg=self._effective_angle,
            phase_deg=self._effective_phase,
        )

    def set_settings(self, settings: PatternSettings) -> None:
        reset_phase = (
            settings.family != self._settings.family
            or settings.mode != self._settings.mode
            or settings.angle_deg != self._settings.angle_deg
            or settings.phase_deg != self._settings.phase_deg
        )
        self._settings = settings
        self._timer.setInterval(max(16, round(1000 / max(1, int(settings.animation_fps)))))
        if reset_phase:
            self._effective_angle = settings.angle_deg
            self._effective_phase = settings.phase_deg
        if settings.mode == "continuous" and self.isVisible():
            self._elapsed.restart()
            self._timer.start()
        else:
            self._timer.stop()
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def show_on_screen(self, screen_index: int) -> None:
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            self.showFullScreen()
        else:
            index = max(0, min(int(screen_index), len(screens) - 1))
            screen = screens[index]
            self.setGeometry(screen.geometry())
            self.show()
            if self.windowHandle() is not None:
                self.windowHandle().setScreen(screen)
            self.setGeometry(screen.geometry())
            self.showFullScreen()
        if self._settings.mode == "continuous":
            self._elapsed.restart()
            self._timer.start()
        self._redraw()

    def hide_output(self) -> None:
        self._timer.stop()
        self.hide()

    def step(self) -> None:
        delta = float(self._settings.step_deg)
        if self._settings.family == "rings":
            self._effective_phase = (self._effective_phase + delta) % 360.0
        else:
            self._effective_angle = (self._effective_angle + delta) % 180.0
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._redraw()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.hide_output()
            event.accept()
            return
        super().keyPressEvent(event)

    def _animate(self) -> None:
        if not self._elapsed.isValid():
            self._elapsed.start()
            return
        seconds = self._elapsed.restart() / 1000.0
        delta = float(self._settings.speed_deg_s) * seconds
        if self._settings.family == "rings":
            self._effective_phase = (self._effective_phase + delta) % 360.0
        else:
            self._effective_angle = (self._effective_angle + delta) % 180.0
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def _redraw(self) -> None:
        if self.width() < 2 or self.height() < 2:
            return
        effective = self.effective_settings
        render_scale = min(1.0, max(0.1, float(effective.render_scale)))
        render_width = max(2, round(self.width() * render_scale))
        render_height = max(2, round(self.height() * render_scale))
        rendered_settings = replace(
            effective,
            period_px=max(2.0, effective.period_px * render_scale),
        )
        data = render_pattern(render_width, render_height, rendered_settings)
        image = QtGui.QImage(
            data.data,
            data.shape[1],
            data.shape[0],
            data.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        if render_width != self.width() or render_height != self.height():
            transform = (
                QtCore.Qt.TransformationMode.SmoothTransformation
                if effective.waveform == "sinusoidal"
                else QtCore.Qt.TransformationMode.FastTransformation
            )
            pixmap = pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                transform,
            )
        self.setPixmap(pixmap)


class PatternControlPanel(QtWidgets.QGroupBox):
    """Operator-friendly pattern controls with an independent full-screen output."""

    stateChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Pattern output", parent)
        self.output = PatternOutputWindow()
        self.output.stateChanged.connect(self.stateChanged)

        form = QtWidgets.QFormLayout(self)
        help_text = QtWidgets.QLabel(
            "Pattern is rendered on a selected display. Start with fine stripes, "
            "checkerboard or a single fine spiral. Esc closes the pattern output."
        )
        help_text.setWordWrap(True)
        form.addRow(help_text)

        self.family = QtWidgets.QComboBox()
        self.family.addItem("Fine stripes", "stripes")
        self.family.addItem("Checkerboard", "checker")
        self.family.addItem("Single spiral", "spiral")
        self.family.addItem("Concentric rings", "rings")
        self.family.setToolTip(
            "Physical reference pattern reflected by the painted surface. "
            "The names describe the generated pattern, not an analysis method."
        )

        self.period = QtWidgets.QDoubleSpinBox()
        self.period.setRange(4.0, 500.0)
        self.period.setValue(24.0)
        self.period.setSingleStep(2.0)
        self.period.setSuffix(" px")
        self.period.setToolTip(
            "Pattern fineness in display pixels. Smaller values make a finer pattern."
        )

        self.waveform = QtWidgets.QComboBox()
        self.waveform.addItem("Binary / sharp", "binary")
        self.waveform.addItem("Sinusoidal", "sinusoidal")
        self.waveform.setToolTip(
            "Binary gives sharp visible edges; sinusoidal is useful for phase-oriented methods."
        )

        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(0.0, 179.9)
        self.angle.setValue(0.0)
        self.angle.setSingleStep(5.0)
        self.angle.setSuffix(" deg")
        self.angle.setToolTip("Static starting orientation of stripes/checker/spiral.")

        self.phase = QtWidgets.QDoubleSpinBox()
        self.phase.setRange(0.0, 359.9)
        self.phase.setValue(0.0)
        self.phase.setSingleStep(10.0)
        self.phase.setSuffix(" deg")
        self.phase.setToolTip("Carrier phase, mainly useful for sinusoidal/ring experiments.")

        self.brightness = QtWidgets.QSpinBox()
        self.brightness.setRange(1, 100)
        self.brightness.setValue(100)
        self.brightness.setSuffix(" %")
        self.brightness.setToolTip(
            "Digital white level of the generated pattern. Display processing/HDR should remain disabled."
        )

        self.invert = QtWidgets.QCheckBox("Invert black/white")
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("Static", "static")
        self.mode.addItem("Stepped", "step")
        self.mode.addItem("Continuous rotation", "continuous")
        self.mode.setToolTip(
            "Stepped mode is preferred for reproducible capture; continuous mode is useful for visual screening."
        )

        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(-360.0, 360.0)
        self.speed.setValue(20.0)
        self.speed.setSingleStep(5.0)
        self.speed.setSuffix(" deg/s")
        self.speed.setToolTip("Continuous pattern rotation/phase speed.")

        self.step_size = QtWidgets.QDoubleSpinBox()
        self.step_size.setRange(0.1, 180.0)
        self.step_size.setValue(15.0)
        self.step_size.setSingleStep(5.0)
        self.step_size.setSuffix(" deg")
        self.step_size.setToolTip("Angle/phase increment for one reproducible step.")

        self.render_scale = QtWidgets.QComboBox()
        self.render_scale.addItem("100% (full display resolution)", 1.0)
        self.render_scale.addItem("50% (recommended)", 0.5)
        self.render_scale.addItem("25% (fast preview)", 0.25)
        self.render_scale.setCurrentIndex(1)
        self.render_scale.setToolTip(
            "Internal pattern rendering resolution. Period remains defined in display pixels; "
            "50% greatly reduces continuous-animation load on a 4K display."
        )

        self.animation_fps = QtWidgets.QSpinBox()
        self.animation_fps.setRange(5, 60)
        self.animation_fps.setValue(20)
        self.animation_fps.setSuffix(" FPS")
        self.animation_fps.setToolTip(
            "Requested software redraw rate for continuous pattern motion. This is not "
            "camera/display hardware synchronization."
        )

        self.screen = QtWidgets.QComboBox()
        self.refresh_screens()

        self.show_button = QtWidgets.QPushButton("Show on display")
        self.hide_button = QtWidgets.QPushButton("Hide pattern")
        self.step_button = QtWidgets.QPushButton("Next step")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.show_button)
        buttons.addWidget(self.hide_button)
        buttons.addWidget(self.step_button)

        form.addRow("Pattern", self.family)
        form.addRow("Period / fineness", self.period)
        form.addRow("Waveform", self.waveform)
        form.addRow("Angle", self.angle)
        form.addRow("Phase", self.phase)
        form.addRow("Brightness", self.brightness)
        form.addRow(self.invert)
        form.addRow("Mode", self.mode)
        form.addRow("Speed", self.speed)
        form.addRow("Step", self.step_size)
        form.addRow("Pattern render quality", self.render_scale)
        form.addRow("Animation redraw", self.animation_fps)
        form.addRow("Output display", self.screen)
        form.addRow(buttons)

        for widget in (
            self.family,
            self.period,
            self.waveform,
            self.angle,
            self.phase,
            self.brightness,
            self.invert,
            self.mode,
            self.speed,
            self.step_size,
            self.render_scale,
            self.animation_fps,
        ):
            _connect(widget, self._settings_changed)
        self.show_button.clicked.connect(self.show_output)
        self.hide_button.clicked.connect(self.output.hide_output)
        self.step_button.clicked.connect(self.output.step)
        self._update_context()
        self._settings_changed()

    def refresh_screens(self) -> None:
        current = self.screen.currentData() if hasattr(self, "screen") else None
        self.screen.clear()
        screens = QtGui.QGuiApplication.screens()
        for index, screen in enumerate(screens):
            geometry = screen.geometry()
            self.screen.addItem(
                f"{index + 1}: {screen.name()} ({geometry.width()}x{geometry.height()})",
                index,
            )
        preferred = 1 if len(screens) > 1 else 0
        index = self.screen.findData(current if current is not None else preferred)
        self.screen.setCurrentIndex(max(0, index))

    def settings(self) -> PatternSettings:
        return PatternSettings(
            family=str(self.family.currentData()),
            period_px=self.period.value(),
            angle_deg=self.angle.value(),
            phase_deg=self.phase.value(),
            brightness=float(self.brightness.value()),
            invert=self.invert.isChecked(),
            waveform=str(self.waveform.currentData()),
            mode=str(self.mode.currentData()),
            speed_deg_s=self.speed.value(),
            step_deg=self.step_size.value(),
            render_scale=float(self.render_scale.currentData()),
            animation_fps=self.animation_fps.value(),
        )

    def current_effective_settings(self) -> PatternSettings:
        return self.output.effective_settings

    def show_output(self) -> None:
        self.output.set_settings(self.settings())
        index = self.screen.currentData()
        self.output.show_on_screen(0 if index is None else int(index))

    def shutdown(self) -> None:
        self.output.hide_output()
        self.output.close()

    def _settings_changed(self, *_args: object) -> None:
        self._update_context()
        self.output.set_settings(self.settings())

    def _update_context(self) -> None:
        mode = str(self.mode.currentData())
        continuous = mode == "continuous"
        stepped = mode == "step"
        self.speed.setVisible(continuous)
        self.animation_fps.setVisible(continuous)
        self.step_size.setVisible(stepped)
        self.step_button.setVisible(stepped)
        layout = self.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            for editor, visible in (
                (self.speed, continuous),
                (self.animation_fps, continuous),
                (self.step_size, stepped),
            ):
                label = layout.labelForField(editor)
                if label is not None:
                    label.setVisible(visible)


def _connect(widget: QtWidgets.QWidget, callback: object) -> None:
    if isinstance(widget, QtWidgets.QComboBox):
        widget.currentIndexChanged.connect(callback)
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.valueChanged.connect(callback)
    elif isinstance(widget, QtWidgets.QCheckBox):
        widget.toggled.connect(callback)
