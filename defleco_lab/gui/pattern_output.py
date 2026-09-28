"""Integrated display-pattern generator for physical deflectometry experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True, slots=True)
class PatternSettings:
    family: str = "stripes"
    period_px: float = 50.0
    angle_deg: float = 0.0
    phase_deg: float = 0.0
    brightness: float = 100.0
    invert: bool = False
    waveform: str = "binary"
    mode: str = "static"
    speed_deg_s: float = 20.0
    phase_speed_cycles_s: float = 0.0
    breathing_depth_percent: float = 0.0
    breathing_hz: float = 0.5
    step_deg: float = 15.0
    step_target: str = "angle"
    center_x_percent: float = 0.0
    center_y_percent: float = 0.0
    phase_tilt_x_deg: float = 0.0
    phase_tilt_y_deg: float = 0.0
    render_scale: float = 0.5
    animation_fps: int = 20
    duty_percent: float = 50.0
    squircle_power: float = 6.0
    spiral_arms: int = 1
    spiral_width_percent: float = 20.0
    spokes: int = 18
    speckle_size_px: int = 24
    seed: int = 12345

    def as_dict(self) -> dict:
        return asdict(self)


def render_pattern(width: int, height: int, settings: PatternSettings) -> np.ndarray:
    """Render one deterministic Mono8 pattern frame.

    Pattern families and parameter semantics intentionally follow the established
    standalone GPixel Deflecto dynamic-pattern PoC where practical.
    """

    width = max(2, int(width))
    height = max(2, int(height))
    period = max(2.0, float(settings.period_px))
    duty = np.clip(float(settings.duty_percent) / 100.0, 0.01, 0.99)
    phase_cycles = (float(settings.phase_deg) % 360.0) / 360.0

    center_x = (width - 1) / 2.0 + width * float(settings.center_x_percent) / 100.0
    center_y = (height - 1) / 2.0 + height * float(settings.center_y_percent) / 100.0
    x = (np.arange(width, dtype=np.float32) - center_x)[None, :]
    y = (np.arange(height, dtype=np.float32) - center_y)[:, None]
    phase_field: float | np.ndarray = phase_cycles
    if float(settings.phase_tilt_x_deg) != 0.0:
        phase_field = phase_field + (
            float(settings.phase_tilt_x_deg) / 360.0
        ) * x / max(1.0, float(width - 1))
    if float(settings.phase_tilt_y_deg) != 0.0:
        phase_field = phase_field + (
            float(settings.phase_tilt_y_deg) / 360.0
        ) * y / max(1.0, float(height - 1))
    angle = math.radians(float(settings.angle_deg))
    c_angle, s_angle = math.cos(angle), math.sin(angle)
    family = settings.family

    xr = None
    yr = None
    if family in {
        "stripes",
        "checker",
        "composite",
        "nested_square",
        "squircle",
    }:
        xr = c_angle * x + s_angle * y
        if family != "stripes":
            yr = -s_angle * x + c_angle * y

    if family == "checker":
        assert xr is not None and yr is not None
        ix = np.floor(xr / period + phase_field).astype(np.int32)
        iy = np.floor(yr / period + phase_field).astype(np.int32)
        signal = ((ix + iy) % 2 == 0).astype(np.float32)
    elif family == "rings":
        radius = np.hypot(x, y)
        cycles = radius / period + phase_field
        signal = _carrier(cycles, settings.waveform, duty)
    elif family == "composite":
        assert xr is not None and yr is not None
        x_signal = _carrier(xr / period + phase_field, settings.waveform, duty)
        y_signal = _carrier(yr / period + phase_field, settings.waveform, duty)
        signal = 0.5 * (x_signal + y_signal)
    elif family == "nested_square":
        assert xr is not None and yr is not None
        metric = np.maximum(np.abs(xr), np.abs(yr))
        signal = _line_carrier(metric / period + phase_field, duty)
    elif family == "squircle":
        assert xr is not None and yr is not None
        power = max(2.0, float(settings.squircle_power))
        ax = np.abs(xr)
        ay = np.abs(yr)
        scale = np.maximum(ax, ay)
        safe = np.where(scale > 0, scale, 1.0)
        metric = scale * ((ax / safe) ** power + (ay / safe) ** power) ** (1.0 / power)
        signal = _line_carrier(metric / period + phase_field, duty)
    elif family in {"spiral", "counter_spiral"}:
        radius = np.hypot(x, y)
        theta = np.arctan2(y, x)
        arms = max(1, int(settings.spiral_arms))
        width_fraction = np.clip(
            float(settings.spiral_width_percent) / 100.0, 0.02, 0.9
        )
        spiral_cycles = radius / period - arms * (theta - angle) / (2.0 * np.pi)
        signal = _line_carrier(spiral_cycles + phase_field, width_fraction)
        if family == "counter_spiral":
            opposite = radius / period + arms * (theta + angle) / (2.0 * np.pi)
            signal = np.maximum(
                signal,
                _line_carrier(opposite + phase_field, width_fraction),
            )
    elif family == "starburst":
        theta = np.arctan2(y, x)
        spokes = max(2, int(settings.spokes))
        sectors = spokes * 2
        angular = np.mod(theta - angle + phase_field * (2.0 * np.pi / sectors), 2.0 * np.pi)
        sector = np.floor(angular / (2.0 * np.pi / sectors)).astype(np.int32)
        signal = (sector % 2 == 0).astype(np.float32)
    elif family == "speckle":
        block = max(1, int(settings.speckle_size_px))
        rows = int(np.ceil(height / block))
        cols = int(np.ceil(width / block))
        rng = np.random.default_rng(max(0, int(settings.seed)))
        cells = (rng.random((rows, cols)) >= 0.5).astype(np.float32)
        signal = np.repeat(np.repeat(cells, block, axis=0), block, axis=1)[:height, :width]
    elif family == "solid":
        signal = np.ones((height, width), np.float32)
    else:
        assert xr is not None
        cycles = xr / period + phase_field
        signal = _carrier(cycles, settings.waveform, duty)

    if settings.invert:
        signal = 1.0 - signal
    peak = round(np.clip(float(settings.brightness), 0.0, 100.0) * 255.0 / 100.0)
    return np.rint(np.clip(signal * peak, 0, 255)).astype(np.uint8)


def _carrier(cycles: np.ndarray, waveform: str, duty: float) -> np.ndarray:
    if waveform == "sinusoidal":
        return (np.sin(2.0 * np.pi * cycles) + 1.0) * 0.5
    return (np.mod(cycles, 1.0) < duty).astype(np.float32)


def _line_carrier(cycles: np.ndarray, width_fraction: float) -> np.ndarray:
    phase = np.mod(cycles, 1.0)
    distance = np.minimum(phase, 1.0 - phase)
    return (distance <= width_fraction * 0.5).astype(np.float32)



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
        self._breathing_phase = 0.0
        self._elapsed = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._animate)

    @property
    def effective_settings(self) -> PatternSettings:
        period = float(self._settings.period_px)
        if self._settings.mode == "continuous" and self._settings.breathing_depth_percent:
            depth = float(self._settings.breathing_depth_percent) / 100.0
            period *= max(0.05, 1.0 + depth * math.sin(self._breathing_phase))
        return replace(
            self._settings,
            period_px=period,
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
            self._breathing_phase = 0.0
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
        if self._settings.step_target == "phase" or self._settings.family == "rings":
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
        rotation_delta = float(self._settings.speed_deg_s) * seconds
        phase_delta = float(self._settings.phase_speed_cycles_s) * 360.0 * seconds
        if self._settings.family == "rings" and phase_delta == 0.0:
            self._effective_phase = (self._effective_phase + rotation_delta) % 360.0
        else:
            self._effective_angle = (self._effective_angle + rotation_delta) % 180.0
            self._effective_phase = (self._effective_phase + phase_delta) % 360.0
        self._breathing_phase = (
            self._breathing_phase
            + 2.0 * np.pi * float(self._settings.breathing_hz) * seconds
        ) % (2.0 * np.pi)
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def _redraw(self) -> None:
        if self.width() < 2 or self.height() < 2:
            return
        effective = self.effective_settings
        render_scale = min(1.0, max(0.1, float(effective.render_scale)))
        render_width = max(2, round(self.width() * render_scale))
        render_height = max(2, round(self.height() * render_scale))

        # QWidget geometry is expressed in Qt device-independent pixels. On a
        # Windows display using e.g. 150% scaling, 2160x3840 physical pixels are
        # typically exposed as 1440x2560 logical pixels. Keep period/speckle
        # controls defined in physical display pixels so experiments remain
        # reproducible regardless of desktop scaling or portrait orientation.
        dpr = max(0.1, float(self.devicePixelRatioF()))
        rendered_settings = replace(
            effective,
            period_px=max(2.0, effective.period_px * render_scale / dpr),
            speckle_size_px=max(
                1, round(effective.speckle_size_px * render_scale / dpr)
            ),
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
        self.family.addItem("Concentric rings", "rings")
        self.family.addItem("Composite X + Y", "composite")
        self.family.addItem("Nested squares", "nested_square")
        self.family.addItem("Squircle / superellipse", "squircle")
        self.family.addItem("Single spiral", "spiral")
        self.family.addItem("Counter-spiral pair", "counter_spiral")
        self.family.addItem("Starburst / spokes", "starburst")
        self.family.addItem("Pseudo-random speckle", "speckle")
        self.family.addItem("Solid field", "solid")
        self.family.setToolTip(
            "Physical reference pattern reflected by the painted surface. "
            "The names describe the generated pattern, not an analysis method."
        )
        self.pattern_hint = QtWidgets.QLabel()
        self.pattern_hint.setWordWrap(True)
        self.pattern_hint.setStyleSheet("color: #aeb6bf;")
        form.addRow(self.pattern_hint)

        self.period = QtWidgets.QDoubleSpinBox()
        self.period.setRange(4.0, 500.0)
        self.period.setValue(50.0)
        self.period.setSingleStep(2.0)
        self.period.setSuffix(" px")
        self.period.setToolTip(
            "Pattern fineness in physical display pixels. Smaller values make a finer pattern. "
            "Desktop scaling is compensated automatically."
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
        self.phase.setToolTip("Carrier phase / radial shift of the generated pattern.")

        self.center_x = QtWidgets.QDoubleSpinBox()
        self.center_x.setRange(-50.0, 50.0)
        self.center_x.setValue(0.0)
        self.center_x.setSingleStep(2.0)
        self.center_x.setSuffix(" %")
        self.center_x.setToolTip("Shift radial/spiral pattern centre horizontally.")

        self.center_y = QtWidgets.QDoubleSpinBox()
        self.center_y.setRange(-50.0, 50.0)
        self.center_y.setValue(0.0)
        self.center_y.setSingleStep(2.0)
        self.center_y.setSuffix(" %")
        self.center_y.setToolTip("Shift radial/spiral pattern centre vertically.")

        self.phase_tilt_x = QtWidgets.QDoubleSpinBox()
        self.phase_tilt_x.setRange(-720.0, 720.0)
        self.phase_tilt_x.setValue(0.0)
        self.phase_tilt_x.setSingleStep(15.0)
        self.phase_tilt_x.setSuffix(" deg")
        self.phase_tilt_x.setToolTip(
            "Virtual X tilt: total carrier-phase change from left to right."
        )

        self.phase_tilt_y = QtWidgets.QDoubleSpinBox()
        self.phase_tilt_y.setRange(-720.0, 720.0)
        self.phase_tilt_y.setValue(0.0)
        self.phase_tilt_y.setSingleStep(15.0)
        self.phase_tilt_y.setSuffix(" deg")
        self.phase_tilt_y.setToolTip(
            "Virtual Y tilt: total carrier-phase change from top to bottom."
        )

        self.duty = QtWidgets.QDoubleSpinBox()
        self.duty.setRange(5.0, 95.0)
        self.duty.setValue(50.0)
        self.duty.setSingleStep(5.0)
        self.duty.setSuffix(" %")
        self.duty.setToolTip(
            "Bright fraction or line width of binary pattern periods. "
            "For sinusoidal carriers this has no effect."
        )

        self.squircle_power = QtWidgets.QDoubleSpinBox()
        self.squircle_power.setRange(2.0, 32.0)
        self.squircle_power.setValue(6.0)
        self.squircle_power.setSingleStep(0.5)
        self.squircle_power.setToolTip(
            "Superellipse power: 2 is circular, higher values approach a square."
        )

        self.spiral_arms = QtWidgets.QSpinBox()
        self.spiral_arms.setRange(1, 12)
        self.spiral_arms.setValue(1)
        self.spiral_arms.setToolTip(
            "Number of Archimedean spiral arms. Start with one fine arm for the current PoC."
        )

        self.spiral_width = QtWidgets.QDoubleSpinBox()
        self.spiral_width.setRange(3.0, 90.0)
        self.spiral_width.setValue(20.0)
        self.spiral_width.setSingleStep(2.0)
        self.spiral_width.setSuffix(" %")
        self.spiral_width.setToolTip("Bright spiral-line width as a fraction of radial pitch.")

        self.spokes = QtWidgets.QSpinBox()
        self.spokes.setRange(2, 96)
        self.spokes.setValue(18)
        self.spokes.setToolTip("Number of bright radial spokes in the starburst pattern.")

        self.speckle_size = QtWidgets.QSpinBox()
        self.speckle_size.setRange(1, 256)
        self.speckle_size.setValue(24)
        self.speckle_size.setSuffix(" px")
        self.speckle_size.setToolTip(
            "Square speckle-cell size in physical display pixels; desktop scaling is compensated."
        )

        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(12345)
        self.seed.setToolTip("Deterministic pseudo-random seed for repeatable speckle patterns.")

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
        self.speed.setToolTip(
            "Continuous rotation speed. Set 0 to keep orientation fixed."
        )

        self.phase_speed = QtWidgets.QDoubleSpinBox()
        self.phase_speed.setRange(-5.0, 5.0)
        self.phase_speed.setValue(0.0)
        self.phase_speed.setSingleStep(0.05)
        self.phase_speed.setSuffix(" cycles/s")
        self.phase_speed.setToolTip(
            "Continuous carrier phase sweep. Can run together with rotation."
        )

        self.breathing_depth = QtWidgets.QDoubleSpinBox()
        self.breathing_depth.setRange(0.0, 80.0)
        self.breathing_depth.setValue(0.0)
        self.breathing_depth.setSingleStep(2.0)
        self.breathing_depth.setSuffix(" %")
        self.breathing_depth.setToolTip(
            "Period pulse/breathing depth. 0 disables it; 10-20% is a useful first experiment."
        )

        self.breathing_hz = QtWidgets.QDoubleSpinBox()
        self.breathing_hz.setRange(0.01, 10.0)
        self.breathing_hz.setValue(0.5)
        self.breathing_hz.setSingleStep(0.1)
        self.breathing_hz.setSuffix(" Hz")
        self.breathing_hz.setToolTip("Frequency of period pulsing/breathing.")

        self.step_size = QtWidgets.QDoubleSpinBox()
        self.step_size.setRange(0.1, 180.0)
        self.step_size.setValue(15.0)
        self.step_size.setSingleStep(5.0)
        self.step_size.setSuffix(" deg")
        self.step_size.setToolTip("Angle/phase increment for one reproducible step.")
        self.step_target = QtWidgets.QComboBox()
        self.step_target.addItem("Angle", "angle")
        self.step_target.addItem("Phase", "phase")
        self.step_target.setToolTip("Choose what the Next step button increments.")

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
        self.screen.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.screen.setMinimumContentsLength(12)
        self.screen.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
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
        form.addRow("Virtual phase tilt X", self.phase_tilt_x)
        form.addRow("Virtual phase tilt Y", self.phase_tilt_y)
        form.addRow("Pattern centre X", self.center_x)
        form.addRow("Pattern centre Y", self.center_y)
        form.addRow("Duty / line width", self.duty)
        form.addRow("Squircle power", self.squircle_power)
        form.addRow("Spiral arms", self.spiral_arms)
        form.addRow("Spiral width", self.spiral_width)
        form.addRow("Starburst spokes", self.spokes)
        form.addRow("Speckle size", self.speckle_size)
        form.addRow("Speckle seed", self.seed)
        form.addRow("Brightness", self.brightness)
        form.addRow(self.invert)
        form.addRow("Mode", self.mode)
        form.addRow("Rotation speed", self.speed)
        form.addRow("Phase sweep", self.phase_speed)
        form.addRow("Period breathing", self.breathing_depth)
        form.addRow("Breathing frequency", self.breathing_hz)
        form.addRow("Step target", self.step_target)
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
            self.phase_tilt_x,
            self.phase_tilt_y,
            self.center_x,
            self.center_y,
            self.duty,
            self.squircle_power,
            self.spiral_arms,
            self.spiral_width,
            self.spokes,
            self.speckle_size,
            self.seed,
            self.brightness,
            self.invert,
            self.mode,
            self.speed,
            self.phase_speed,
            self.breathing_depth,
            self.breathing_hz,
            self.step_target,
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
            dpr = max(0.1, float(screen.devicePixelRatio()))
            physical_width = round(geometry.width() * dpr)
            physical_height = round(geometry.height() * dpr)
            orientation = "portrait" if physical_height > physical_width else "landscape"
            if abs(dpr - 1.0) < 0.01:
                description = (
                    f"{physical_width}x{physical_height} px, {orientation}"
                )
            else:
                description = (
                    f"{physical_width}x{physical_height} px, {orientation}; "
                    f"Qt {geometry.width()}x{geometry.height()} @ {dpr:.2f}x"
                )
            self.screen.addItem(
                f"{index + 1}: {screen.name()} ({description})",
                index,
            )
        preferred = 1 if len(screens) > 1 else 0
        index = self.screen.findData(current if current is not None else preferred)
        self.screen.setCurrentIndex(max(0, index))

    def settings(self) -> PatternSettings:
        family = str(self.family.currentData())
        mode = str(self.mode.currentData())
        if family in {"speckle", "solid"}:
            mode = "static"
        return PatternSettings(
            family=family,
            period_px=self.period.value(),
            angle_deg=self.angle.value(),
            phase_deg=self.phase.value(),
            brightness=float(self.brightness.value()),
            duty_percent=self.duty.value(),
            squircle_power=self.squircle_power.value(),
            spiral_arms=self.spiral_arms.value(),
            spiral_width_percent=self.spiral_width.value(),
            spokes=self.spokes.value(),
            speckle_size_px=self.speckle_size.value(),
            seed=self.seed.value(),
            invert=self.invert.isChecked(),
            waveform=str(self.waveform.currentData()),
            mode=mode,
            speed_deg_s=self.speed.value(),
            phase_speed_cycles_s=self.phase_speed.value(),
            breathing_depth_percent=self.breathing_depth.value(),
            breathing_hz=self.breathing_hz.value(),
            step_deg=self.step_size.value(),
            step_target=str(self.step_target.currentData()),
            center_x_percent=self.center_x.value(),
            center_y_percent=self.center_y.value(),
            phase_tilt_x_deg=self.phase_tilt_x.value(),
            phase_tilt_y_deg=self.phase_tilt_y.value(),
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
        family = str(self.family.currentData())
        hints = {
            "stripes": "Directional baseline. Fine stripes + rotation were visually strong in the first car test.",
            "checker": "Two-direction grid. Useful when local line mixing or corner distortion is of interest.",
            "rings": "Radial fringes. Phase stepping expands/contracts the rings without rotating them.",
            "composite": "Combined X/Y carrier for a single-frame two-direction baseline.",
            "nested_square": "Concentric square fringes from the standalone V02 generator.",
            "squircle": "Superellipse between rings and squares; power 2 is circular.",
            "spiral": "Archimedean-style line spiral. One fine arm is the current recommended starting point.",
            "counter_spiral": "Opposite-handed spiral pair; more complex, intended for comparison rather than default use.",
            "starburst": "Alternating radial sectors; tests angular disturbance sensitivity.",
            "speckle": "Repeatable pseudo-random texture for future optical-flow/DIC-oriented experiments.",
            "solid": "Uniform field for illumination/reference checks; no dynamic structure.",
        }
        self.pattern_hint.setText(hints.get(family, ""))
        mode = str(self.mode.currentData())
        animatable = family not in {"speckle", "solid"}
        continuous = animatable and mode == "continuous"
        stepped = animatable and mode == "step"

        visibility = {
            self.period: family not in {"starburst", "speckle", "solid"},
            self.waveform: family in {"stripes", "rings", "composite"},
            self.angle: family
            in {
                "stripes",
                "checker",
                "composite",
                "nested_square",
                "squircle",
                "spiral",
                "counter_spiral",
                "starburst",
            },
            self.phase: family not in {"speckle", "solid"},
            self.phase_tilt_x: family not in {"speckle", "solid"},
            self.phase_tilt_y: family not in {"speckle", "solid"},
            self.center_x: family
            in {"rings", "nested_square", "squircle", "spiral", "counter_spiral", "starburst"},
            self.center_y: family
            in {"rings", "nested_square", "squircle", "spiral", "counter_spiral", "starburst"},
            self.duty: family
            in {"stripes", "rings", "nested_square", "squircle"},
            self.squircle_power: family == "squircle",
            self.spiral_arms: family in {"spiral", "counter_spiral"},
            self.spiral_width: family in {"spiral", "counter_spiral"},
            self.spokes: family == "starburst",
            self.speckle_size: family == "speckle",
            self.seed: family == "speckle",
            self.mode: animatable,
            self.speed: continuous,
            self.phase_speed: continuous,
            self.breathing_depth: continuous,
            self.breathing_hz: continuous and self.breathing_depth.value() > 0,
            self.animation_fps: continuous,
            self.step_target: stepped and family != "rings",
            self.step_size: stepped,
        }

        for editor, visible in visibility.items():
            editor.setVisible(visible)
        self.step_button.setVisible(stepped)

        layout = self.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            for editor, visible in visibility.items():
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
