"""Integrated display-pattern generator for physical deflectometry experiments."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGL, QtWidgets

from .gpu_identity import (
    display_adapter_for_rect,
    gpu_preferences,
    process_identity,
    render_path_status,
)
from .pattern_motion import frame_motion_samples
from .pattern_shader import PatternShader


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
    motion_smoothing: bool = True
    step_deg: float = 15.0
    step_target: str = "angle"
    center_x_percent: float = 0.0
    center_y_percent: float = 0.0
    phase_tilt_x_deg: float = 0.0
    phase_tilt_y_deg: float = 0.0
    render_scale: float = 1.0
    animation_fps: int = 60
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
    phase_period = 720.0 if settings.family == "starburst" else 360.0
    phase_cycles = (float(settings.phase_deg) % phase_period) / 360.0

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



class PatternOutputWindow(QtOpenGL.QOpenGLWindow):
    """Native GPU surface; schedule animation only after the previous buffer swap."""

    stateChanged = QtCore.Signal(object)
    renderFailed = QtCore.Signal(str)
    graphicsInfoChanged = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        fmt.setSwapInterval(1)
        fmt.setDepthBufferSize(0)
        fmt.setStencilBufferSize(0)
        fmt.setRedBufferSize(8)
        fmt.setGreenBufferSize(8)
        fmt.setBlueBufferSize(8)
        fmt.setAlphaBufferSize(0)
        fmt.setSamples(0)
        fmt.setOption(QtGui.QSurfaceFormat.FormatOption.ResetNotification)
        self.setFormat(fmt)
        self.setTitle("Defleco LAB - Pattern Output")
        self.setFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self._settings = PatternSettings()
        self._effective_angle = self._settings.angle_deg
        self._effective_phase = self._settings.phase_deg
        self._breathing_phase = 0.0
        self._elapsed = QtCore.QElapsedTimer()
        self._shader = PatternShader()
        self._render_error = ""
        self._settings_revision = 0
        self._painted_revision = -1
        self._presented_revision = -1
        self._last_motion_samples: tuple[PatternSettings, ...] = ()
        self._graphics_info: dict = {}
        self._swap_times: deque[float] = deque(maxlen=241)
        self._context_lifecycle: list[dict] = []
        self._panel_native_size: tuple[int, int] | None = None
        self._record_context_event("constructed")
        self.frameSwapped.connect(self._frame_swapped)

    def display_diagnostics(self) -> dict:
        intervals = np.diff(self._swap_times) * 1000.0
        screen = self.screen()
        identity = process_identity()
        geometry = screen.geometry() if screen else QtCore.QRect()
        # Exact rectangle mapping only; a scaled/mixed-DPI ambiguity stays unknown.
        adapter = display_adapter_for_rect(
            geometry.x(), geometry.y(), geometry.width(), geometry.height()
        )
        if screen and abs(screen.devicePixelRatio() - 1.0) > 1e-6:
            adapter = {
                "status": "unknown", "adapter_name": None, "device_name": None,
                "mapping_method": "scaled Qt/WinAPI rectangle mapping not verified",
            }
        width, height = round(self.width() * self.devicePixelRatio()), round(
            self.height() * self.devicePixelRatio()
        )
        panel_width, panel_height = self._panel_native_size or (None, None)
        scale_x = panel_width / width if panel_width and width else None
        scale_y = panel_height / height if panel_height and height else None
        return {
            "process": identity,
            "windows_gpu_preferences": gpu_preferences(
                [identity["executable"], identity["base_executable"], identity["process_image_path"]]
            ),
            "display_adapter": adapter,
            "render_path_status": render_path_status(
                self._graphics_info.get("renderer"), adapter.get("adapter_name")
            ),
            "context_lifecycle": list(self._context_lifecycle),
            "device_pixel_ratio": self.devicePixelRatio(),
            "framebuffer_width": width,
            "framebuffer_height": height,
            "physical_panel_width": panel_width,
            "physical_panel_height": panel_height,
            "panel_size_source": "operator supplied; not auto-detected",
            "estimated_scale_x": scale_x,
            "estimated_scale_y": scale_y,
            "panel_scaling_warning": bool(
                scale_x is not None and scale_y is not None
                and (abs(scale_x - 2.0) > 1e-6 or abs(scale_y - 2.0) > 1e-6)
            ),
            "period_coordinate_basis": "framebuffer pixels before TV scaling",
            "settings": self._settings.as_dict(),
            "last_submitted_settings": self.effective_settings.as_dict(),
            "temporal_samples": len(self._last_motion_samples) or 1,
            "graphics": self._graphics_info,
            "screen": screen.name() if screen else None,
            "physical_size": [round(self.width() * self.devicePixelRatio()),
                              round(self.height() * self.devicePixelRatio())],
            "refresh_hz_reported": screen.refreshRate() if screen else None,
            "swap_intervals_ms": {
                "count": len(intervals),
                "median": float(np.median(intervals)) if len(intervals) else None,
                "p95": float(np.percentile(intervals, 95)) if len(intervals) else None,
                "maximum": float(np.max(intervals)) if len(intervals) else None,
            },
            "render_error": self._render_error,
            "scanout_verified": False,
        }

    def set_panel_native_size(self, width: int, height: int) -> None:
        self._panel_native_size = (int(width), int(height)) if width > 0 and height > 0 else None

    def _record_context_event(self, stage: str) -> None:
        context = self.context()
        self._context_lifecycle.append({
            "stage": stage,
            "window_screen": self.screen().name() if self.screen() else None,
            "context_exists": context is not None,
            "context_valid": context.isValid() if context else False,
            "context_screen": context.screen().name() if context and context.screen() else None,
        })
        self._context_lifecycle = self._context_lifecycle[-24:]

    @property
    def has_presented_frame(self) -> bool:
        return (
            self.isExposed() and not self._render_error
            and self._presented_revision == self._settings_revision
        )

    @property
    def effective_settings(self) -> PatternSettings:
        # This is the last submitted frame, not a fresh clock sample: experiments
        # must not record settings for an image that has never been drawn.
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

    def _advance(self, seconds: float) -> None:
        """Advance by monotonic elapsed time; never integrate nominal FPS."""
        rotation = float(self._settings.speed_deg_s) * seconds
        phase = float(self._settings.phase_speed_cycles_s) * 360.0 * seconds
        phase_period = 720.0 if self._settings.family == "starburst" else 360.0
        if self._settings.family == "rings" and phase == 0.0:
            self._effective_phase = (self._effective_phase + rotation) % 360.0
        else:
            # A one-arm spiral is 360-degree periodic, not 180-degree periodic.
            self._effective_angle = (self._effective_angle + rotation) % 360.0
            self._effective_phase = (self._effective_phase + phase) % phase_period
        self._breathing_phase = (
            self._breathing_phase + 2.0 * math.pi * self._settings.breathing_hz * seconds
        ) % (2.0 * math.pi)

    def _sample_animation(self) -> None:
        if self._settings.mode == "continuous" and self._elapsed.isValid():
            seconds = self._elapsed.nsecsElapsed() / 1_000_000_000.0
            self._elapsed.start()
            self._advance(seconds)

    def set_settings(self, settings: PatternSettings) -> None:
        self._settings_revision += 1
        self._sample_animation()
        reset_phase = (
            settings.family != self._settings.family
            or settings.mode != self._settings.mode
            or settings.angle_deg != self._settings.angle_deg
            or settings.phase_deg != self._settings.phase_deg
        )
        self._settings = settings
        if reset_phase:
            self._effective_angle = settings.angle_deg
            self._effective_phase = settings.phase_deg
            self._breathing_phase = 0.0
        self._elapsed.invalidate()
        if settings.mode == "continuous" and self.isExposed():
            self._elapsed.start()
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def show_on_screen(self, screen_index: int) -> None:
        self._record_context_event("before_target_screen")
        screens = QtGui.QGuiApplication.screens()
        if screens:
            screen = screens[max(0, min(int(screen_index), len(screens) - 1))]
            self.setScreen(screen)
            self.setGeometry(screen.geometry())
        self._record_context_event("target_screen_assigned_before_expose")
        self.showFullScreen()
        QtCore.QTimer.singleShot(1500, self._check_context)
        self._redraw()

    def _check_context(self) -> None:
        if self.isVisible() and not self.isValid() and not self._render_error:
            self._render_error = "OpenGL 3.3 output could not be initialized."
            self.renderFailed.emit(self._render_error)

    def hide_output(self) -> None:
        self._sample_animation()
        self._elapsed.invalidate()
        self._presented_revision = -1
        self.hide()

    def step(self) -> None:
        self._settings_revision += 1
        delta = float(self._settings.step_deg)
        if self._settings.step_target == "phase" or self._settings.family == "rings":
            period = 720.0 if self._settings.family == "starburst" else 360.0
            self._effective_phase = (self._effective_phase + delta) % period
        else:
            self._effective_angle = (self._effective_angle + delta) % 360.0
        self._redraw()
        self.stateChanged.emit(self.effective_settings)

    def exposeEvent(self, event: QtGui.QExposeEvent) -> None:
        self._presented_revision = -1
        self._swap_times.clear()
        self._elapsed.invalidate()
        if self.isExposed() and self._settings.mode == "continuous":
            self._elapsed.start()
        super().exposeEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.hide_output()
            event.accept()
            return
        super().keyPressEvent(event)

    def initializeGL(self) -> None:
        self._record_context_event("initializeGL")
        try:
            self._shader.initialize(self.context())
            self._render_error = ""
            gl = self.context().functions()
            fmt = self.context().format()
            self._graphics_info = {
                "vendor": gl.glGetString(0x1F00),
                "renderer": gl.glGetString(0x1F01),
                "version": gl.glGetString(0x1F02),
                "qt_version": QtCore.qVersion(),
                "requested_swap_interval": self.format().swapInterval(),
                "context_swap_interval": fmt.swapInterval(),
                "swap_behavior": str(fmt.swapBehavior()),
                "vsync_note": "Requested/read back; physical scanout is not verified.",
            }
            self.graphicsInfoChanged.emit(self._graphics_info)
            self.context().aboutToBeDestroyed.connect(self._release_gl)
        except RuntimeError as exc:
            self._render_error = str(exc)
            self.renderFailed.emit(self._render_error)

    def _release_gl(self) -> None:
        self.makeCurrent()
        self._shader.close()
        self.doneCurrent()

    def paintGL(self) -> None:
        if not self.context() or not self.context().isValid():
            self._render_error = "Graphics context was lost; reopen pattern output."
            self.renderFailed.emit(self._render_error)
            return
        if self._render_error:
            functions = self.context().functions()
            functions.glClearColor(0.0, 0.0, 0.0, 1.0)
            functions.glClear(0x00004000)
            return
        if self._settings.mode == "continuous":
            if not self._elapsed.isValid():
                self._elapsed.start()
            self._sample_animation()
        # QWindow is in logical pixels; gl_FragCoord and the period are physical
        # display pixels. No intermediate low-resolution image or CPU scaling.
        dpr = self.devicePixelRatio()
        width, height = round(self.width() * dpr), round(self.height() * dpr)
        screen = self.screen()
        self._last_motion_samples = frame_motion_samples(
            self._settings, self._effective_angle, self._effective_phase,
            self._breathing_phase, screen.refreshRate() if screen else 60.0,
        )
        try:
            self._shader.draw(self.effective_settings, width, height,
                              motion_samples=self._last_motion_samples)
        except RuntimeError as exc:
            self._render_error = str(exc)
            self.renderFailed.emit(self._render_error)
            return
        self._painted_revision = self._settings_revision
        self.stateChanged.emit(self.effective_settings)

    def _frame_swapped(self) -> None:
        if self.isExposed() and not self._render_error:
            self._swap_times.append(perf_counter())
            self._presented_revision = self._painted_revision
        if self.isExposed() and self._settings.mode == "continuous" and not self._render_error:
            self.update()

    def _redraw(self) -> None:
        # Parameter edits while hidden must not build full-resolution images.
        if self.isExposed():
            self.update()


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
        self.angle.setRange(0.0, 359.9)
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
        self.mode.addItem("Continuous animation", "continuous")
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

        self.motion_smoothing = QtWidgets.QCheckBox("Smooth movement over one display frame")
        self.motion_smoothing.setChecked(True)
        self.motion_smoothing.setToolTip(
            "Reduces strobing of fine moving patterns by averaging 4 temporal samples. "
            "Moving edges become softer. Static and stepped patterns stay sharp. "
            "Disable for instantaneous sharp-frame comparisons."
        )

        self.panel_width = QtWidgets.QSpinBox()
        self.panel_height = QtWidgets.QSpinBox()
        for field, default in ((self.panel_width, 2160), (self.panel_height, 3840)):
            field.setRange(0, 16384)
            field.setValue(default)
            field.setSuffix(" px")
            field.setSpecialValueText("Unknown")
            field.setToolTip(
                "Operator-supplied native panel size for scaling diagnostics only. "
                "Does not change Windows resolution or pattern generation."
            )

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
        self.diagnostics_button = QtWidgets.QPushButton("Save display diagnostics")
        self.diagnostics_button.clicked.connect(self._save_diagnostics)

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
        form.addRow(self.motion_smoothing)
        form.addRow("Step target", self.step_target)
        form.addRow("Step", self.step_size)
        self.render_status = QtWidgets.QLabel(
            "Native display resolution; animation follows display refresh."
        )
        self.render_status.setWordWrap(True)
        form.addRow(self.render_status)
        self.output.renderFailed.connect(self._render_failed)
        self.output.graphicsInfoChanged.connect(self._graphics_changed)
        form.addRow("Output display", self.screen)
        form.addRow("Native panel width (assumed)", self.panel_width)
        form.addRow("Native panel height (assumed)", self.panel_height)
        form.addRow(buttons)
        form.addRow(self.diagnostics_button)

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
            self.motion_smoothing,
            self.step_target,
            self.step_size,
        ):
            _connect(widget, self._settings_changed)
        self.show_button.clicked.connect(self.show_output)
        self.hide_button.clicked.connect(self.output.hide_output)
        self.step_button.clicked.connect(self.output.step)
        self.panel_width.valueChanged.connect(self._panel_size_changed)
        self.panel_height.valueChanged.connect(self._panel_size_changed)
        self._panel_size_changed()
        self._update_context()
        self._settings_changed()

    def _render_failed(self, message: str) -> None:
        self.render_status.setText("Pattern output unavailable: " + message)
        self.output.hide_output()

    def _graphics_changed(self, info: dict) -> None:
        diagnostics = self.output.display_diagnostics()
        self.render_status.setText(
            f"{info.get('renderer', 'GPU')}; {diagnostics['render_path_status']}. VSync requested "
            f"(context interval {info.get('context_swap_interval')}). "
            "Physical display synchronization is not verified."
        )

    def _panel_size_changed(self, *_args: object) -> None:
        self.output.set_panel_native_size(self.panel_width.value(), self.panel_height.value())

    def _save_diagnostics(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save display diagnostics", "pattern-display-diagnostics.json", "JSON (*.json)"
        )
        if filename:
            try:
                Path(filename).write_text(
                    json.dumps(self.output.display_diagnostics(), indent=2), encoding="utf-8"
                )
            except OSError as exc:
                self.render_status.setText(f"Could not save diagnostics: {exc}")

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
            motion_smoothing=self.motion_smoothing.isChecked(),
            step_deg=self.step_size.value(),
            step_target=str(self.step_target.currentData()),
            center_x_percent=self.center_x.value(),
            center_y_percent=self.center_y.value(),
            phase_tilt_x_deg=self.phase_tilt_x.value(),
            phase_tilt_y_deg=self.phase_tilt_y.value(),
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
            self.motion_smoothing: continuous,
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
