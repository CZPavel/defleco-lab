"""Bounded binary-stripe artifact diagnostics; no cameras or driver changes.

Example: python tools/pattern_artifact_diagnostics.py --width 2160 --height 3840
Add --screen 1 to sample six pre-swap onscreen frames during a six-second run.
Readbacks verify rendered pixels, not physical display scanout or absence of tearing.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGL, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defleco_lab.gui.pattern_motion import frame_motion_samples
from defleco_lab.gui.pattern_output import PatternOutputWindow, PatternSettings
from defleco_lab.gui.pattern_shader import PatternShader


def native_read_pixels():
    # PySide's void* binding rejects 64-bit integer buffer addresses on Windows.
    # glReadPixels is an exported GL 1.1 entry point; no extension loader is needed.
    if sys.platform == "win32":
        library = ctypes.WinDLL("opengl32.dll")
    else:
        name = ctypes.util.find_library("OpenGL" if sys.platform == "darwin" else "GL")
        if not name:
            raise RuntimeError("Cannot locate native OpenGL readback library")
        library = ctypes.CDLL(name)
    function = library.glReadPixels
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    function.restype = None
    return function


_READ_PIXELS = None


def settings_at(seconds: float, period: float) -> PatternSettings:
    return PatternSettings(
        family="stripes", waveform="binary", mode="continuous",
        period_px=period * (1.0 + 0.2 * math.sin(math.pi * seconds)),
        angle_deg=(20.0 * seconds) % 360.0,
        speed_deg_s=20, breathing_depth_percent=20, breathing_hz=0.5,
    )


def raw_reference(width: int, height: int, settings: PatternSettings) -> np.ndarray:
    """Independent float64 evaluation of affine stripe coverage, top-origin."""
    x = np.arange(width, dtype=np.float64)[None, :] - (width - 1) / 2
    y = np.arange(height, dtype=np.float64)[:, None] - (height - 1) / 2
    x -= width * settings.center_x_percent / 100
    y -= height * settings.center_y_percent / 100
    angle = math.radians(settings.angle_deg)
    period = max(2.0, settings.period_px)
    gx = math.cos(angle) / period + settings.phase_tilt_x_deg / 360 / max(1, width - 1)
    gy = math.sin(angle) / period + settings.phase_tilt_y_deg / 360 / max(1, height - 1)
    cycles = gx * x + gy * y + (settings.phase_deg % 360) / 360
    duty = min(0.99, max(0.01, settings.duty_percent / 100))
    distance = np.abs(np.remainder(cycles - duty / 2 + 0.5, 1) - 0.5)
    coverage = np.clip((duty / 2 - distance) / max(abs(gx) + abs(gy), 1e-6) + 0.5, 0, 1)
    if settings.invert:
        coverage = 1 - coverage
    peak = round(min(100, max(0, settings.brightness)) * 255 / 100)
    return coverage * peak


def reference(width: int, height: int, settings: PatternSettings,
              samples: tuple[PatternSettings, ...] = ()) -> np.ndarray:
    if not samples:
        return np.rint(raw_reference(width, height, settings)).astype(np.uint8)
    total = np.zeros((height, width), dtype=np.float64)
    for sample in samples:
        total += raw_reference(width, height, sample)
    return np.rint(total / len(samples)).astype(np.uint8)


def read_pixels(context: QtGui.QOpenGLContext, width: int, height: int) -> np.ndarray:
    global _READ_PIXELS
    if _READ_PIXELS is None:
        _READ_PIXELS = native_read_pixels()
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    _READ_PIXELS(0, 0, width, height, 0x1908, 0x1401,
                 pixels.ctypes.data)  # RGBA / UNSIGNED_BYTE
    return pixels[::-1, :, 0].copy()


def errors(context: QtGui.QOpenGLContext) -> list[int]:
    result = []
    for _ in range(16):
        error = context.functions().glGetError()
        if not error:
            break
        result.append(error)
    return result


def identity(context: QtGui.QOpenGLContext) -> dict:
    gl, fmt = context.functions(), context.format()
    return {
        "vendor": gl.glGetString(0x1F00), "renderer": gl.glGetString(0x1F01),
        "version": gl.glGetString(0x1F02),
        "context": {"major": fmt.majorVersion(), "minor": fmt.minorVersion(),
                    "profile": str(fmt.profile()), "swap_interval": fmt.swapInterval(),
                    "samples": fmt.samples(), "swap_behavior": str(fmt.swapBehavior())},
    }


class Comparison:
    def __init__(self, output: Path | None) -> None:
        self.output, self.saved = output, 0

    def check(self, actual: np.ndarray, settings: PatternSettings, label: str,
              samples: tuple[PatternSettings, ...] = ()) -> dict:
        height, width = actual.shape
        expected = reference(width, height, settings, samples)
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        bad = delta > 3  # Allow tiny float32 derivative/coverage rounding differences.
        ys, xs = np.nonzero(bad)
        fraction = float(bad.mean())
        result = {
            "label": label, "width": width, "height": height,
            "angle_deg": settings.angle_deg, "period_px": settings.period_px,
            "max_error": int(delta.max()), "mean_error": float(delta.mean()),
            "fraction_error_over_3": fraction,
            "bad_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if xs.size else None,
            "worst_64px_tile_bad_fraction": max(
                float(bad[y:y + 64, x:x + 64].mean())
                for y in range(0, height, 64) for x in range(0, width, 64)
            ),
            # A small corrupt tile can disappear in a global percentage on 4K.
            # Every sampled pixel must satisfy the explicit rounding tolerance.
            "passed": int(delta.max()) <= 3,
        }
        if not result["passed"] and self.output is not None and self.saved < 3:
            self.output.mkdir(parents=True, exist_ok=True)
            path = self.output / f"failure_{self.saved + 1}_{label}.png"
            image = QtGui.QImage(actual.data, width, height, actual.strides[0],
                                 QtGui.QImage.Format.Format_Grayscale8)
            if not image.save(str(path)):
                raise RuntimeError(f"Cannot save {path}")
            result["capture"] = str(path)
            self.saved += 1
        return result


def offscreen(args: argparse.Namespace, comparison: Comparison) -> dict:
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QtGui.QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        raise RuntimeError("Cannot create OpenGL context")
    surface = QtGui.QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        raise RuntimeError("Cannot activate offscreen context")
    report = identity(context)
    shader = PatternShader()
    framebuffer = None
    try:
        shader.initialize(context)
        framebuffer = QtOpenGL.QOpenGLFramebufferObject(args.width, args.height)
        if not framebuffer.isValid() or not framebuffer.bind():
            raise RuntimeError("Cannot bind diagnostic framebuffer")
        frames = []
        # Include both sides of 90/180/270/360-degree and breathing phase wraps.
        times = np.unique(np.concatenate((np.linspace(0, 18, args.frames),
                         [4.499, 4.501, 8.999, 9.001, 13.499, 13.501, 17.999, 18.001])))
        for index, seconds in enumerate(times):
            settings = settings_at(float(seconds), args.period)
            samples = frame_motion_samples(
                replace(settings, period_px=args.period, motion_smoothing=args.motion_smoothing),
                settings.angle_deg, 0, math.pi * float(seconds), 60.0,
            )
            shader.draw(settings, args.width, args.height, motion_samples=samples)
            actual = read_pixels(context, args.width, args.height)
            row = comparison.check(actual, settings, f"offscreen_{index:02d}", samples)
            row["elapsed_s"] = float(seconds)
            shader.draw(settings, args.width, args.height, motion_samples=samples)
            repeat = read_pixels(context, args.width, args.height)
            row["repeat_mismatched_pixels"] = int(np.count_nonzero(actual != repeat))
            row["gl_errors"] = errors(context)
            row["passed"] &= not row["repeat_mismatched_pixels"] and not row["gl_errors"]
            frames.append(row)
        report["frames"] = frames
        report["passed"] = all(row["passed"] for row in frames)
    finally:
        if framebuffer is not None:
            framebuffer.release()
            del framebuffer
        shader.close()
        context.doneCurrent()
    return report


def onscreen(app: QtWidgets.QApplication, screen: int, period: float,
             comparison: Comparison, motion_smoothing: bool = False) -> dict:
    if not 0 <= screen < len(app.screens()):
        raise ValueError("Requested screen is not connected")

    class CaptureWindow(PatternOutputWindow):
        def __init__(self) -> None:
            super().__init__()
            self.start = perf_counter()
            self.samples: list[dict] = []
            self.captures: list[tuple] = []
            self.swaps: list[tuple[float, bool]] = []
            self.sampled = False
            self.failures: list[str] = []
            self.frameSwapped.connect(self.record_swap)
            self.renderFailed.connect(self.failures.append)
            self.gpu: dict = {}

        def record_swap(self) -> None:
            self.swaps.append((perf_counter(), self.sampled))
            self.sampled = False

        def paintGL(self) -> None:
            super().paintGL()
            elapsed = perf_counter() - self.start
            if len(self.captures) >= 6 or elapsed < 0.3 + len(self.captures) * 0.85:
                return
            if self._render_error:
                return
            context = self.context()
            if not self.gpu:
                self.gpu = identity(context)
            width = round(self.width() * self.devicePixelRatio())
            height = round(self.height() * self.devicePixelRatio())
            # Read inside paintGL, before Qt swaps; never grabFramebuffer recursively.
            actual = read_pixels(context, width, height)
            self.captures.append((actual, self.effective_settings,
                                  self._last_motion_samples, errors(context), elapsed))
            self.sampled = True

    window = CaptureWindow()
    window.set_settings(replace(settings_at(0, period), period_px=period,
                                motion_smoothing=motion_smoothing))
    window.show_on_screen(screen)
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(6000, loop.quit)
    try:
        loop.exec()
        window.hide_output()
        for index, (actual, settings, samples, gl_errors, elapsed) in enumerate(window.captures):
            row = comparison.check(actual, settings, f"onscreen_{index:02d}", samples)
            row.update(gl_errors=gl_errors, wall_elapsed_s=elapsed,
                       temporal_samples=len(samples) or 1)
            row["passed"] &= not gl_errors
            window.samples.append(row)
        clean_intervals = [(b[0] - a[0]) * 1000
                           for a, b in zip(window.swaps, window.swaps[1:])
                           if not a[1] and not b[1]]
        return {
            "screen": app.screens()[screen].name(), "gpu": window.gpu,
            "display_diagnostics": window.display_diagnostics(),
            "frames": window.samples, "render_errors": window.failures,
            "swaps": len(window.swaps),
            "unsampled_interval_count": len(clean_intervals),
            "unsampled_interval_median_ms": float(np.median(clean_intervals))
            if clean_intervals else None,
            "unsampled_interval_p95_ms": float(np.percentile(clean_intervals, 95))
            if clean_intervals else None,
            "passed": len(window.samples) == 6 and not window.failures
            and all(row["passed"] for row in window.samples),
        }
    finally:
        window.hide_output()
        window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--period", type=float, default=50)
    parser.add_argument("--frames", type=int, default=24,
                        help="Base sweep samples (2..48), plus eight wrap probes")
    parser.add_argument("--output", type=Path, help="Save at most three failing frames here")
    parser.add_argument("--screen", type=int, help="Optional zero-based physical display")
    parser.add_argument("--motion-smoothing", action="store_true",
                        help="Validate temporal integration as well as sharp snapshots")
    args = parser.parse_args()
    if not (2 <= args.width <= 8192 and 2 <= args.height <= 8192
            and 2 <= args.frames <= 48 and math.isfinite(args.period) and args.period >= 2):
        parser.error("Dimensions must be 2..8192; frames 2..48; period finite and >=2")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    comparison = Comparison(args.output)
    report = {"settings": {"family": "stripes", "waveform": "binary",
              "period_px": args.period, "breathing_depth_percent": 20,
              "breathing_hz": 0.5, "speed_deg_s": 20,
              "motion_smoothing": args.motion_smoothing},
              "offscreen": offscreen(args, comparison)}
    if args.screen is not None:
        report["onscreen"] = onscreen(app, args.screen, args.period, comparison, args.motion_smoothing)
    report["passed"] = report["offscreen"]["passed"] and (
        "onscreen" not in report or report["onscreen"]["passed"])
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
