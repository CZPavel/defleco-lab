"""Bounded A/B test in the real application PatternOutputWindow, never an offscreen context.

Launch with the SAME pythonw.exe as the production launcher when testing Windows
per-executable GPU preferences. No resolution, registry or driver writes occur.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defleco_lab.app import configure_application, configure_opencv_runtime
from defleco_lab.gui import pattern_output
from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.pattern_output import PatternSettings, render_pattern, render_pattern_rgb
from defleco_lab.processing import load_builtin_methods
from tools.pattern_artifact_diagnostics import read_pixels_rgb


def rgb_suite(app: QtWidgets.QApplication, args: argparse.Namespace) -> int:
    """Bounded palette changes through the live panel, with production AA enabled."""
    class CaptureOutput(pattern_output.PatternOutputWindow):
        def __init__(self):
            super().__init__()
            self.armed = False
            self.capture = None
            self.swaps = 0
            self.frameSwapped.connect(self.record_swap)

        def record_swap(self):
            self.swaps += 1

        def paintGL(self):
            super().paintGL()
            if self.armed and not self._render_error:
                width = round(self.width() * self.devicePixelRatio())
                height = round(self.height() * self.devicePixelRatio())
                self.capture = (read_pixels_rgb(self.context(), width, height),
                                self.effective_settings, self.context(), self.swaps)
                self.armed = False

    original = pattern_output.PatternOutputWindow
    pattern_output.PatternOutputWindow = CaptureOutput
    window = MainWindow()
    pattern_output.PatternOutputWindow = original
    window.show()
    panel, rows, failures = window.pattern_output, [], []
    output = panel.output
    original_colors = panel.dark_color.rgb(), panel.light_color.rgb()
    output.renderFailed.connect(failures.append)
    panel.screen.setCurrentIndex(panel.screen.findData(args.screen))
    panel.mode.setCurrentIndex(panel.mode.findData("static"))
    panel.period.setValue(50)
    panel.motion_smoothing.setChecked(False)
    cases = [
        ("red_black_stripes", "stripes", "binary", "#000000", "#ff0000", 0, 100, False),
        ("blue_yellow_stripes", "stripes", "sinusoidal", "#0000ff", "#ffff00", 0, 100, False),
        ("green_magenta_checker", "checker", "binary", "#00ff00", "#ff00ff", 0, 100, False),
        ("colored_rings_0", "rings", "sinusoidal", "#182a91", "#eeb437", 0, 100, False),
        ("colored_rings_90", "rings", "sinusoidal", "#182a91", "#eeb437", 90, 100, False),
        ("colored_spiral_0", "spiral", "binary", "#184a91", "#ee6437", 0, 100, False),
        ("colored_spiral_90", "spiral", "binary", "#184a91", "#ee6437", 90, 100, False),
        ("speckle", "speckle", "binary", "#1450b0", "#ee9030", 0, 100, False),
        ("speckle_dim_invert", "speckle", "binary", "#1450b0", "#ee9030", 0, 43, True),
        ("solid", "solid", "binary", "#143080", "#e05030", 0, 73, False),
    ]
    owning_context = None
    previous = {}
    try:
        for index, (label, family, waveform, dark, light, phase, brightness, invert) in enumerate(cases):
            panel.family.setCurrentIndex(panel.family.findData(family))
            panel.waveform.setCurrentIndex(panel.waveform.findData(waveform))
            panel._set_colors(QtGui.QColor(dark).getRgb()[:3], QtGui.QColor(light).getRgb()[:3])
            panel.phase.setValue(phase)
            panel.brightness.setValue(brightness)
            panel.invert.setChecked(invert)
            if index == 0:
                output.show_on_screen(args.screen)
            output.capture = None
            swaps_before = output.swaps
            # Present the edited state first; capture a subsequent paint of it.
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(250, loop.quit)
            loop.exec()
            presented = output.swaps > swaps_before
            output.armed = True
            output.requestUpdate()
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(200, loop.quit)
            loop.exec()
            if output.capture is None:
                raise RuntimeError(f"No framebuffer capture for {label}: {failures}")
            actual, settings, context, swaps = output.capture
            if owning_context is None:
                owning_context = context
            expected = render_pattern_rgb(actual.shape[1], actual.shape[0], settings)
            if waveform == "binary" and family not in {"speckle", "solid"}:
                mono = render_pattern(actual.shape[1], actual.shape[0], settings)
                kernel = np.ones((3, 3), np.uint8)
                valid = cv2.erode(mono, kernel) == cv2.dilate(mono, kernel)
                # The 3x3 interior test needs a complete pixel neighbourhood.
                # OpenCV's morphology ignores missing neighbours at the image
                # border, while GL coverage still includes edges outside it.
                valid[[0, -1], :] = False
                valid[:, [0, -1]] = False
            else:
                valid = np.ones(actual.shape[:2], bool)
            delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            maximum = int(delta[valid].max()) if valid.any() else 255
            changed = None
            if family in previous:
                changed = bool(np.any(previous[family] != actual))
            previous[family] = actual.copy()
            row = {"label": label, "settings": settings.as_dict(),
                   "valid_fraction": float(valid.mean()), "max_channel_error": maximum,
                   "same_context": context is owning_context, "presented_before_capture": presented,
                   "frame_swaps": swaps, "changed_from_previous_family_frame": changed,
                   "passed": bool(maximum <= 3 and valid.mean() > 0.2 and context is owning_context
                                  and presented and changed is not False)}
            if index in (0, 1):
                path = args.output / f"rgb_{label}.png"
                image = QtGui.QImage(actual.data, actual.shape[1], actual.shape[0], actual.strides[0],
                                     QtGui.QImage.Format.Format_RGB888)
                if not image.save(str(path)):
                    raise RuntimeError(f"Cannot save {path}")
                row["file"] = path.name
            rows.append(row)
        diagnostics = output.display_diagnostics()
        renderer = diagnostics["graphics"].get("renderer", "")
        expected_renderer = (args.expect_renderer is None or
                             args.expect_renderer.lower() in renderer.lower())
        report = {"test_label": args.label, "graphics": diagnostics["graphics"],
                  "display_diagnostics": diagnostics, "frames": rows, "render_errors": failures,
                  "fullscreen": (output.visibility() == QtGui.QWindow.Visibility.FullScreen), "physical_observation": "UNCERTAIN",
                  "readback_note": "Pre-swap framebuffer capture after an earlier presentation; not scanout.",
                  "software_test_passed": bool(expected_renderer and not failures
                                               and (output.visibility() == QtGui.QWindow.Visibility.FullScreen)
                                               and all(row["passed"] for row in rows))}
        (args.output / "rgb-suite.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if report["software_test_passed"] else 2
    finally:
        output.hide_output()
        panel._set_colors(*original_colors)
        window.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--screen", type=int, default=1)
    parser.add_argument("--rgb-suite", action="store_true")
    parser.add_argument("--preset", choices=("primary", "slow"), default="primary")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--expect-renderer", choices=("Intel", "NVIDIA"))
    parser.add_argument("--dark-color", default="#000000", help="Dark endpoint as #RRGGBB")
    parser.add_argument("--light-color", default="#ffffff", help="Light endpoint as #RRGGBB")
    args = parser.parse_args()
    colors = [QtGui.QColor(value) for value in (args.dark_color, args.light_color)]
    if not all(color.isValid() for color in colors):
        parser.error("Endpoint colors must be valid Qt colors, for example #RRGGBB")
    if not 5 <= args.seconds <= 60:
        parser.error("seconds must be 5..60")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    configure_application(app)
    configure_opencv_runtime()
    load_builtin_methods()
    if not 0 <= args.screen < len(app.screens()):
        parser.error("Screen index is not connected")
    screen = app.screens()[args.screen]
    geometry = screen.geometry()
    size = (round(geometry.width() * screen.devicePixelRatio()),
            round(geometry.height() * screen.devicePixelRatio()))
    if size != (1080, 1920) or abs(screen.refreshRate() - 60) > 0.5:
        parser.error(f"Target must already be 1080x1920@60; found {size}@{screen.refreshRate()}")

    if args.rgb_suite:
        return rgb_suite(app, args)

    class ObservedOutput(pattern_output.PatternOutputWindow):
        def __init__(self):
            super().__init__()
            self.captures = []
            self.started = perf_counter()
            self.swaps = []
            self.sampled = False
            self.frameSwapped.connect(self._record_swap)

        def _record_swap(self):
            self.swaps.append((perf_counter(), self.sampled))
            self.sampled = False

        def paintGL(self):
            super().paintGL()
            elapsed = perf_counter() - self.started
            if self._render_error or len(self.captures) >= 2:
                return
            if elapsed < 1.0 + len(self.captures) * (args.seconds / 2):
                return
            width = round(self.width() * self.devicePixelRatio())
            height = round(self.height() * self.devicePixelRatio())
            pixels = read_pixels_rgb(self.context(), width, height)
            self.captures.append((pixels, self.effective_settings.as_dict()))
            self.sampled = True

    pattern_output.PatternOutputWindow = ObservedOutput
    window = MainWindow()
    window.setWindowTitle(f"Defleco LAB — GPU A/B {args.label}")
    window.show()
    window.pattern_dock.raise_()
    panel = window.pattern_output
    panel.waveform.setCurrentIndex(panel.waveform.findData("sinusoidal"))
    panel.mode.setCurrentIndex(panel.mode.findData("continuous"))
    panel.period.setValue(50)
    panel.speed.setValue(20 if args.preset == "primary" else 5)
    panel.breathing_depth.setValue(20 if args.preset == "primary" else 5)
    panel.breathing_hz.setValue(0.5 if args.preset == "primary" else 0.125)
    panel.motion_smoothing.setChecked(False)
    panel.screen.setCurrentIndex(panel.screen.findData(args.screen))
    output = panel.output
    errors = []
    output.renderFailed.connect(errors.append)
    output.set_settings(PatternSettings(
        family="stripes", waveform="sinusoidal", mode="continuous", period_px=50,
        speed_deg_s=panel.speed.value(), breathing_depth_percent=panel.breathing_depth.value(),
        breathing_hz=panel.breathing_hz.value(), motion_smoothing=False,
        dark_color=colors[0].getRgb()[:3], light_color=colors[1].getRgb()[:3],
    ))
    output.show_on_screen(args.screen)

    def save_startup():
        (args.output / "startup.json").write_text(
            json.dumps(output.display_diagnostics(), indent=2), encoding="utf-8"
        )

    QtCore.QTimer.singleShot(600, save_startup)
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(round(args.seconds * 1000), loop.quit)
    loop.exec()
    # Snapshot before hide; hide invalidates presentation/cadence diagnostics.
    report = output.display_diagnostics()
    intervals = [(b[0] - a[0]) * 1000 for a, b in zip(output.swaps[8:], output.swaps[9:])
                 if not a[1] and not b[1]]
    renderer = report["graphics"].get("renderer", "")
    expected = args.expect_renderer is None or args.expect_renderer.lower() in renderer.lower()
    report.update(
        test_label=args.label, preset=args.preset, duration_seconds=args.seconds,
        renderer_expected=args.expect_renderer,
        renderer_confirmed=expected if args.expect_renderer else None,
        physical_observation="UNCERTAIN", result="UNCERTAIN",
        physical_observation_note="Requires user observation of the TV; readback is not scanout.",
        render_errors=errors,
        unsampled_cadence={
            "interval_count": len(intervals),
            "median_ms": float(np.median(intervals)) if intervals else None,
            "p95_ms": float(np.percentile(intervals, 95)) if intervals else None,
            "maximum_ms": float(np.max(intervals)) if intervals else None,
            "fps_from_median": 1000 / float(np.median(intervals)) if intervals else None,
        },
    )
    output.hide_output()
    report["readbacks"] = []
    for index, (pixels, settings) in enumerate(output.captures):
        path = args.output / f"readback_{index + 1}.png"
        image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                             QtGui.QImage.Format.Format_RGB888)
        if not image.save(str(path)):
            raise RuntimeError(f"Cannot save {path}")
        reference = render_pattern_rgb(pixels.shape[1], pixels.shape[0], PatternSettings(**settings))
        delta = np.abs(pixels.astype(np.int16) - reference.astype(np.int16))
        report["readbacks"].append({
            "file": path.name, "settings": settings,
            "max_error": int(delta.max()), "mean_error": float(delta.mean()),
            "pixel_check_passed": bool(delta.max() <= 3),
        })
    report["software_test_passed"] = bool(
        expected and not errors and len(output.captures) == 2
        and all(row["pixel_check_passed"] for row in report["readbacks"])
    )
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    window.close()
    return 0 if report["software_test_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
