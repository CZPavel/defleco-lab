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

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defleco_lab.app import configure_application, configure_opencv_runtime
from defleco_lab.gui import pattern_output
from defleco_lab.gui.main_window import MainWindow
from defleco_lab.gui.pattern_output import PatternSettings, render_pattern
from defleco_lab.processing import load_builtin_methods
from tools.pattern_artifact_diagnostics import read_pixels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--screen", type=int, default=1)
    parser.add_argument("--preset", choices=("primary", "slow"), default="primary")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--expect-renderer", choices=("Intel", "NVIDIA"))
    args = parser.parse_args()
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
            pixels = read_pixels(self.context(), width, height)
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
                             QtGui.QImage.Format.Format_Grayscale8)
        if not image.save(str(path)):
            raise RuntimeError(f"Cannot save {path}")
        reference = render_pattern(pixels.shape[1], pixels.shape[0], PatternSettings(**settings))
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
