"""Explicit, bounded OpenGL acceptance; no camera acquisition or device writes.

Run normally for GPU/CPU pixel comparisons and a native 4K render benchmark.
Pass --screen N to additionally present a short animation on zero-based screen N.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6 import QtCore, QtGui, QtOpenGL, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defleco_lab.gui.pattern_output import (
    PatternOutputWindow,
    PatternSettings,
    render_pattern,
    render_pattern_rgb,
)
from defleco_lab.gui.pattern_shader import PatternShader
from tools.pattern_artifact_diagnostics import errors, identity, raw_reference, read_pixels_rgb


def rgb_comparisons(shader: PatternShader, context: QtGui.QOpenGLContext) -> dict:
    """Exercise RGB framebuffer channels, including cached speckle appearance edits."""
    width, height = 241, 173
    comparisons, temporal = [], []
    families = ("stripes", "checker", "rings", "composite", "nested_square", "squircle",
                "spiral", "counter_spiral", "starburst", "speckle", "solid")
    for family in families:
        for waveform, invert, brightness in (("binary", False, 100),
                                              ("sinusoidal", True, 73),
                                              ("binary", True, 0)):
            settings = PatternSettings(
                family=family, waveform=waveform, invert=invert, brightness=brightness,
                dark_color=(18, 79, 143), light_color=(231, 158, 32),
                angle_deg=213.7, phase_deg=71.3, period_px=27.3,
                phase_tilt_x_deg=34, phase_tilt_y_deg=-67,
                center_x_percent=13, center_y_percent=-7, spokes=17, spiral_arms=3,
            )
            shader.draw(settings, width, height, antialias=False)
            actual = read_pixels_rgb(context, width, height)
            expected = render_pattern_rgb(width, height, settings)
            delta = np.abs(actual.astype(int) - expected.astype(int))
            within_one = float(np.mean(np.all(delta <= 1, axis=2)))
            row = {"family": family, "waveform": waveform, "invert": invert,
                       "brightness": brightness, "within_one": within_one, "max_error": int(delta.max())}
            comparisons.append(row)
            assert within_one >= 0.999, row
        # A color edit after a cached speckle draw must update every channel.
        settings = replace(settings, brightness=100, invert=False,
                           dark_color=(44, 151, 23), light_color=(219, 33, 190))
        samples = tuple(replace(settings, angle_deg=213.7 + i * 0.3,
                                phase_deg=71.3 + i * 2, period_px=27.3 + i * 0.15)
                        for i in range(8))
        shader.draw(settings, width, height, antialias=False, motion_samples=samples)
        actual = read_pixels_rgb(context, width, height)
        # Binary signals with 0/255 endpoints avoid premature rounding in this
        # independent reference for averaging before color mapping.
        raw = [render_pattern(width, height, replace(sample, brightness=100)) / 255.0
               for sample in samples]
        signal = np.mean(raw, axis=0)
        dark, light = np.array(settings.dark_color), np.array(settings.light_color)
        expected = np.rint(dark + (light-dark) * signal[:, :, None])
        delta = np.abs(actual.astype(float) - expected)
        row = {"family": family, "within_one": float(np.mean(np.all(delta <= 1, axis=2))),
                   "max_error": float(delta.max())}
        temporal.append(row)
        assert row["within_one"] >= 0.999, row
        # Default palette must remain three identical copies of Mono8.
        default = replace(settings, dark_color=(0, 0, 0), light_color=(255, 255, 255),
                          brightness=73, invert=True)
        shader.draw(default, width, height, antialias=False)
        actual = read_pixels_rgb(context, width, height)
        assert np.array_equal(actual[:, :, 0], actual[:, :, 1])
        assert np.array_equal(actual[:, :, 0], actual[:, :, 2])
        delta = np.abs(actual.astype(int) - render_pattern(width, height, default)[:, :, None])
        assert np.mean(np.all(delta <= 1, axis=2)) >= 0.999, family
    # Compare colored temporal AA against the independent analytic stripe
    # coverage reference, rather than an already-quantized Mono8 frame.
    settings = PatternSettings(
        family="stripes", period_px=27.3, angle_deg=31.7, phase_deg=71.3,
        brightness=73, invert=True, dark_color=(18, 79, 143), light_color=(231, 158, 32),
    )
    samples = tuple(replace(settings, angle_deg=31.7 + i * 0.3,
                            phase_deg=71.3 + i * 2, period_px=27.3 + i * 0.15)
                    for i in range(8))
    shader.draw(settings, width, height, motion_samples=samples)
    actual = read_pixels_rgb(context, width, height)
    signal = np.mean([raw_reference(width, height, sample) / 186 for sample in samples], axis=0)
    dark, light = np.array(settings.dark_color), np.array(settings.light_color)
    expected = np.rint((dark + (light-dark) * signal[:, :, None]) * (186/255))
    delta = np.abs(actual.astype(float) - expected)
    aa = {"max_error": float(delta.max()), "mean_error": float(delta.mean())}
    assert delta.max() <= 1, aa
    gl_errors = errors(context)
    assert not gl_errors, gl_errors
    return {"rgb_comparisons": comparisons, "rgb_temporal_comparisons": temporal,
            "rgb_temporal_antialias": aa, "rgb_gl_errors": gl_errors}


def gpu_acceptance(rgb_only: bool = False) -> dict:
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QtGui.QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        raise RuntimeError("Cannot create OpenGL 3.3 context")
    surface = QtGui.QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        raise RuntimeError("Cannot activate OpenGL context")
    shader = PatternShader()
    shader.initialize(context)
    comparisons = []
    framebuffer = QtOpenGL.QOpenGLFramebufferObject(241, 173)
    assert framebuffer.isValid()
    framebuffer.bind()
    rgb = rgb_comparisons(shader, context)
    if rgb_only:
        report = {"graphics": identity(context), **rgb}
        framebuffer.release()
        del framebuffer
        shader.close()
        context.doneCurrent()
        return report
    families = (
        "stripes", "checker", "rings", "composite", "nested_square", "squircle",
        "spiral", "counter_spiral", "starburst", "speckle", "solid",
    )
    for family in families:
        for waveform, invert in (("binary", False), ("sinusoidal", True)):
            settings = PatternSettings(
                family=family, waveform=waveform, invert=invert,
                period_px=27.3, angle_deg=213.7,
                phase_deg=431.3 if family == "starburst" and invert else 71.3,
                center_x_percent=13, center_y_percent=-7,
                phase_tilt_x_deg=34, phase_tilt_y_deg=-67, brightness=73,
                spiral_arms=3, spokes=17, duty_percent=37,
            )
            shader.draw(settings, 241, 173, antialias=False)
            image = framebuffer.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            actual = np.frombuffer(image.constBits(), np.uint8).reshape(173, 241, 4)[:, :, 0]
            expected = render_pattern(241, 173, settings)
            delta = np.abs(actual.astype(int) - expected.astype(int))
            fraction = float(np.mean(delta <= 1))
            comparisons.append({"family": family, "waveform": waveform, "within_one": fraction})
            if fraction < 0.999:
                raise AssertionError(f"GPU/CPU mismatch: {comparisons[-1]}, max={delta.max()}")

    # Fractional motion of a binary edge must change pixel coverage gradually.
    settings = PatternSettings(period_px=40)
    shader.draw(settings, 241, 173)
    image = framebuffer.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    coverage = np.frombuffer(image.constBits(), np.uint8).reshape(173, 241, 4)[:, :, 0]
    assert np.any((coverage > 0) & (coverage < 255)), "Missing subpixel edge coverage"
    # Poison caller GL state and the old contents. The draw must own all state
    # needed to overwrite the complete framebuffer, rather than leave stale tiles.
    gl = context.functions()
    gl.glClearColor(0.3, 0.4, 0.7, 1)
    gl.glClear(0x00004000)
    gl.glEnable(0x0C11)  # SCISSOR_TEST
    gl.glScissor(0, 0, 16, 16)
    gl.glColorMask(False, False, False, False)
    gl.glEnable(0x0B44)  # CULL_FACE
    gl.glCullFace(0x0404)  # FRONT
    gl.glEnable(0x8C89)  # RASTERIZER_DISCARD
    shader.draw(settings, 241, 173, antialias=False)
    image = framebuffer.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    actual = np.frombuffer(image.constBits(), np.uint8).reshape(173, 241, 4)[:, :, 0]
    assert np.max(np.abs(actual.astype(int) - render_pattern(241, 173, settings))) <= 1

    temporal_comparisons = []
    for family in families:
        settings = PatternSettings(family=family, phase_deg=11.3, angle_deg=13.7)
        samples = tuple(replace(settings, angle_deg=13.7 + i * 0.3,
                                phase_deg=11.3 + i * 2, period_px=50 + i * 0.15)
                        for i in range(8))
        shader.draw(settings, 241, 173, antialias=False, motion_samples=samples)
        image = framebuffer.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        actual = np.frombuffer(image.constBits(), np.uint8).reshape(173, 241, 4)[:, :, 0]
        expected = np.rint(np.mean([render_pattern(241, 173, s) for s in samples], axis=0))
        fraction = float(np.mean(np.abs(actual.astype(float) - expected) <= 1))
        temporal_comparisons.append({"family": family, "within_one": fraction})
        assert fraction >= 0.999, temporal_comparisons[-1]
    seam_errors = {}
    for family in ("spiral", "counter_spiral", "starburst"):
        settings = PatternSettings(family=family, period_px=27.3, spiral_arms=3, spokes=17)
        shader.draw(settings, 241, 173)
        image = framebuffer.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        actual = np.frombuffer(image.constBits(), np.uint8).reshape(173, 241, 4)[:, :, 0]
        # A supersampled reference crosses atan's -pi/pi seam without taking
        # its discontinuous numerical derivative. Exclude the singular centre.
        reference = render_pattern(241 * 8, 173 * 8, replace(settings, period_px=27.3 * 8))
        reference = reference.reshape(173, 8, 241, 8).mean(axis=(1, 3))
        error = float(np.mean(np.abs(actual[84:89, 10:70] - reference[84:89, 10:70])))
        seam_errors[family] = error
        assert error < 12, f"Angular antialias seam: {family}, mean error {error}"
    framebuffer.release()
    del framebuffer

    # glFinish measures completed GPU work rather than CPU command enqueue time.
    framebuffer = QtOpenGL.QOpenGLFramebufferObject(2160, 3840)
    assert framebuffer.isValid()
    framebuffer.bind()
    benchmarks = []
    for family in ("stripes", "rings", "spiral", "squircle"):
        settings = PatternSettings(family=family)
        shader.draw(settings, 2160, 3840)
        context.functions().glFinish()
        times = []
        for index in range(12):
            start = perf_counter()
            shader.draw(replace(settings, phase_deg=index * 2.0), 2160, 3840)
            context.functions().glFinish()
            times.append((perf_counter() - start) * 1000)
        benchmarks.append({"family": family, "median_ms": float(np.median(times)),
                               "p95_ms": float(np.percentile(times, 95))})
    framebuffer.release()
    del framebuffer
    shader.close()
    context.doneCurrent()
    return {**rgb, "comparisons": comparisons, "temporal_comparisons": temporal_comparisons,
            "gl_state_recovery": "passed", "aa_seam_mean_error": seam_errors,
            "gpu_2160x3840": benchmarks}


def onscreen_acceptance(app: QtWidgets.QApplication, screen: int) -> dict:
    if not 0 <= screen < len(app.screens()):
        raise ValueError("Screen index is not connected")
    output = PatternOutputWindow()
    times = []
    errors = []
    output.frameSwapped.connect(lambda: times.append(perf_counter()))
    output.renderFailed.connect(errors.append)
    output.set_settings(PatternSettings(
        family="spiral", mode="continuous", speed_deg_s=12,
        phase_speed_cycles_s=0.15,
    ))
    output.show_on_screen(screen)
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(4500, loop.quit)
    loop.exec()
    output.hide_output()
    assert not errors, errors
    assert output.isValid(), "GPU surface failed to initialize"
    assert len(times) > 20, f"Only {len(times)} frames swapped"
    intervals = np.diff(times[10:]) * 1000
    report = {
        "screen": app.screens()[screen].name(),
        "requested_swap_interval": output.format().swapInterval(),
        "actual_swap_interval": output.context().format().swapInterval(),
        "frames": len(times), "median_ms": float(np.median(intervals)),
        "p95_ms": float(np.percentile(intervals, 95)),
        "maximum_ms": float(np.max(intervals)),
        "over_25ms": int(np.sum(intervals > 25)),
    }
    # Hidden/static outputs must not continue requesting animation frames.
    # One completed swap may already have a signal queued when hide is called.
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(100, loop.quit)
    loop.exec()
    count = len(times)
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(100, loop.quit)
    loop.exec()
    assert len(times) == count
    output.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=int)
    parser.add_argument("--rgb-only", action="store_true",
                        help="Run bounded RGB checks without the unrelated 4K benchmark")
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    report = gpu_acceptance(rgb_only=args.rgb_only)
    if args.screen is not None:
        report["onscreen"] = onscreen_acceptance(app, args.screen)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
