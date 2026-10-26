from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from defleco_lab.gui.pattern_output import (
    PatternControlPanel,
    PatternOutputWindow,
    PatternSettings,
    render_pattern,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_spiral_rotation_crosses_half_turn_without_discontinuity():
    app = _app()
    output = PatternOutputWindow()
    output.set_settings(PatternSettings(family="spiral", angle_deg=179.9, speed_deg_s=20))
    before = render_pattern(320, 240, output.effective_settings)
    output._advance(0.01)
    assert abs(output.effective_settings.angle_deg - 180.1) < 1e-8
    after = render_pattern(320, 240, output.effective_settings)
    assert np.mean(before != after) < 0.01
    output.close()
    assert app is not None


def test_elapsed_time_not_frame_count_controls_animation():
    app = _app()
    settings = PatternSettings(
        family="spiral", mode="continuous", speed_deg_s=-25,
        phase_speed_cycles_s=0.4, breathing_depth_percent=15,
    )
    regular, stalled = PatternOutputWindow(), PatternOutputWindow()
    for output in (regular, stalled):
        output.set_settings(settings)
    for _ in range(60):
        regular._advance(1 / 60)
    for seconds in (0.17, 0.013, 0.407, 0.41):
        stalled._advance(seconds)
    a, b = regular.effective_settings, stalled.effective_settings
    assert abs(a.angle_deg - b.angle_deg) < 1e-8
    assert abs(a.phase_deg - b.phase_deg) < 1e-8
    assert abs(a.period_px - b.period_px) < 1e-8
    regular.close()
    stalled.close()
    assert app is not None


def test_phase_step_and_ring_rotation_compatibility():
    app = _app()
    output = PatternOutputWindow()
    output.set_settings(PatternSettings(mode="step", step_target="phase", step_deg=90))
    before = render_pattern(120, 80, output.effective_settings)
    output.step()
    assert output.effective_settings.phase_deg == 90
    assert not np.array_equal(before, render_pattern(120, 80, output.effective_settings))
    output.set_settings(PatternSettings(family="rings", speed_deg_s=20))
    output._advance(0.5)
    assert output.effective_settings.phase_deg == 10
    assert output.effective_settings.angle_deg == 0
    output.close()
    assert app is not None


def test_starburst_phase_crosses_360_without_inverting_whole_image():
    app = _app()
    output = PatternOutputWindow()
    output.set_settings(PatternSettings(
        family="starburst", phase_deg=359.9, phase_speed_cycles_s=1,
        speed_deg_s=0, spokes=17,
    ))
    before = render_pattern(321, 241, output.effective_settings)
    output._advance(0.2 / 360)
    assert abs(output.effective_settings.phase_deg - 360.1) < 1e-8
    after = render_pattern(321, 241, output.effective_settings)
    assert np.mean(before != after) < 0.01
    output.close()
    assert app is not None


def test_hidden_edits_are_lazy_and_panel_uses_native_cadence():
    app = _app()
    panel = PatternControlPanel()
    panel.output._shader.draw = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Hidden edits must not render")
    )
    panel.phase.setValue(35)
    panel.period.setValue(40)
    assert panel.settings().phase_deg == 35
    assert panel.settings().render_scale == 1
    assert panel.settings().animation_fps == 60
    panel.output.set_settings(replace(panel.settings(), mode="continuous"))
    assert not panel.output._elapsed.isValid()
    panel.shutdown()
    panel.close()
    assert app is not None


def test_display_diagnostics_distinguish_framebuffer_and_operator_panel_size():
    import json

    app = _app()
    output = PatternOutputWindow()
    output.resize(1080, 1920)
    output.set_panel_native_size(2160, 3840)
    diagnostics = output.display_diagnostics()
    assert diagnostics["framebuffer_width"] == 1080
    assert diagnostics["framebuffer_height"] == 1920
    assert diagnostics["estimated_scale_x"] == diagnostics["estimated_scale_y"] == 2
    assert not diagnostics["panel_scaling_warning"]
    assert "operator supplied" in diagnostics["panel_size_source"]
    assert diagnostics["context_lifecycle"][0]["context_exists"] is False
    assert diagnostics["process"]["pid"] > 0
    assert json.loads(json.dumps(diagnostics))["scanout_verified"] is False
    output.resize(1000, 1920)
    assert output.display_diagnostics()["panel_scaling_warning"]
    output.set_panel_native_size(0, 0)
    assert output.display_diagnostics()["estimated_scale_x"] is None
    output.close()
    assert app is not None


def test_capture_settle_starts_after_presentation_and_aborts_on_lost_output(monkeypatch, tmp_path):
    from defleco_lab.gui import main_window

    events = []
    now = [100.0]
    monkeypatch.setattr(main_window, "perf_counter", lambda: now[0])
    output = SimpleNamespace(has_presented_frame=False, hide_output=lambda: events.append("hidden"))
    fake = SimpleNamespace(
        _experiment_root=tmp_path, _experiment_index=0,
        _experiment_case_started=0.0, _experiment_case_requested=100.0,
        pattern_output=SimpleNamespace(output=output),
        experiment_timer=SimpleNamespace(isActive=lambda: True, stop=lambda: events.append("stopped")),
        experiment=SimpleNamespace(failed=events.append),
        _write_capture_manifest=lambda status: events.append(status),
    )
    fake._pattern_output_failed = lambda message: main_window.MainWindow._pattern_output_failed(
        fake, message
    )
    now[0] = 102.0
    main_window.MainWindow._experiment_poll(fake)
    assert fake._experiment_case_started == 0
    assert events == []
    output.has_presented_frame = True
    main_window.MainWindow._experiment_poll(fake)
    assert fake._experiment_case_started == 102.0
    output.has_presented_frame = False
    main_window.MainWindow._experiment_poll(fake)
    assert "stopped" in events and "error" in events
    assert any("not presented" in str(event) for event in events)
