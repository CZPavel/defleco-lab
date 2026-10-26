from __future__ import annotations

from defleco_lab.gui.pattern_output import PatternSettings
from tools.pattern_artifact_diagnostics import Comparison, reference


def test_artifact_check_rejects_small_corrupt_patch_even_when_global_fraction_is_tiny():
    settings = PatternSettings(brightness=0)
    actual = reference(512, 512, settings)
    actual[100:108, 100:108] = 255
    result = Comparison(None).check(actual, settings, "injected_patch")
    assert result["fraction_error_over_3"] < 0.001
    assert result["max_error"] == 255
    assert result["worst_64px_tile_bad_fraction"] > 0
    assert not result["passed"]


def test_artifact_check_accepts_reference_and_rejects_one_severe_bad_pixel():
    settings = PatternSettings()
    actual = reference(320, 240, settings)
    assert Comparison(None).check(actual, settings, "reference")["passed"]
    actual[20, 20] = 255 - actual[20, 20]
    assert not Comparison(None).check(actual, settings, "bad_pixel")["passed"]
