"""CPU palette contracts; actual GL readback lives in pattern_render_acceptance."""
from dataclasses import replace

import numpy as np
import pytest

from defleco_lab.gui.pattern_output import PatternSettings, render_pattern, render_pattern_rgb

FAMILIES = ("stripes", "checker", "rings", "composite", "nested_square", "squircle",
            "spiral", "counter_spiral", "starburst", "speckle", "solid")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("waveform,invert,brightness", [
    ("binary", False, 100), ("sinusoidal", True, 73), ("binary", True, 0),
])
def test_default_rgb_is_exact_mono8_replication(family, waveform, invert, brightness):
    settings = PatternSettings(family=family, waveform=waveform, invert=invert,
                               brightness=brightness, phase_deg=71.3, angle_deg=31.7)
    mono = render_pattern(83, 67, settings)
    rgb = render_pattern_rgb(83, 67, settings)
    assert rgb.shape == (67, 83, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, np.repeat(mono[:, :, None], 3, axis=2))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("invert", [False, True])
def test_binary_palette_maps_signal_before_brightness(family, invert):
    settings = PatternSettings(family=family, invert=invert, brightness=73,
                               dark_color=(19, 83, 141), light_color=(231, 157, 32),
                               phase_deg=71.3, angle_deg=31.7)
    # Composite has the exact extra level 0.5; avoid quantizing it in this reference.
    mono = render_pattern(83, 67, replace(settings, brightness=100))
    signal = mono.astype(float) / 255
    if family == "composite":
        signal = np.where(mono == 128, 0.5, signal)
    dark, light = np.array(settings.dark_color), np.array(settings.light_color)
    expected = np.rint((dark + (light-dark) * signal[:, :, None]) * (186/255))
    np.testing.assert_array_equal(render_pattern_rgb(83, 67, settings), expected.astype(np.uint8))


def test_sinusoidal_palette_uses_unrounded_signal_and_both_endpoints():
    settings = PatternSettings(waveform="sinusoidal", period_px=8,
                               dark_color=(12, 80, 210), light_color=(240, 140, 30))
    actual = render_pattern_rgb(9, 3, settings)[1]
    # x = -4..4: centre is half-signal, +/- quarter period hit endpoints.
    np.testing.assert_array_equal(actual[2], settings.dark_color)
    np.testing.assert_array_equal(actual[4], (126, 110, 120))
    np.testing.assert_array_equal(actual[6], settings.light_color)


def test_black_brightness_is_black_even_with_nonzero_dark_endpoint():
    settings = PatternSettings(family="speckle", brightness=0, invert=True,
                               dark_color=(40, 150, 90), light_color=(200, 20, 240))
    assert not render_pattern_rgb(51, 39, settings).any()
