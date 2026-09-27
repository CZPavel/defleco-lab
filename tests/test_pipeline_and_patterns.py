from __future__ import annotations

import numpy as np

from defleco_lab.gui.pattern_output import PatternSettings, render_pattern
from defleco_lab.processing.pipeline import apply_postprocessing, apply_preprocessing


def test_preprocessing_defaults_preserve_mono8_intensity() -> None:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    out = apply_preprocessing(image, None)
    assert out.dtype == np.float32
    assert np.allclose(out, image)


def test_local_background_postprocessing_removes_constant_response() -> None:
    response = np.full((32, 32), 7.0, np.float32)
    out = apply_postprocessing(response, {"local_background_sigma": 4.0, "absolute": True})
    assert np.max(np.abs(out)) < 1e-5


def test_pattern_renderer_covers_integrated_v02_families() -> None:
    families = (
        "stripes",
        "checker",
        "rings",
        "composite",
        "nested_square",
        "squircle",
        "spiral",
        "counter_spiral",
        "starburst",
        "speckle",
        "solid",
    )
    images = {
        name: render_pattern(160, 120, PatternSettings(family=name, period_px=20))
        for name in families
    }
    for image in images.values():
        assert image.shape == (120, 160)
        assert image.dtype == np.uint8
        assert 0 <= int(image.min()) <= int(image.max()) <= 255

    for name in families:
        if name != "solid":
            assert np.ptp(images[name]) > 0
    assert not np.array_equal(images["stripes"], images["checker"])
    assert not np.array_equal(images["spiral"], images["counter_spiral"])
    assert not np.array_equal(images["nested_square"], images["squircle"])


def test_sinusoidal_pattern_contains_intermediate_gray_levels() -> None:
    image = render_pattern(160, 120, PatternSettings(family="stripes", period_px=24, waveform="sinusoidal"))
    # Integer pixel sampling of a 24 px sine contains repeated symmetric levels,
    # but it must remain a genuine multi-level carrier rather than a binary pattern.
    assert len(np.unique(image)) > 8


def test_radial_pattern_is_orientation_agnostic_in_portrait_geometry() -> None:
    settings = PatternSettings(family="rings", period_px=24, waveform="sinusoidal")
    portrait = render_pattern(216, 384, settings)
    landscape = render_pattern(384, 216, settings)
    assert portrait.shape == (384, 216)
    assert landscape.shape == (216, 384)
    # Rings depend only on radius from the image centre, so swapping width/height
    # must only transpose the image. This catches accidental landscape assumptions.
    assert np.array_equal(portrait.T, landscape)
