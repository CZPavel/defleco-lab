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


def test_pattern_renderer_produces_distinct_families() -> None:
    images = [
        render_pattern(160, 120, PatternSettings(family=name, period_px=20))
        for name in ("stripes", "checker", "spiral", "rings")
    ]
    for image in images:
        assert image.shape == (120, 160)
        assert image.dtype == np.uint8
        assert image.min() == 0
        assert image.max() == 255
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[0], images[2])
    assert not np.array_equal(images[2], images[3])


def test_sinusoidal_pattern_contains_intermediate_gray_levels() -> None:
    image = render_pattern(160, 120, PatternSettings(family="stripes", period_px=24, waveform="sinusoidal"))
    # Integer pixel sampling of a 24 px sine contains repeated symmetric levels,
    # but it must remain a genuine multi-level carrier rather than a binary pattern.
    assert len(np.unique(image)) > 8
