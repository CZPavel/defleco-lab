import json

import cv2
import numpy as np
import pytest

from defleco_lab.gui.surface_io import (
    export_surface,
    load_preset,
    original_rgb,
    render_surface,
    save_preset,
)
from defleco_lab.processing.base import ProcessingResult


def test_bgr16_display_does_not_clip_to_255():
    image = np.zeros((8, 9, 3), np.uint16)
    image[..., 2] = 32768
    original = image.copy()
    rgb = original_rgb(image)
    assert rgb[0, 0].tolist() == [128, 0, 0]
    np.testing.assert_array_equal(image, original)


def test_enhanced_overlay_uses_same_enhanced_base_and_preserves_signed_response():
    image = np.full((1, 3), 128, np.uint8)
    response = np.array([[-.1, 0, .1]], np.float32)
    result = ProcessingResult(response)
    common = {"range": "fixed", "low": 0, "high": .1,
              "polarity": "absolute", "enhanced_polarity": "signed"}
    enhanced = render_surface(image, result, {**common, "mode": "enhanced"})
    heatmap = render_surface(image, result, {**common, "mode": "heatmap"})
    overlay = render_surface(image, result, {**common, "mode": "overlay",
                                            "overlay_base": "enhanced", "opacity": .5})
    np.testing.assert_array_equal(overlay[0, 1], enhanced[0, 1])
    expected = np.rint((enhanced[0, 0].astype(float) + heatmap[0, 0]) * .5).astype(np.uint8)
    np.testing.assert_array_equal(overlay[0, 0], expected)
    assert enhanced[0, 0, 0] < 128 < enhanced[0, 2, 0]
    np.testing.assert_array_equal(result.primary, response)


def test_enhancement_display_gain_does_not_change_response():
    image = np.full((2, 2), .4, np.float32)
    response = np.full((2, 2), .5, np.float32)
    enhanced = np.full((2, 2), .6, np.float32)
    result = ProcessingResult(response, {"enhanced": enhanced})
    actual = render_surface(image, result, {"mode": "enhanced", "enhancement_gain": 1.5})
    assert actual[0, 0, 0] == 179
    np.testing.assert_array_equal(result.primary, response)
    np.testing.assert_array_equal(result.intermediates["enhanced"], enhanced)


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_export_roundtrip_and_refuse_overwrite(tmp_path, suffix):
    image = np.full((8, 9), 32768, np.uint16)
    response = np.linspace(-0.5, 0.5, 72, dtype=np.float32).reshape(8, 9)
    result = ProcessingResult(response.copy())
    settings = {"mode": "heatmap", "range": "fixed", "low": -0.5, "high": 0.5}
    path = tmp_path / ("škrábanec" + suffix)
    metadata = {
        "method_id": "example",
        "parameters": {},
        "roi": {"x": 10, "y": 20, "width": 9, "height": 8},
    }
    paths = export_surface(path, image, result, metadata, settings)
    np.testing.assert_array_equal(np.load(paths["response"]), response)
    decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(
        cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), render_surface(image, result, settings)
    )
    sidecar = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert sidecar["roi"] == metadata["roi"]
    assert sidecar["display_settings"] == settings
    assert sidecar["response_dtype"] == "float32"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_surface(path, image, result, metadata, settings)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(result.primary, response)


def test_existing_sidecar_keeps_original_and_rolls_back_new_output(tmp_path):
    path = tmp_path / "result.png"
    path.with_suffix(".json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_surface(
            path, np.zeros((3, 4), np.uint8), ProcessingResult(np.zeros((3, 4), np.float32)), {}, {}
        )
    assert not path.exists()
    assert path.with_suffix(".json").read_text() == "keep"


def test_source_overwrite_is_rejected(tmp_path):
    path = tmp_path / "original.png"
    with pytest.raises(ValueError, match="original"):
        export_surface(
            path,
            np.zeros((3, 4), np.uint8),
            ProcessingResult(np.zeros((3, 4), np.float32)),
            {"source_filename": str(path)},
            {},
        )


def test_zero_response_overlay_is_transparent_and_response_unchanged():
    image = np.full((3, 4), 128, np.uint8)
    response = np.zeros((3, 4), np.float32)
    result = ProcessingResult(response)
    rgb = render_surface(
        image, result, {"mode": "overlay", "range": "fixed", "low": 0, "high": 1, "opacity": 1}
    )
    np.testing.assert_array_equal(rgb, original_rgb(image))
    assert response.max() == 0


def test_signed_overlay_shows_both_signs_and_keeps_zero_transparent():
    image = np.full((1, 3), 128, np.uint8)
    result = ProcessingResult(np.array([[-1, 0, 1]], np.float32))
    rgb = render_surface(
        image, result, {"mode": "overlay", "range": "fixed", "low": -1, "high": 1, "opacity": 1}
    )
    assert rgb[0, 1].tolist() == [128, 128, 128]
    assert rgb[0, 0].tolist() != [128, 128, 128]
    assert rgb[0, 2].tolist() != [128, 128, 128]


def test_roi_overlay_uses_full_source_normalization():
    image = np.array([[100.0, 200.0]], np.float32)
    result = ProcessingResult(
        np.zeros((1, 2), np.float32), {"_source_normalized": np.array([[0.1, 0.2]], np.float32)}
    )
    rgb = render_surface(image, result, {"mode": "overlay"})
    np.testing.assert_array_equal(rgb[0, :, 0], [26, 51])


@pytest.mark.parametrize(
    "settings",
    [
        {"range": "fixed", "low": 1, "high": 0},
        {"opacity": float("nan")},
        {"polarity": "other"},
        {"mode": "other"},
        {"percentile_low": 99, "percentile_high": 1},
    ],
)
def test_invalid_display_rejected(settings):
    with pytest.raises(ValueError):
        render_surface(
            np.zeros((3, 4), np.uint8), ProcessingResult(np.zeros((3, 4), np.float32)), settings
        )


def test_preset_roundtrip_and_invalid_algorithm(tmp_path):
    from defleco_lab.processing.surface import surface_registry

    info = surface_registry.infos()[0]
    config = {
        "method_id": info.id,
        "parameters": {},
        "display_settings": {"mode": "grayscale", "polarity": "signed"},
    }
    path = tmp_path / "preset.json"
    save_preset(path, config)
    assert load_preset(path)["method_id"] == info.id
    path.write_text('{"method_id":"does_not_exist"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_preset(path)
