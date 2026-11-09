import json

import cv2
import numpy as np
import pytest

from defleco_lab.processing.surface import kernels, normalize_input, surface_registry
from defleco_lab.processing.surface.fusion import (
    FUSION_VERSION,
    fusion_registry,
    recipe_metadata,
)


def scene(dtype=np.uint8, color=False):
    _, x = np.mgrid[:96, :112]
    result = (0.30 + 0.25 * x / 111).astype(np.float32)
    cv2.line(result, (12, 20), (100, 20), 0.9, 2)
    cv2.line(result, (15, 70), (90, 35), 0.08, 3)
    cv2.circle(result, (70, 70), 4, 0.8, -1)
    if color:
        result = np.stack([result * 0.8, result, result * 0.7], axis=-1)
    if dtype == np.uint8:
        result = np.rint(result * 255).astype(np.uint8)
    elif dtype == np.uint16:
        result = np.rint(result * 65535).astype(np.uint16)
    return result


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("color", [False, True])
def test_all_seven_methods_preserve_source_and_return_finite_float_maps(dtype, color):
    image = scene(dtype, color)
    before = image.copy()
    assert len(fusion_registry.infos()) == 7
    for info in fusion_registry.infos():
        result = fusion_registry.create(info.id).process([image])
        assert result.primary.shape == image.shape[:2]
        assert result.primary.dtype == np.float32 and np.isfinite(result.primary).all()
        assert result.intermediates["enhanced"].shape == image.shape
        assert result.intermediates["enhanced"].dtype == np.float32
        assert np.isfinite(result.intermediates["enhanced"]).all()
        assert 0 <= result.diagnostics["enhanced_clip_fraction"] <= 1
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("info", fusion_registry.infos(), ids=lambda i: i.id)
def test_zero_final_intensity_preserves_original_in_all_recipes(info):
    image = scene(np.uint16, True)
    enhanced = (
        fusion_registry.create(info.id, intensity=0).process([image]).intermediates["enhanced"]
    )
    np.testing.assert_array_equal(enhanced, normalize_input(image)[0])


@pytest.mark.parametrize(
    "name", ["fusion_k01", "fusion_k02", "fusion_k03", "fusion_k05", "fusion_k07"]
)
def test_bright_and_dark_lines_are_not_interchanged(name):
    image = np.full((96, 112), 0.5, np.float32)
    cv2.line(image, (10, 25), (100, 25), 0.9, 3)
    cv2.line(image, (10, 65), (100, 65), 0.1, 3)
    kwargs = (
        {"clahe_enabled": False}
        if "clahe_enabled" in fusion_registry.create(name).parameters
        else {}
    )
    if name in {"fusion_k01", "fusion_k05"}:
        kwargs["morphology_angle"] = 90.0
    bright = fusion_registry.create(name, polarity="bright", **kwargs).process([image]).primary
    dark = fusion_registry.create(name, polarity="dark", **kwargs).process([image]).primary
    assert bright[25, 50] > 0.05 and bright[25, 50] > abs(bright[65, 50]) * 5
    assert dark[65, 50] < -0.05 and abs(dark[65, 50]) > abs(dark[25, 50]) * 5


@pytest.mark.parametrize(
    "name,enabled",
    [
        ("fusion_k01", "clahe_enabled"),
        ("fusion_k01", "morphology_enabled"),
        ("fusion_k02", "clahe_enabled"),
        ("fusion_k02", "hessian_enabled"),
        ("fusion_k03", "median_enabled"),
        ("fusion_k03", "hessian_enabled"),
        ("fusion_k04", "fft_enabled"),
        ("fusion_k04", "clahe_enabled"),
        ("fusion_k05", "local_enabled"),
        ("fusion_k05", "morphology_enabled"),
        ("fusion_k06", "unsharp_enabled"),
        ("fusion_k06", "clahe_enabled"),
        ("fusion_k07", "clahe_enabled"),
        ("fusion_k07", "hessian_enabled"),
        ("fusion_k07", "morphology_enabled"),
        ("fusion_k07", "median_enabled"),
    ],
)
def test_turning_off_each_step_changes_selected_scene(name, enabled):
    image = scene(np.uint8)
    default = fusion_registry.create(name).process([image])
    altered = fusion_registry.create(name, **{enabled: False}).process([image])
    assert not np.array_equal(default.intermediates["enhanced"], altered.intermediates["enhanced"])


def test_disabled_and_zero_weight_branches_never_run(monkeypatch):
    calls = []
    originals = {
        name: getattr(kernels, name)
        for name in ("hessian_signed", "directional_morph", "median_residual")
    }
    for name, original in originals.items():

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(kernels, name, spy)
    result = fusion_registry.create(
        "fusion_k07", hessian_enabled=False, morphology_weight=0
    ).process([scene()])
    assert "hessian_signed" not in calls and "directional_morph" not in calls
    assert "median_residual" in calls
    assert {k for k in result.intermediates if k.startswith("branch_")} == {"branch_median"}


def test_changing_weight_changes_fusion_but_not_branch_computations():
    a = fusion_registry.create("fusion_k07").process([scene()])
    b = fusion_registry.create("fusion_k07", hessian_weight=0.9).process([scene()])
    for branch in ("hessian", "morphology", "median"):
        np.testing.assert_array_equal(
            a.intermediates["branch_" + branch], b.intermediates["branch_" + branch]
        )
    assert not np.allclose(a.primary, b.primary)


def test_signed_background_intermediates_are_passed_to_hessian_and_morph_without_renormalizing(
    monkeypatch,
):
    images = []
    original_hessian = kernels.hessian_signed
    original_morph = kernels.directional_morph

    def capture_hessian(source, sigma):
        images.append(source.copy())
        return original_hessian(source, sigma)

    monkeypatch.setattr(kernels, "hessian_signed", capture_hessian)
    fusion_registry.create("fusion_k03").process([scene(np.float32)])
    assert images and images[0].min() < 0 < images[0].max()
    images.clear()

    def capture_morph(source, size, angle, operation):
        images.append(source.copy())
        return original_morph(source, size, angle, operation)

    monkeypatch.setattr(kernels, "directional_morph", capture_morph)
    fusion_registry.create("fusion_k05").process([scene(np.float32)])
    assert images and images[0].min() < 0 < images[0].max()


def test_preset_roundtrip_reproduces_numerical_result():
    options = {
        "hessian_sigmas": [1.0, 3.0, 6.0],
        "hessian_weight": 0.7,
        "morphology_weight": 0.1,
        "fusion_mode": "Maximum",
        "intensity": 0.8,
    }
    meta = recipe_metadata("fusion_k07", options)
    restored = json.loads(json.dumps(meta))
    assert restored["recipe_id"] == "fusion_k07" and restored["recipe_version"] == FUSION_VERSION
    assert restored["weights"]["hessian_weight"] == 0.7
    assert "hessian" in restored["active_steps"] and restored["normalization"]
    a = fusion_registry.create("fusion_k07", **options).process([scene()])
    b = fusion_registry.create(restored["recipe_id"], **restored["parameters"]).process([scene()])
    np.testing.assert_array_equal(a.primary, b.primary)
    np.testing.assert_array_equal(a.intermediates["enhanced"], b.intermediates["enhanced"])


def test_preview_scales_spatial_parameters_and_exposes_effective_values():
    result = fusion_registry.create("fusion_k07").process([scene()], spatial_scale=0.5)
    assert result.diagnostics["effective_morphology_size"] == 17
    assert result.diagnostics["effective_median_size"] == 17
    assert json.loads(result.diagnostics["effective_hessian_sigmas"]) == [1.0, 2.0]


@pytest.mark.parametrize(
    "params",
    [
        {"hessian_sigmas": []},
        {"hessian_sigmas": [0]},
        {"hessian_sigmas": [1, 2, 3, 4, 5]},
        {"hessian_sigmas": [float("nan")]},
        {"hessian_view": "scale_3"},
        {"morphology_size": 30},
        {"hessian_weight": -1},
        {"intensity": float("inf")},
    ],
)
def test_invalid_parameters_rejected(params):
    with pytest.raises(ValueError):
        fusion_registry.create("fusion_k07", **params)


def test_extracted_kernels_match_existing_surface_output():
    image = scene(np.float32)
    normalized, _ = normalize_input(image)
    cases = [
        ("surface_clahe", {"clip": 2.0, "grid": 8}, kernels.clahe_luminance(normalized, 2.0, 8)),
        ("surface_median_residual", {"size": 31}, kernels.median_residual(normalized, 31)),
        ("surface_local_residual", {"sigma": 4.0}, kernels.local_residual(normalized, 4.0)),
        ("surface_hessian", {"sigma": 2.0}, kernels.hessian_signed(normalized, 2.0)),
        ("surface_fft_highpass", {"sigma": 4.0}, kernels.fft_highpass(normalized, 4.0)),
        (
            "surface_directional_top_hat",
            {"size": 31, "angle": 0.0},
            kernels.directional_morph(normalized, 31, 0.0, cv2.MORPH_TOPHAT),
        ),
    ]
    for name, params, expected in cases:
        actual = surface_registry.create(name, **params).process([image]).primary
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, err_msg=name)


def test_uniform_input_does_not_trigger_median_branch_quantization():
    result = fusion_registry.create(
        "fusion_k07", clahe_enabled=False, hessian_weight=0, morphology_weight=0
    ).process([np.full((64, 64), 0.5, np.float32)])
    assert np.max(np.abs(result.primary)) == 0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("level", [0.1, 0.5, 0.9])
def test_constant_images_do_not_create_fusion_features(dtype, color, level):
    image = np.full((64, 64, 3) if color else (64, 64), level, np.float32)
    if dtype != np.float32:
        image = np.rint(image * np.iinfo(dtype).max).astype(dtype)
    for info in fusion_registry.infos():
        result = fusion_registry.create(info.id).process([image])
        assert np.max(np.abs(result.primary)) < 1e-4, info.id


def test_float_median_preserves_weak_mono16_contrast_and_zero_constant():
    image = np.full((64, 64), 32768, np.uint16)
    image[25:30, 10:54] += 2
    normalized, _ = normalize_input(image)
    response = kernels.median_residual(normalized, 31, float_precision=True)
    assert 0 < response[27, 32] < 0.00004
    assert response[10, 10] == 0
    result = fusion_registry.create("fusion_k03").process([image])
    assert result.intermediates["median_residual_signed"][27, 32] > 0


@pytest.mark.parametrize(
    "method,first,second",
    [
        ("fusion_k03", "median_enabled", "hessian_enabled"),
        ("fusion_k05", "local_enabled", "morphology_enabled"),
    ],
)
def test_first_step_still_visible_when_second_disabled(method, first, second):
    image = scene()
    first_only = fusion_registry.create(method, **{second: False}).process([image])
    both_off = fusion_registry.create(method, **{first: False, second: False}).process([image])
    assert np.max(np.abs(first_only.primary)) > 0
    assert np.max(np.abs(both_off.primary)) == 0
    np.testing.assert_array_equal(both_off.intermediates["enhanced"], normalize_input(image)[0])


def test_selected_hessian_scale_is_only_scale_computed(monkeypatch):
    original = kernels.hessian_signed
    calls = []

    def spy(image, sigma):
        calls.append(sigma)
        return original(image, sigma)

    monkeypatch.setattr(kernels, "hessian_signed", spy)
    result = fusion_registry.create(
        "fusion_k02", hessian_sigmas=[1, 2, 4, 8], hessian_view="scale_3"
    ).process([scene()])
    assert calls == [4.0]
    assert "hessian_scale_3_normalization_scale" in result.diagnostics
    assert "hessian_scale_1_normalization_scale" not in result.diagnostics


def test_k06_unsharp_clipping_is_reported_only_at_final_output():
    image = np.full((64, 64), 0.1, np.float32)
    image[20:40, 20:40] = 0.95
    result = fusion_registry.create(
        "fusion_k06", clahe_enabled=False, unsharp_amount=4, intensity=1.0
    ).process([image])
    assert result.diagnostics["unsharp_clip_fraction"] > 0
    assert result.diagnostics["enhanced_clip_fraction"] > 0
    assert result.intermediates["enhanced"].max() == 1


def test_zero_weight_branch_is_not_listed_as_active():
    metadata = recipe_metadata("fusion_k07", {"hessian_weight": 0})
    assert "hessian" in metadata["enabled_steps"]
    assert "hessian" not in metadata["active_steps"]
    assert metadata["step_parameters"]["hessian_enabled"] is True
