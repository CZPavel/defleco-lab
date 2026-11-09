import numpy as np
import pytest

from defleco_lab.processing.surface import normalize_input, surface_registry


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("color", [False, True])
def test_all_methods_finite_shape_and_source_unchanged(dtype, color):
    _y, x = np.mgrid[:48, :56]
    a = (0.35 + 0.2 * np.sin(x / 4) + 0.25 * ((x > 25) & (x < 29))).astype(np.float32)
    if color:
        a = np.stack([a, a * 0.8, a * 0.6], axis=-1)
    if dtype != np.float32:
        a = (a * np.iinfo(dtype).max).astype(dtype)
    before = a.copy()
    for info in surface_registry.infos():
        result = surface_registry.create(info.id).process([a])
        assert result.primary.shape == a.shape[:2], info.id
        assert result.primary.dtype == np.float32, info.id
        assert np.isfinite(result.primary).all(), info.id
        assert result.diagnostics["input_dtype"] == str(a.dtype)
        if "enhanced" in result.intermediates:
            enhanced = result.intermediates["enhanced"]
            assert enhanced.dtype == np.float32 and enhanced.min() >= 0 and enhanced.max() <= 1
    np.testing.assert_array_equal(a, before)


def test_normalization_and_rejections():
    a = np.array([[0, 65535]], np.uint16)
    b, meta = normalize_input(a)
    np.testing.assert_allclose(b, [[0, 1]])
    assert meta["input_max"] == 65535
    np.testing.assert_allclose(normalize_input(np.array([[10.0, 20.0]]))[0], [[0, 1]])
    for a in [
        np.array([[np.nan]]),
        np.ones((3, 3, 4), np.uint8),
        np.ones((0, 3)),
        np.ones((3, 3), np.int16),
    ]:
        with pytest.raises(ValueError):
            normalize_input(a)


@pytest.mark.parametrize(
    "method,params",
    [
        ("surface_dog", {"sigma": 4, "large": 2}),
        ("surface_dog", {"sigma": float("nan")}),
        ("surface_top_hat", {"size": 4}),
        ("surface_gabor", {"orientations": 100}),
        ("surface_frangi", {"polarity": "both"}),
        ("surface_clahe", {"grid": 2.5}),
        ("surface_gamma", {"gamma": True}),
    ],
)
def test_invalid_parameters_rejected_before_processing(method, params):
    with pytest.raises(ValueError):
        surface_registry.create(method, **params)


def test_known_structures_polarity_and_constant():
    a = np.full((80, 80), 0.5, np.float32)
    a[30:34, 15:65] = 0.9
    a[48:52, 15:65] = 0.1
    bright = (
        surface_registry.create("surface_local_residual", sigma=4, polarity="bright")
        .process([a])
        .primary
    )
    dark = (
        surface_registry.create("surface_local_residual", sigma=4, polarity="dark")
        .process([a])
        .primary
    )
    assert bright[31, 40] > dark[31, 40]
    assert dark[49, 40] > bright[49, 40]
    for name in ["surface_sobel", "surface_dog", "surface_local_residual", "surface_local_std"]:
        result = surface_registry.create(name).process([np.full((32, 32), 0.5, np.float32)]).primary
        assert np.max(np.abs(result)) < 1e-3, name
    for name in ["surface_frangi", "surface_sato", "surface_meijering"]:
        b = surface_registry.create(name, polarity="bright").process([a]).primary
        d = surface_registry.create(name, polarity="dark").process([a]).primary
        assert b[31, 40] > d[31, 40], name
        assert d[49, 40] > b[49, 40], name


def test_signed_outputs_and_directional_gradient():
    a = np.full((64, 64), 0.5, np.float32)
    a[20:24, 12:52] = 0.9
    a[40:44, 12:52] = 0.1
    for name in [
        "local_residual",
        "median_residual",
        "dog",
        "log_response",
        "hessian",
        "multiscale_residual",
        "open_close_residual",
    ]:
        out = surface_registry.create("surface_" + name).process([a]).primary
        assert out.min() < -1e-5 and out.max() > 1e-5, name
    x = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    for name in ["sobel", "scharr"]:
        gx = surface_registry.create("surface_" + name, output="gradient_x").process([x]).primary
        gy = surface_registry.create("surface_" + name, output="gradient_y").process([x]).primary
        assert gx[30, 30] > 0.01 and abs(gy[30, 30]) < 1e-6


def test_spot_scratch_shading_glare_and_texture():
    import cv2

    y, x = np.mgrid[:96, :96]
    background = (0.2 + 0.5 * x / 95).astype(np.float32)
    image = background.copy()
    image[45:50, 45:50] += 0.22
    residual = surface_registry.create("surface_local_residual", sigma=5).process([image]).primary
    assert residual[47, 47] > abs(residual[20, 50]) * 5
    spot = surface_registry.create("surface_multiscale_top_hat", size=15).process([image]).primary
    assert spot[47, 47] > spot[20, 50] + 0.1
    scratch = np.full((96, 96), 0.7, np.float32)
    cv2.line(scratch, (15, 15), (80, 80), 0.15, 3)
    directional = (
        surface_registry.create("surface_directional_black_hat", size=21, angle=135)
        .process([scratch])
        .primary
    )
    assert directional[48, 48] > 0.3
    glare = np.clip(
        background + 0.35 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 500), 0, 1
    ).astype(np.float32)
    corrected = surface_registry.create("surface_shading", sigma=15).process([glare]).primary
    assert np.isfinite(corrected).all() and corrected.max() <= 1
    rng = np.random.default_rng(7)
    noisy = np.clip(0.5 + rng.normal(0, 0.06, (96, 96)), 0, 1).astype(np.float32)
    denoised = surface_registry.create("surface_nlmeans", amount=3).process([noisy]).primary
    assert np.std(denoised[10:-10, 10:-10]) < np.std(noisy[10:-10, 10:-10])
    texture = surface_registry.create("surface_local_std", size=9).process([noisy]).primary
    assert texture[20:76, 20:76].mean() > 0.03


def test_frangi_polarity_gate_and_frequency_validation():
    from skimage.filters import frangi

    a = np.full((80, 80), 0.5, np.float32)
    a[30:34, 15:65] = 0.9
    raw_wrong = frangi(a, sigmas=[2, 4], black_ridges=True)
    gated = surface_registry.create("surface_frangi", polarity="dark").process([a]).primary
    assert raw_wrong[31, 40] > 0 and gated[31, 40] == 0
    with pytest.raises(ValueError):
        surface_registry.create("surface_fft_bandpass", low_cutoff=0.2, high_cutoff=0.1)
