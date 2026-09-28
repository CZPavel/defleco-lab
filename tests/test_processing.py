import cv2
import numpy as np
import pytest

from defleco_lab.processing.multi_frame import *
from defleco_lab.processing.single_frame import *


def image():
    y, x = np.mgrid[:96, :128]
    return (127 + 60 * np.sin(2 * np.pi * x / 12) + 30 * np.sin(2 * np.pi * y / 19)).astype(
        np.float32
    )


@pytest.mark.parametrize(
    "method",
    [
        Gradient(),
        Laplacian(),
        DifferenceOfGaussians(),
        LocalBackgroundResidual(),
        StructureTensor(),
        GaborBank(periods=[12], orientations=4),
        DirectionalResidual(length=11),
        FringeLineGeometry(min_length_px=8, max_segments=300),
    ],
)
def test_every_filter_returns_correct_size_finite_output(method):
    out = method.process([image()]).primary
    assert out.shape == image().shape and np.isfinite(out).all()


def test_structure_tensor_orientation_works_on_known_synthetic_lines():
    a = np.tile((np.sin(np.arange(128) * 2 * np.pi / 16) * 100 + 127).astype(np.float32), (96, 1))
    theta = StructureTensor().process([a]).intermediates["orientation"]
    assert np.median(np.abs(theta[10:-10, 10:-10])) < 0.12


def test_gabor_bank_responds_strongest_near_matching_orientation_frequency():
    a = np.tile((127 + 60 * np.sin(2 * np.pi * np.arange(128) / 12)).astype(np.float32), (96, 1))
    r = GaborBank(periods=[12, 30], orientations=8, sigma=5).process([a])
    orientation = r.intermediates["dominant_orientation"]
    assert (
        np.mean(
            np.minimum(abs(orientation[16:-16, 16:-16]), abs(orientation[16:-16, 16:-16] - np.pi))
        )
        < 0.25
    )


def test_farneback_returns_finite_two_channel_flow():
    a = image()
    b = cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, 0]]), (128, 96))
    flow = FarnebackFlow().process([a, b]).intermediates["flow"]
    assert flow.shape == (96, 128, 2) and np.isfinite(flow).all()


def test_temporal_statistics_match_known_synthetic_sequence():
    frames = [np.full((8, 9), x, np.float32) for x in range(4)]
    r = TemporalStatistics(window=4).process(frames)
    assert np.allclose(r.intermediates["mean"], 1.5) and np.allclose(r.intermediates["range"], 3)


def test_unknown_method_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        StructureTensor(not_a_parameter=123)


def test_gradient_default_keeps_raw_numerical_response():
    a = image()
    result = Gradient().process([a], keep_intermediates=False).primary
    assert result.max() > 1.0
    assert np.isfinite(result).all()


def test_structure_tensor_can_return_orientation_residual_as_primary():
    a = image()
    result = StructureTensor(output="orientation_residual").process(
        [a], keep_intermediates=False
    )
    assert result.primary.shape == a.shape
    assert np.isfinite(result.primary).all()
    assert not result.intermediates


def test_gabor_can_return_orientation_residual_as_primary():
    a = image()
    result = GaborBank(
        periods=[12], orientations=4, sigma=5, output="orientation_residual"
    ).process([a], keep_intermediates=False)
    assert result.primary.shape == a.shape
    assert np.isfinite(result.primary).all()
    assert not result.intermediates



def test_structure_tensor_line_suppression_prefers_mixed_directions() -> None:
    size = 128
    stripe = np.zeros((size, size), np.float32)
    stripe[:, size // 2 :] = 255.0

    cross = stripe.copy()
    cross[size // 2 :, :] += 255.0
    cross = np.clip(cross, 0, 255)

    line_response = StructureTensor(
        tensor_sigma=3.0, output="linearity_suppressed"
    ).process([stripe], keep_intermediates=False).primary
    cross_response = StructureTensor(
        tensor_sigma=3.0, output="linearity_suppressed"
    ).process([cross], keep_intermediates=False).primary

    center = np.s_[size // 2 - 5 : size // 2 + 6, size // 2 - 5 : size // 2 + 6]
    assert float(cross_response[center].mean()) > float(line_response[center].mean()) + 1.0


def test_structure_tensor_junction_response_is_finite() -> None:
    a = image()
    result = StructureTensor(output="junction_response").process(
        [a], keep_intermediates=False
    )
    assert result.primary.shape == a.shape
    assert np.isfinite(result.primary).all()



def test_fringe_line_geometry_extracts_explicit_vector_segments() -> None:
    a = np.zeros((160, 220), np.float32)
    for x in range(20, 210, 30):
        cv2.line(a, (x, 10), (x, 150), 255, 3)

    result = FringeLineGeometry(
        min_length_px=20,
        neighbor_radius_px=70,
        max_segments=500,
        output="vector_lines",
    ).process([a], keep_intermediates=True)

    assert result.diagnostics["line_count"] > 0
    assert np.count_nonzero(result.primary) > 0
    assert "geometry_residual" in result.intermediates
