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
