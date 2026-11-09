"""Shared Surface kernels. All inputs are float arrays; signed maps stay signed."""

from __future__ import annotations

import math

import cv2
import numpy as np


def to_u8(a):
    return np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)


def clahe_luminance(a, clip, grid):
    return (
        cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
        .apply(to_u8(a))
        .astype(np.float32)
        / 255
    )


def blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma)


def median_residual(a, size, *, float_precision=False):
    if float_precision:
        # scipy accepts float32 and arbitrary odd footprints; no signed-map re-normalization.
        from scipy.ndimage import median_filter

        if np.min(a) == np.max(a):
            return np.zeros_like(a, dtype=np.float32)
        background = median_filter(a, size=size, mode="nearest")
    else:
        # Matches existing Surface LAB exactly.
        background = cv2.medianBlur(to_u8(a), size).astype(np.float32) / 255
    return a - background


def local_residual(a, sigma):
    return a - blur(a, sigma)


def oriented_footprint(size, angle):
    footprint = np.zeros((size, size), np.uint8)
    center = size // 2
    distance = (size - 1) / 2
    radians = math.radians(angle)
    delta = np.rint([distance * math.cos(radians), distance * math.sin(radians)]).astype(int)
    cv2.line(
        footprint,
        (center - delta[0], center - delta[1]),
        (center + delta[0], center + delta[1]),
        1,
        1,
    )
    return footprint


def directional_morph(a, size, angle, operation):
    return cv2.morphologyEx(a, operation, oriented_footprint(size, angle))


def hessian_signed(a, sigma):
    b = blur(a, sigma)
    xx = cv2.Sobel(b, cv2.CV_32F, 2, 0)
    yy = cv2.Sobel(b, cv2.CV_32F, 0, 2)
    xy = cv2.Sobel(b, cv2.CV_32F, 1, 1)
    root = np.sqrt((xx - yy) ** 2 + 4 * xy**2)
    l1 = (xx + yy + root) / 2
    l2 = (xx + yy - root) / 2
    return -np.where(np.abs(l1) > np.abs(l2), l1, l2)


def fft_highpass(a, sigma):
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    mask = 1 - np.exp(-(frequency**2) * (2 * np.pi * sigma) ** 2 / 2)
    return np.fft.ifft2(np.fft.fft2(a) * mask).real


def unsharp(a, sigma, amount):
    return np.clip(unsharp_raw(a, sigma, amount), 0, 1)


def unsharp_raw(a, sigma, amount):
    return a + amount * (a - blur(a, sigma))
