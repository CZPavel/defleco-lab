from __future__ import annotations

import cv2
import numpy as np

from ..base import MethodInfo, ProcessingMethod, ProcessingResult
from ..preprocessing import gray32, normalize_map


def _blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), float(sigma)) if float(sigma) > 0 else a


class Gradient(ProcessingMethod):
    info = MethodInfo(
        "gradient",
        "Sobel / Scharr Gradient",
        "Single frame",
        1,
        False,
        "Spatial intensity derivatives.",
        parameters={
            "operator": {"default": "scharr", "choices": ["sobel", "scharr"]},
            "sigma": {"default": 0.8, "minimum": 0.0, "maximum": 100.0, "step": 0.1, "units": "px"},
            "ksize": {"default": 3, "minimum": 1, "maximum": 31, "step": 2},
            "normalize": {
                "default": False,
                "tooltip": (
                    "Normalize the numeric gradient response to 0..1. Leave OFF for "
                    "quantitative comparison; the visualization layer can scale the display."
                ),
            },
        },
    )

    def _process(self, frames, **context):
        a = _blur(gray32(frames[-1]), self.parameters["sigma"])
        op = self.parameters["operator"]
        if op == "scharr":
            gx, gy = cv2.Scharr(a, cv2.CV_32F, 1, 0), cv2.Scharr(a, cv2.CV_32F, 0, 1)
        else:
            gx, gy = (
                cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=int(self.parameters["ksize"])),
                cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=int(self.parameters["ksize"])),
            )
        mag = cv2.magnitude(gx, gy)
        primary = normalize_map(mag) if self.parameters["normalize"] else mag
        if not context.get("keep_intermediates", True):
            return ProcessingResult(primary)
        return ProcessingResult(primary, {"gradient_x": gx, "gradient_y": gy, "magnitude": mag})


class Laplacian(ProcessingMethod):
    info = MethodInfo(
        "laplacian",
        "Laplacian",
        "Single frame",
        1,
        False,
        "Second spatial derivative.",
        parameters={
            "sigma": {"default": 0.8, "minimum": 0.0, "maximum": 100.0, "step": 0.1, "units": "px"},
            "ksize": {"default": 3, "minimum": 1, "maximum": 31, "step": 2},
        },
    )

    def _process(self, frames, **context):
        signed = cv2.Laplacian(
            _blur(gray32(frames[-1]), self.parameters["sigma"]),
            cv2.CV_32F,
            ksize=int(self.parameters["ksize"]),
        )
        absolute = np.abs(signed)
        if not context.get("keep_intermediates", True):
            return ProcessingResult(absolute)
        return ProcessingResult(absolute, {"signed": signed, "absolute": absolute})


class DifferenceOfGaussians(ProcessingMethod):
    info = MethodInfo(
        "dog",
        "Difference of Gaussians",
        "Single frame",
        1,
        False,
        "Band-pass residual.",
        parameters={
            "sigma_small": {
                "default": 1.0,
                "minimum": 0.01,
                "maximum": 100.0,
                "step": 0.1,
                "units": "px",
            },
            "sigma_large": {
                "default": 4.0,
                "minimum": 0.01,
                "maximum": 200.0,
                "step": 0.1,
                "units": "px",
            },
        },
    )

    def _process(self, frames, **context):
        a = gray32(frames[-1])
        signed = _blur(a, self.parameters["sigma_small"]) - _blur(a, self.parameters["sigma_large"])
        absolute = np.abs(signed)
        if not context.get("keep_intermediates", True):
            return ProcessingResult(absolute)
        return ProcessingResult(absolute, {"signed": signed, "absolute": absolute})


class LocalBackgroundResidual(ProcessingMethod):
    info = MethodInfo(
        "local_residual",
        "Local Background Residual",
        "Single frame",
        1,
        False,
        "Subtracts a smooth Gaussian background.",
        parameters={
            "sigma": {"default": 8.0, "minimum": 0.01, "maximum": 500.0, "step": 0.5, "units": "px"}
        },
    )

    def _process(self, frames, **context):
        a = gray32(frames[-1])
        bg = _blur(a, self.parameters["sigma"])
        signed = a - bg
        primary = np.abs(signed)
        if not context.get("keep_intermediates", True):
            return ProcessingResult(primary)
        return ProcessingResult(primary, {"background": bg, "signed": signed})


class StructureTensor(ProcessingMethod):
    info = MethodInfo(
        "structure_tensor",
        "Structure Tensor",
        "Single frame",
        1,
        False,
        "Local orientation and anisotropy using pi-periodic double-angle smoothing.",
        parameters={
            "operator": {"default": "scharr", "choices": ["sobel", "scharr"]},
            "derivative_sigma": {
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 0.1,
                "units": "px",
            },
            "tensor_sigma": {
                "default": 2.0,
                "minimum": 0.01,
                "maximum": 200.0,
                "step": 0.1,
                "units": "px",
            },
            "orientation_smooth_sigma": {
                "default": 8.0,
                "minimum": 0.01,
                "maximum": 500.0,
                "step": 0.5,
                "units": "px",
            },
            "output": {
                "default": "coherence",
                "choices": ["coherence", "anisotropy", "orientation_residual", "orientation"],
                "tooltip": (
                    "Choose the map shown as the primary processed result. "
                    "Orientation residual is often the most useful defect-enhancement view."
                ),
            },
        },
    )

    def _process(self, frames, **context):
        a = _blur(gray32(frames[-1]), self.parameters["derivative_sigma"])
        if self.parameters["operator"] == "scharr":
            gx, gy = cv2.Scharr(a, cv2.CV_32F, 1, 0), cv2.Scharr(a, cv2.CV_32F, 0, 1)
        else:
            gx, gy = (
                cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3),
            )
        s = float(self.parameters["tensor_sigma"])
        jxx, jyy, jxy = _blur(gx * gx, s), _blur(gy * gy, s), _blur(gx * gy, s)
        root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy).astype(np.float32)
        coherence = (root / (jxx + jyy + 1e-6)).astype(np.float32)

        output = self.parameters["output"]
        keep = context.get("keep_intermediates", True)
        need_orientation = keep or output in {"orientation_residual", "orientation"}

        theta = None
        orientation_residual = None
        if need_orientation:
            theta = (0.5 * np.arctan2(2 * jxy, jxx - jyy)).astype(np.float32)
            if keep or output == "orientation_residual":
                os = float(self.parameters["orientation_smooth_sigma"])
                cos2 = _blur(np.cos(2 * theta) * coherence, os)
                sin2 = _blur(np.sin(2 * theta) * coherence, os)
                smooth = 0.5 * np.arctan2(sin2, cos2)
                residual = 0.5 * np.arctan2(
                    np.sin(2 * (theta - smooth)), np.cos(2 * (theta - smooth))
                )
                orientation_residual = np.abs(residual).astype(np.float32)

        primary_maps = {
            "coherence": coherence,
            "anisotropy": root,
        }
        if theta is not None:
            primary_maps["orientation"] = theta
        if orientation_residual is not None:
            primary_maps["orientation_residual"] = orientation_residual
        primary = primary_maps[output]

        if not keep:
            return ProcessingResult(primary)

        l1 = ((jxx + jyy + root) / 2).astype(np.float32)
        l2 = ((jxx + jyy - root) / 2).astype(np.float32)
        assert theta is not None and orientation_residual is not None
        return ProcessingResult(
            primary,
            {
                "orientation": theta,
                "coherence": coherence,
                "lambda1": l1,
                "lambda2": l2,
                "anisotropy": root,
                "orientation_residual": orientation_residual,
            },
        )


class GaborBank(ProcessingMethod):
    info = MethodInfo(
        "gabor",
        "Gabor Filter Bank",
        "Single frame",
        1,
        False,
        "Oriented frequency-selective filter bank.",
        parameters={
            "periods": {"default": [12.0], "type": "list", "item_type": "float", "units": "px"},
            "orientations": {"default": 8, "minimum": 1, "maximum": 64},
            "sigma": {"default": 5.0, "minimum": 0.1, "maximum": 200.0, "step": 0.1, "units": "px"},
            "gamma": {"default": 0.5, "minimum": 0.01, "maximum": 10.0, "step": 0.05},
            "psi": {
                "default": 0.0,
                "minimum": -6.2832,
                "maximum": 6.2832,
                "step": 0.1,
                "decimals": 4,
                "units": "rad",
            },
            "scale": {"default": 1.0, "minimum": 0.1, "maximum": 1.0, "step": 0.1},
            "orientation_smooth_sigma": {
                "default": 8.0,
                "minimum": 0.1,
                "maximum": 500.0,
                "step": 0.5,
                "units": "px",
            },
            "output": {
                "default": "maximum_response",
                "choices": [
                    "maximum_response",
                    "orientation_residual",
                    "dominant_orientation",
                    "selected_frequency_response",
                ],
                "tooltip": (
                    "Choose the primary Gabor result. Orientation residual highlights "
                    "local departures from the smoothly varying dominant orientation."
                ),
            },
        },
    )

    def _process(self, frames, **context):
        src = gray32(frames[-1])
        scale = float(self.parameters["scale"])
        a = (
            cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale != 1
            else src
        )
        periods = self.parameters["periods"]
        periods = [periods] if np.isscalar(periods) else list(periods)
        n = int(self.parameters["orientations"])
        output = self.parameters["output"]
        keep = context.get("keep_intermediates", True)

        need_orientation = keep or output in {"orientation_residual", "dominant_orientation"}
        need_selected = keep or output == "selected_frequency_response"

        maximum = np.full(a.shape, -np.inf, np.float32)
        orientation = np.zeros(a.shape, np.float32) if need_orientation else None
        selected = np.full(a.shape, -np.inf, np.float32) if need_selected else None

        for period_index, p in enumerate(periods):
            for theta in np.linspace(0, np.pi, n, endpoint=False):
                k = max(7, round(float(self.parameters["sigma"]) * 6) | 1)
                common = (
                    (k, k),
                    float(self.parameters["sigma"]),
                    float(theta),
                    float(p) * scale,
                    float(self.parameters["gamma"]),
                )
                kr = cv2.getGaborKernel(*common, float(self.parameters["psi"]), ktype=cv2.CV_32F)
                ki = cv2.getGaborKernel(
                    *common, float(self.parameters["psi"]) + np.pi / 2, ktype=cv2.CV_32F
                )
                kr -= kr.mean()
                ki -= ki.mean()
                kr /= np.linalg.norm(kr) + 1e-9
                ki /= np.linalg.norm(ki) + 1e-9
                real = cv2.filter2D(a, cv2.CV_32F, kr)
                imag = cv2.filter2D(a, cv2.CV_32F, ki)
                response = cv2.magnitude(real, imag)

                stronger = response > maximum
                maximum[stronger] = response[stronger]
                if orientation is not None:
                    orientation[stronger] = float(theta)
                if selected is not None and period_index == 0:
                    np.maximum(selected, response, out=selected)

        orientation_residual = None
        if orientation is not None and (keep or output == "orientation_residual"):
            sigma = float(self.parameters["orientation_smooth_sigma"]) * scale
            sigma = max(0.1, sigma)
            smooth = 0.5 * np.arctan2(
                cv2.GaussianBlur(np.sin(2 * orientation), (0, 0), sigma),
                cv2.GaussianBlur(np.cos(2 * orientation), (0, 0), sigma),
            )
            residual = 0.5 * np.arctan2(
                np.sin(2 * (orientation - smooth)), np.cos(2 * (orientation - smooth))
            )
            orientation_residual = np.abs(residual).astype(np.float32)

        if scale != 1:
            size = (src.shape[1], src.shape[0])
            maximum = cv2.resize(maximum, size)
            if orientation is not None:
                orientation = cv2.resize(orientation, size, interpolation=cv2.INTER_NEAREST)
            if selected is not None:
                selected = cv2.resize(selected, size)
            if orientation_residual is not None:
                orientation_residual = cv2.resize(orientation_residual, size)

        maps = {"maximum_response": maximum}
        if orientation is not None:
            maps["dominant_orientation"] = orientation
        if selected is not None:
            maps["selected_frequency_response"] = selected
        if orientation_residual is not None:
            maps["orientation_residual"] = orientation_residual

        primary = maps[output]
        if not keep:
            return ProcessingResult(primary)
        return ProcessingResult(primary, maps)


class DirectionalResidual(ProcessingMethod):
    info = MethodInfo(
        "directional_residual",
        "Directional Residual",
        "Single frame",
        1,
        False,
        "Experimental orientation-binned line-background approximation.",
        limitations="Not a reproduction of a proprietary algorithm.",
        parameters={
            "orientations": {"default": 8, "minimum": 1, "maximum": 64},
            "length": {"default": 21, "minimum": 3, "maximum": 255, "step": 2, "units": "px"},
        },
    )

    def _process(self, frames, **_):
        a = gray32(frames[-1])
        best = np.full(a.shape, np.inf, np.float32)
        k = int(self.parameters["length"]) | 1
        for t in np.linspace(0, np.pi, int(self.parameters["orientations"]), endpoint=False):
            kernel = np.zeros((k, k), np.float32)
            c = k // 2
            d = (k // 2 - 1) * np.array([np.cos(t), np.sin(t)])
            cv2.line(
                kernel,
                tuple(np.rint([c - d[0], c - d[1]]).astype(int)),
                tuple(np.rint([c + d[0], c + d[1]]).astype(int)),
                1,
                1,
            )
            kernel /= kernel.sum()
            best = np.minimum(best, np.abs(a - cv2.filter2D(a, cv2.CV_32F, kernel)))
        return ProcessingResult(best, {"residual": best})
