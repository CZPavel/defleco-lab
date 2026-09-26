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
        parameters={"operator": "scharr", "sigma": 0.8, "ksize": 3, "normalize": True},
    )

    def _process(self, frames, **_):
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
        return ProcessingResult(primary, {"gradient_x": gx, "gradient_y": gy, "magnitude": mag})


class Laplacian(ProcessingMethod):
    info = MethodInfo(
        "laplacian",
        "Laplacian",
        "Single frame",
        1,
        False,
        "Second spatial derivative.",
        parameters={"sigma": 0.8, "ksize": 3},
    )

    def _process(self, frames, **_):
        signed = cv2.Laplacian(
            _blur(gray32(frames[-1]), self.parameters["sigma"]),
            cv2.CV_32F,
            ksize=int(self.parameters["ksize"]),
        )
        absolute = np.abs(signed)
        return ProcessingResult(absolute, {"signed": signed, "absolute": absolute})


class DifferenceOfGaussians(ProcessingMethod):
    info = MethodInfo(
        "dog",
        "Difference of Gaussians",
        "Single frame",
        1,
        False,
        "Band-pass residual.",
        parameters={"sigma_small": 1.0, "sigma_large": 4.0},
    )

    def _process(self, frames, **_):
        a = gray32(frames[-1])
        signed = _blur(a, self.parameters["sigma_small"]) - _blur(a, self.parameters["sigma_large"])
        return ProcessingResult(np.abs(signed), {"signed": signed, "absolute": np.abs(signed)})


class LocalBackgroundResidual(ProcessingMethod):
    info = MethodInfo(
        "local_residual",
        "Local Background Residual",
        "Single frame",
        1,
        False,
        "Subtracts a smooth Gaussian background.",
        parameters={"sigma": 8.0},
    )

    def _process(self, frames, **_):
        a = gray32(frames[-1])
        bg = _blur(a, self.parameters["sigma"])
        signed = a - bg
        return ProcessingResult(np.abs(signed), {"background": bg, "signed": signed})


class StructureTensor(ProcessingMethod):
    info = MethodInfo(
        "structure_tensor",
        "Structure Tensor",
        "Single frame",
        1,
        False,
        "Local orientation and anisotropy using pi-periodic double-angle smoothing.",
        parameters={
            "operator": "scharr",
            "derivative_sigma": 0.8,
            "tensor_sigma": 2.0,
            "orientation_smooth_sigma": 8.0,
        },
    )

    def _process(self, frames, **_):
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
        root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
        l1 = (jxx + jyy + root) / 2
        l2 = (jxx + jyy - root) / 2
        theta = (0.5 * np.arctan2(2 * jxy, jxx - jyy)).astype(np.float32)
        coherence = (root / (jxx + jyy + 1e-6)).astype(np.float32)
        os = float(self.parameters["orientation_smooth_sigma"])
        c = _blur(np.cos(2 * theta) * coherence, os)
        ss = _blur(np.sin(2 * theta) * coherence, os)
        smooth = 0.5 * np.arctan2(ss, c)
        residual = 0.5 * np.arctan2(np.sin(2 * (theta - smooth)), np.cos(2 * (theta - smooth)))
        return ProcessingResult(
            coherence,
            {
                "orientation": theta,
                "coherence": coherence,
                "lambda1": l1,
                "lambda2": l2,
                "anisotropy": root,
                "orientation_residual": np.abs(residual),
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
            "periods": [12.0],
            "orientations": 8,
            "sigma": 5.0,
            "gamma": 0.5,
            "psi": 0.0,
            "scale": 1.0,
        },
    )

    def _process(self, frames, **_):
        src = gray32(frames[-1])
        scale = float(self.parameters["scale"])
        a = (
            cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale != 1
            else src
        )
        periods = self.parameters["periods"]
        periods = [periods] if np.isscalar(periods) else periods
        n = int(self.parameters["orientations"])
        responses = []
        angles = []
        period_ids = []
        for p in periods:
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
                responses.append(cv2.magnitude(real, imag))
                angles.append(theta)
                period_ids.append(float(p))
        stack = np.stack(responses)
        idx = np.argmax(stack, axis=0)
        maximum = np.max(stack, axis=0)
        orientation = np.take(np.asarray(angles, np.float32), idx)
        if scale != 1:
            maximum = cv2.resize(maximum, (src.shape[1], src.shape[0]))
            orientation = cv2.resize(
                orientation, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        selected = np.max(stack[np.asarray(period_ids) == float(periods[0])], axis=0)
        smooth = 0.5 * np.arctan2(
            cv2.GaussianBlur(np.sin(2 * orientation), (0, 0), 8),
            cv2.GaussianBlur(np.cos(2 * orientation), (0, 0), 8),
        )
        residual = 0.5 * np.arctan2(
            np.sin(2 * (orientation - smooth)), np.cos(2 * (orientation - smooth))
        )
        if scale != 1:
            selected = cv2.resize(selected, (src.shape[1], src.shape[0]))
            residual = cv2.resize(residual, (src.shape[1], src.shape[0]))
        return ProcessingResult(
            maximum,
            {
                "maximum_response": maximum,
                "dominant_orientation": orientation,
                "selected_frequency_response": selected,
                "orientation_residual": np.abs(residual),
            },
        )


class DirectionalResidual(ProcessingMethod):
    info = MethodInfo(
        "directional_residual",
        "Directional Residual",
        "Single frame",
        1,
        False,
        "Experimental orientation-binned line-background approximation.",
        limitations="Not a reproduction of a proprietary algorithm.",
        parameters={"orientations": 8, "length": 21},
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
