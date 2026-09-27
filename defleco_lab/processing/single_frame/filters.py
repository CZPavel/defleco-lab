from __future__ import annotations

import cv2
import numpy as np

from ..base import MethodInfo, ProcessingMethod, ProcessingResult
from ..preprocessing import gray32


def _blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), float(sigma)) if float(sigma) > 0 else a


class Gradient(ProcessingMethod):
    info = MethodInfo(
        "gradient",
        "Sobel / Scharr Gradient",
        "Single frame",
        1,
        False,
        (
            "Spatial intensity derivatives. Use magnitude as the visual baseline, "
            "or vector_residual to suppress slowly varying predictable gradient structure."
        ),
        recommended_use=(
            "Fine reflected stripes/checker/spiral. Start with Scharr magnitude; "
            "then compare vector_residual for local surface disturbances."
        ),
        parameters={
            "operator": {
                "default": "scharr",
                "choices": ["sobel", "scharr"],
                "tooltip": "Scharr is a good default for small rotationally balanced derivatives.",
            },
            "sigma": {
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 0.1,
                "units": "px",
                "tooltip": "Input smoothing before the derivative. Keep small for fine patterns.",
            },
            "ksize": {
                "default": 3,
                "minimum": 1,
                "maximum": 31,
                "step": 2,
                "tooltip": "Sobel kernel size; ignored when Scharr is selected.",
                "visible_if": {"operator": "sobel"},
            },
            "residual_sigma": {
                "default": 8.0,
                "minimum": 0.1,
                "maximum": 500.0,
                "step": 0.5,
                "units": "px",
                "tooltip": (
                    "Local scale used to predict the smooth gradient field. "
                    "The vector residual keeps deviations from that prediction."
                ),
                "visible_if": {"output": "vector_residual"},
            },
            "output": {
                "default": "magnitude",
                "choices": [
                    "magnitude",
                    "gradient_x",
                    "gradient_y",
                    "orientation",
                    "vector_residual",
                ],
                "tooltip": (
                    "Magnitude is the raw edge-strength baseline. Vector residual subtracts "
                    "a locally smooth Gx/Gy field and is intended to suppress predictable lines."
                ),
            },
        },
    )

    def _process(self, frames, **context):
        a = _blur(gray32(frames[-1]), self.parameters["sigma"])
        op = self.parameters["operator"]
        if op == "scharr":
            gx = cv2.Scharr(a, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(a, cv2.CV_32F, 0, 1)
        else:
            ksize = int(self.parameters["ksize"])
            gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=ksize)

        output = self.parameters["output"]
        keep = context.get("keep_intermediates", True)
        mag = cv2.magnitude(gx, gy)
        orientation = None
        residual = None
        if keep or output == "orientation":
            orientation = np.arctan2(gy, gx).astype(np.float32)
        if keep or output == "vector_residual":
            sigma = float(self.parameters["residual_sigma"])
            smooth_x = _blur(gx, sigma)
            smooth_y = _blur(gy, sigma)
            residual = cv2.magnitude(gx - smooth_x, gy - smooth_y)

        primary = {
            "magnitude": mag,
            "gradient_x": gx,
            "gradient_y": gy,
            "orientation": orientation,
            "vector_residual": residual,
        }[output]
        assert primary is not None

        if not keep:
            return ProcessingResult(primary)
        assert orientation is not None and residual is not None
        return ProcessingResult(
            primary,
            {
                "magnitude": mag,
                "gradient_x": gx,
                "gradient_y": gy,
                "orientation": orientation,
                "vector_residual": residual,
            },
        )


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
        (
            "Local fringe orientation and two-dimensional structure from the image gradient. "
            "The additional line-suppressed outputs are intended to reduce ordinary long lines "
            "and emphasize locally mixed/crossing structure."
        ),
        recommended_use=(
            "Static reflected stripes/fringes: orientation_residual. "
            "Checkerboard or locally tangled lines: compare linearity_suppressed and junction_response."
        ),
        parameters={
            "operator": {
                "default": "scharr",
                "choices": ["sobel", "scharr"],
                "tooltip": "Derivative operator used to build the local tensor.",
            },
            "derivative_sigma": {
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 0.1,
                "units": "px",
                "tooltip": "Small pre-smoothing before gradient calculation.",
            },
            "tensor_sigma": {
                "default": 2.0,
                "minimum": 0.01,
                "maximum": 200.0,
                "step": 0.1,
                "units": "px",
                "tooltip": "Local neighbourhood over which gradient directions are combined.",
            },
            "orientation_smooth_sigma": {
                "default": 8.0,
                "minimum": 0.01,
                "maximum": 500.0,
                "step": 0.5,
                "units": "px",
                "tooltip": (
                    "Expected smooth orientation scale. Orientation residual highlights "
                    "local deviations from this slowly varying field."
                ),
                "visible_if": {"output": "orientation_residual"},
            },
            "output": {
                "default": "orientation_residual",
                "choices": [
                    "orientation_residual",
                    "linearity_suppressed",
                    "junction_response",
                    "coherence",
                    "anisotropy",
                    "orientation",
                ],
                "tooltip": (
                    "orientation_residual highlights local bending; linearity_suppressed "
                    "and junction_response reduce simple one-direction line responses."
                ),
            },
        },
    )

    def _process(self, frames, **context):
        a = _blur(gray32(frames[-1]), self.parameters["derivative_sigma"])
        if self.parameters["operator"] == "scharr":
            gx = cv2.Scharr(a, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(a, cv2.CV_32F, 0, 1)
        else:
            gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)

        sigma = float(self.parameters["tensor_sigma"])
        jxx = _blur(gx * gx, sigma)
        jyy = _blur(gy * gy, sigma)
        jxy = _blur(gx * gy, sigma)
        energy = (jxx + jyy).astype(np.float32)
        root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy).astype(np.float32)
        coherence = (root / (energy + 1e-6)).astype(np.float32)

        output = self.parameters["output"]
        keep = context.get("keep_intermediates", True)
        need_orientation = output in {"orientation", "orientation_residual"} or keep
        need_eigen = output in {"linearity_suppressed", "junction_response"} or keep

        theta = None
        residual = None
        if need_orientation:
            theta = (0.5 * np.arctan2(2 * jxy, jxx - jyy)).astype(np.float32)
            if output == "orientation_residual" or keep:
                smooth_sigma = float(self.parameters["orientation_smooth_sigma"])
                cos2 = _blur(np.cos(2 * theta) * coherence, smooth_sigma)
                sin2 = _blur(np.sin(2 * theta) * coherence, smooth_sigma)
                smooth = 0.5 * np.arctan2(sin2, cos2)
                residual = np.abs(
                    0.5
                    * np.arctan2(
                        np.sin(2 * (theta - smooth)),
                        np.cos(2 * (theta - smooth)),
                    )
                ).astype(np.float32)

        l1 = None
        l2 = None
        linearity_suppressed = None
        junction = None
        if need_eigen:
            l1 = ((energy + root) / 2).astype(np.float32)
            l2 = ((energy - root) / 2).astype(np.float32)
            # A pure line has a very small second eigenvalue.  Mixed directions,
            # corners and local tangling make both eigenvalues significant.
            linearity_suppressed = np.maximum(l2, 0).astype(np.float32)
            junction = (
                np.maximum(l1, 0) * np.maximum(l2, 0) / (l1 + l2 + 1e-6)
            ).astype(np.float32)

        primary_map = {
            "coherence": coherence,
            "anisotropy": root,
            "orientation": theta,
            "orientation_residual": residual,
            "linearity_suppressed": linearity_suppressed,
            "junction_response": junction,
        }[output]
        assert primary_map is not None

        if not keep:
            return ProcessingResult(primary_map)

        assert theta is not None and residual is not None
        assert l1 is not None and l2 is not None
        assert linearity_suppressed is not None and junction is not None
        return ProcessingResult(
            primary_map,
            {
                "orientation_residual": residual,
                "linearity_suppressed": linearity_suppressed,
                "junction_response": junction,
                "coherence": coherence,
                "anisotropy": root,
                "orientation": theta,
                "lambda1": l1,
                "lambda2": l2,
            },
        )


class GaborBank(ProcessingMethod):
    info = MethodInfo(
        "gabor",
        "Gabor Filter Bank",
        "Single frame",
        1,
        False,
        "Oriented frequency-selective filter bank for reflected fringe structures.",
        recommended_use="Static stripes/fringes; compare maximum_response and orientation_residual.",
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
            "scale": {
                "default": 1.0,
                "minimum": 0.1,
                "maximum": 1.0,
                "step": 0.1,
                "tooltip": "Optional additional internal Gabor downscale. Prefer the global Processing scale first.",
            },
            "output": {
                "default": "maximum_response",
                "choices": [
                    "maximum_response",
                    "orientation_residual",
                    "dominant_orientation",
                    "selected_frequency_response",
                ],
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

        sigma = max(0.1, float(self.parameters["sigma"]) * scale)
        kernel_size = max(7, round(sigma * 6) | 1)
        for period_index, period in enumerate(periods):
            wavelength = max(1.0, float(period) * scale)
            for theta in np.linspace(0, np.pi, n, endpoint=False):
                common = (
                    (kernel_size, kernel_size),
                    sigma,
                    float(theta),
                    wavelength,
                    float(self.parameters["gamma"]),
                )
                kr = cv2.getGaborKernel(
                    *common, float(self.parameters["psi"]), ktype=cv2.CV_32F
                )
                ki = cv2.getGaborKernel(
                    *common,
                    float(self.parameters["psi"]) + np.pi / 2,
                    ktype=cv2.CV_32F,
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

        residual = None
        if orientation is not None:
            smooth_sigma = max(0.1, 8.0 * scale)
            smooth = 0.5 * np.arctan2(
                cv2.GaussianBlur(np.sin(2 * orientation), (0, 0), smooth_sigma),
                cv2.GaussianBlur(np.cos(2 * orientation), (0, 0), smooth_sigma),
            )
            residual = np.abs(
                0.5
                * np.arctan2(
                    np.sin(2 * (orientation - smooth)),
                    np.cos(2 * (orientation - smooth)),
                )
            ).astype(np.float32)

        if scale != 1:
            size = (src.shape[1], src.shape[0])
            maximum = cv2.resize(maximum, size)
            if orientation is not None:
                orientation = cv2.resize(orientation, size, interpolation=cv2.INTER_NEAREST)
            if selected is not None:
                selected = cv2.resize(selected, size)
            if residual is not None:
                residual = cv2.resize(residual, size)

        outputs = {
            "maximum_response": maximum,
            "orientation_residual": residual,
            "dominant_orientation": orientation,
            "selected_frequency_response": selected,
        }
        primary = outputs[output]
        assert primary is not None
        if not keep:
            return ProcessingResult(primary)

        assert orientation is not None and selected is not None and residual is not None
        return ProcessingResult(
            primary,
            {
                "maximum_response": maximum,
                "orientation_residual": residual,
                "dominant_orientation": orientation,
                "selected_frequency_response": selected,
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
