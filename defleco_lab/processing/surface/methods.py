"""Single-frame surface exploration methods; responses retain their native scale."""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage import feature, filters
from skimage.filters import rank
from skimage.morphology import disk

from ..base import MethodInfo, ProcessingMethod, ProcessingResult
from ..registry import MethodRegistry
from ..single_frame.filters import (
    DifferenceOfGaussians,
    DirectionalResidual,
    GaborBank,
    Gradient,
    Laplacian,
    LocalBackgroundResidual,
    StructureTensor,
)

surface_registry = MethodRegistry()


def normalize_input(image):
    """Normalize unsigned integer sensor range or finite float image to [0, 1]."""
    if not isinstance(image, np.ndarray) or not image.size or image.ndim not in (2, 3):
        raise ValueError("Expected a nonempty grayscale or BGR image")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Expected exactly three BGR channels")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError("Expected uint8, uint16, float32 or float64")
    if not np.isfinite(image).all():
        raise ValueError("Input must be finite")
    lo, hi = float(image.min()), float(image.max())
    if image.dtype in (np.uint8, np.uint16):
        scale = float(np.iinfo(image.dtype).max)
        result = image.astype(np.float32) / scale
        method = f"dtype range / {int(scale)}"
    elif 0 <= lo and hi <= 1:
        result = image.astype(np.float32, copy=True)
        method = "float [0,1] unchanged"
    elif hi > lo:
        result = ((image.astype(np.float64) - lo) / (hi - lo)).astype(np.float32)
        method = "float image min-max"
    else:
        result = np.full(image.shape, np.clip(lo, 0, 1), dtype=np.float32)
        method = "constant float clipped"
    return result, {
        "input_dtype": str(image.dtype),
        "input_min": lo,
        "input_max": hi,
        "normalization": method,
    }


def _gray(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a


def _blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma)


def _u8(a):
    return np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)


def _param(default, minimum=None, maximum=None, choices=None):
    d = {"default": default}
    if minimum is not None:
        d.update(minimum=minimum, maximum=maximum)
    if choices is not None:
        d["choices"] = choices
    return d


P = {
    "sigma": _param(2.0, 0.2, 20.0),
    "large": _param(8.0, 0.4, 40.0),
    "amount": _param(1.0, 0.0, 4.0),
    "gamma": _param(1.0, 0.1, 5.0),
    "clip": _param(2.0, 0.1, 10.0),
    "grid": _param(8, 2, 32),
    "size": _param(15, 3, 63),
    "angle": _param(0.0, 0.0, 179.0),
    "polarity": _param("signed", choices=["signed", "absolute", "bright", "dark"]),
    "ridge_polarity": _param("absolute", choices=["absolute", "bright", "dark"]),
    "output": _param("magnitude", choices=["magnitude", "gradient_x", "gradient_y"]),
    "low_cutoff": _param(0.02, 0.001, 0.49),
    "high_cutoff": _param(0.15, 0.002, 0.5),
    "offset": _param(0.0, -1.0, 1.0),
    "threshold": _param(0.15, 0.01, 0.9),
    "period": _param(12.0, 3.0, 64.0),
    "orientations": _param(8, 2, 16),
    "radius": _param(4, 1, 15),
}

# id, display name, practical use, parameters
SPECS = [
    ("brightness", "Brightness / gain", "Linear brightness and offset adjustment", "amount offset"),
    ("gamma", "Gamma", "Nonlinear tonal adjustment", "gamma"),
    ("clahe", "CLAHE", "Local grayscale contrast", "clip grid"),
    ("equalize", "Histogram equalization", "Global grayscale contrast", ""),
    ("stretch", "Percentile stretch", "Suppress extreme intensity outliers", ""),
    ("log", "Log compression", "Compress highlights and lift dark areas", "amount"),
    ("lab_clahe", "LAB luminance CLAHE", "Color-preserving local luminance contrast", "clip grid"),
    ("gaussian", "Gaussian denoise", "Suppress fine random noise", "sigma"),
    ("median", "Median denoise", "Suppress impulse noise", "size"),
    ("bilateral", "Bilateral denoise", "Smooth while preserving stronger edges", "sigma amount"),
    ("nlmeans", "Non-local means", "Suppress repeated-pattern and random noise", "amount"),
    ("unsharp", "Unsharp mask", "Enhance fine detail", "sigma amount"),
    (
        "local_residual",
        "Local background residual",
        "Deviation from Gaussian background",
        "sigma polarity",
    ),
    ("median_residual", "Median residual", "Deviation from local median", "size polarity"),
    (
        "dog",
        "Difference of Gaussians",
        "Band-pass structures between two scales",
        "sigma large polarity",
    ),
    (
        "log_response",
        "Laplacian of Gaussian",
        "Second derivative at selected scale",
        "sigma polarity",
    ),
    ("shading", "Shading correction", "Divide by slowly varying illumination", "sigma"),
    ("retinex", "Single-scale Retinex", "Log-domain illumination suppression", "sigma"),
    (
        "multiscale_retinex",
        "Multiscale Retinex",
        "Log residual averaged over three scales",
        "sigma",
    ),
    ("homomorphic", "Homomorphic high-pass", "Frequency-domain illumination suppression", "sigma"),
    ("top_hat", "Top-hat", "Bright features smaller than footprint", "size"),
    ("black_hat", "Black-hat", "Dark features smaller than footprint", "size"),
    ("morph_gradient", "Morphological gradient", "Local bright-to-dark range", "size"),
    (
        "open_close_residual",
        "Open-close residual",
        "Signed deviation from morphological background",
        "size polarity",
    ),
    (
        "multiscale_top_hat",
        "Multiscale top-hat",
        "Bright features over three footprint scales",
        "size",
    ),
    (
        "multiscale_black_hat",
        "Multiscale black-hat",
        "Dark features over three footprint scales",
        "size",
    ),
    (
        "directional_top_hat",
        "Directional top-hat",
        "Bright elongated features at chosen angle",
        "size angle",
    ),
    (
        "directional_black_hat",
        "Directional black-hat",
        "Dark elongated features at chosen angle",
        "size angle",
    ),
    (
        "multiscale_residual",
        "Multiscale residual",
        "Maximum deviation across three scales",
        "sigma polarity",
    ),
    ("sobel", "Sobel gradient", "First derivative edge strength or signed axis", "sigma output"),
    (
        "scharr",
        "Scharr gradient",
        "Rotationally balanced edge strength or signed axis",
        "sigma output",
    ),
    ("laplacian", "Laplacian magnitude", "Second derivative edge response", "sigma"),
    ("canny", "Canny edges", "Thresholded thin edges", "sigma threshold"),
    (
        "steerable",
        "Directional derivative",
        "Gradient response at selected angle",
        "sigma angle polarity",
    ),
    ("hessian", "Hessian ridge", "Largest curvature with polarity", "sigma polarity"),
    ("frangi", "Frangi", "Multiscale vesselness/ridge response", "sigma polarity"),
    ("sato", "Sato", "Multiscale tubeness/ridge response", "sigma polarity"),
    ("meijering", "Meijering", "Multiscale neuriteness/ridge response", "sigma polarity"),
    ("gabor", "Gabor bank", "Oriented frequency-selective response", "sigma period orientations"),
    ("tensor", "Structure tensor", "Local orientation irregularity", "sigma"),
    (
        "vector_residual",
        "Gradient vector residual",
        "Suppress smoothly varying gradient field",
        "sigma",
    ),
    (
        "directional_residual",
        "Directional residual",
        "Deviation from best oriented local line",
        "size orientations",
    ),
    ("fft_highpass", "FFT high-pass", "Suppress low spatial frequencies", "sigma"),
    (
        "fft_bandpass",
        "FFT band-pass",
        "Select spatial frequency band in cycles per pixel",
        "low_cutoff high_cutoff",
    ),
    ("entropy", "Local entropy", "Local intensity complexity", "radius"),
    ("lbp", "Local binary pattern", "Eight-neighbor texture code", "radius"),
    ("local_std", "Local standard deviation", "Local intensity variation", "size"),
    ("local_contrast", "Local contrast", "Local intensity range", "size"),
    (
        "preset_scratches",
        "Scratch preset",
        "CLAHE followed by multiscale dark ridge",
        "clip grid sigma",
    ),
    ("preset_defects", "Defect preset", "CLAHE followed by local residual", "clip grid sigma"),
]


def _pol(a, mode):
    return (
        a
        if mode == "signed"
        else (np.abs(a) if mode == "absolute" else np.maximum(a if mode == "bright" else -a, 0))
    )


def _foot(size, angle):
    f = np.zeros((size, size), np.uint8)
    c = size // 2
    d = (size - 1) / 2
    t = math.radians(angle)
    v = np.rint([d * math.cos(t), d * math.sin(t)]).astype(int)
    cv2.line(f, (c - v[0], c - v[1]), (c + v[0], c + v[1]), 1, 1)
    return f


def _ridge(name, g, sigma, polarity):
    fn = getattr(filters, name)
    sigmas = [sigma, sigma * 2]

    def one(black):
        response = fn(g, sigmas=sigmas, black_ridges=black)
        # Frangi 0.26 can respond to opposite-polarity ideal lines when the
        # smaller eigenvalue is zero. Gate using Sato's signed ridge response.
        if name == "frangi":
            response = np.where(filters.sato(g, sigmas=sigmas, black_ridges=black) > 0, response, 0)
        return response

    if polarity == "absolute":
        return np.maximum(one(True), one(False))
    return one(polarity == "dark")


class _SurfaceMethod(ProcessingMethod):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        for key, value in self.parameters.items():
            spec = self.info.parameters[key]
            if "choices" in spec:
                if value not in spec["choices"]:
                    raise ValueError(f"{key}: invalid choice")
            else:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                ):
                    raise ValueError(f"{key}: expected finite number")
                if not spec["minimum"] <= value <= spec["maximum"]:
                    raise ValueError(f"{key}: out of range")
                if key in {"size", "grid", "orientations", "radius"} and int(value) != value:
                    raise ValueError(f"{key}: expected integer")
                if key == "size" and int(value) % 2 != 1:
                    raise ValueError("size must be odd")
        if (
            "low_cutoff" in self.parameters
            and self.parameters["low_cutoff"] >= self.parameters["high_cutoff"]
        ):
            raise ValueError("low_cutoff must be smaller than high_cutoff")
        if "large" in self.parameters and self.parameters["large"] <= self.parameters["sigma"]:
            raise ValueError("large must exceed sigma")

    def _process(self, frames, **context):
        a, metadata = normalize_input(frames[-1])
        g = _gray(a)
        p = self.parameters
        name = self.info.id.removeprefix("surface_")
        sigma = float(p.get("sigma", 2))
        polarity = p.get("polarity", p.get("ridge_polarity", "absolute"))
        size = int(p.get("size", 15))
        enhanced = None
        clahe = lambda x: (
            cv2.createCLAHE(
                clipLimit=float(p.get("clip", 2)), tileGridSize=(int(p.get("grid", 8)),) * 2
            )
            .apply(_u8(x))
            .astype(np.float32)
            / 255
        )
        if name == "brightness":
            enhanced = np.clip(a * p["amount"] + p["offset"], 0, 1)
        elif name == "gamma":
            enhanced = np.power(a, p["gamma"])
        elif name == "clahe":
            enhanced = clahe(g)
        elif name == "equalize":
            enhanced = cv2.equalizeHist(_u8(g)).astype(np.float32) / 255
        elif name == "stretch":
            lo, hi = np.percentile(g, [1, 99])
            enhanced = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
        elif name == "log":
            enhanced = (
                np.log1p(a * p["amount"]) / max(np.log1p(p["amount"]), 1e-6)
                if p["amount"]
                else a.copy()
            )
        elif name == "lab_clahe":
            if a.ndim == 3:
                lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe(lab[:, :, 0] / 100) * 100
                enhanced = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)
            else:
                enhanced = clahe(g)
        elif name == "gaussian":
            enhanced = _blur(a, sigma)
        elif name == "median":
            enhanced = cv2.medianBlur(_u8(a), size).astype(np.float32) / 255
        elif name == "bilateral":
            enhanced = cv2.bilateralFilter(a, 0, max(0.01, p["amount"] * 0.1), sigma)
        elif name == "nlmeans":
            source = _u8(a)
            enhanced = (
                cv2.fastNlMeansDenoisingColored(
                    source, None, p["amount"] * 5, p["amount"] * 5, 7, 21
                )
                if a.ndim == 3
                else cv2.fastNlMeansDenoising(source, None, p["amount"] * 5, 7, 21)
            ).astype(np.float32) / 255
        elif name == "unsharp":
            blur = _blur(a, sigma)
            enhanced = np.clip(a + p["amount"] * (a - blur), 0, 1)
        elif name == "local_residual":
            signed = g - _blur(g, sigma)
            out = _pol(signed, polarity)
            # Existing filter supplies the same absolute response.
            if polarity == "absolute":
                out = (
                    LocalBackgroundResidual(sigma=sigma)
                    .process([g], keep_intermediates=False)
                    .primary
                )
        elif name == "median_residual":
            background = cv2.medianBlur(_u8(g), size).astype(np.float32) / 255
            out = _pol(g - background, polarity)
        elif name == "dog":
            signed = _blur(g, sigma) - _blur(g, float(p["large"]))
            out = _pol(signed, polarity)
            if polarity == "absolute":
                out = (
                    DifferenceOfGaussians(sigma_small=sigma, sigma_large=p["large"])
                    .process([g], keep_intermediates=False)
                    .primary
                )
        elif name == "log_response":
            out = _pol(-cv2.Laplacian(_blur(g, sigma), cv2.CV_32F, ksize=3), polarity)
        elif name == "shading":
            enhanced = np.clip(g / np.maximum(_blur(g, sigma), 0.01) * float(np.mean(g)), 0, 1)
        elif name in ("retinex", "multiscale_retinex"):
            scales = (sigma,) if name == "retinex" else (sigma, sigma * 2, sigma * 4)
            out = sum(np.log(g + 0.01) - np.log(_blur(g, s) + 0.01) for s in scales) / len(scales)
        elif name in ("homomorphic", "fft_highpass", "fft_bandpass"):
            src = np.log(g + 0.01) if name == "homomorphic" else g
            h, w = g.shape
            fy = np.fft.fftfreq(h)[:, None]
            fx = np.fft.fftfreq(w)[None, :]
            frequency = np.sqrt(fx * fx + fy * fy)
            mask = (
                ((frequency >= p["low_cutoff"]) & (frequency <= p["high_cutoff"]))
                if name == "fft_bandpass"
                else 1 - np.exp(-(frequency**2) * (2 * np.pi * sigma) ** 2 / 2)
            )
            out = np.fft.ifft2(np.fft.fft2(src) * mask).real
        elif name == "open_close_residual":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            opened = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
            out = _pol(g - (opened + closed) / 2, polarity)
        elif name in ("multiscale_top_hat", "multiscale_black_hat"):
            op = cv2.MORPH_TOPHAT if name == "multiscale_top_hat" else cv2.MORPH_BLACKHAT
            sizes = sorted({max(3, (size // 2) | 1), size, min(63, (size * 2 + 1))})
            out = np.maximum.reduce(
                [
                    cv2.morphologyEx(g, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n)))
                    for n in sizes
                ]
            )
        elif name in (
            "top_hat",
            "black_hat",
            "morph_gradient",
            "directional_top_hat",
            "directional_black_hat",
        ):
            footprint = (
                _foot(size, p["angle"])
                if name.startswith("directional_")
                else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            )
            op = (
                cv2.MORPH_GRADIENT
                if name == "morph_gradient"
                else (cv2.MORPH_BLACKHAT if "black_hat" in name else cv2.MORPH_TOPHAT)
            )
            out = cv2.morphologyEx(g, op, footprint)
        elif name == "multiscale_residual":
            responses = np.stack([g - _blur(g, s) for s in (sigma, sigma * 2, sigma * 4)])
            signed = np.take_along_axis(
                responses, np.argmax(np.abs(responses), axis=0)[None], axis=0
            )[0]
            out = _pol(signed, polarity)
        elif name in ("sobel", "scharr", "vector_residual"):
            out = (
                Gradient(
                    operator="sobel" if name == "sobel" else "scharr",
                    sigma=sigma,
                    residual_sigma=max(0.2, sigma * 4),
                    output="vector_residual" if name == "vector_residual" else p["output"],
                )
                .process([g], keep_intermediates=False)
                .primary
            )
        elif name == "laplacian":
            out = Laplacian(sigma=sigma).process([g], keep_intermediates=False).primary
        elif name == "canny":
            out = (
                cv2.Canny(
                    _u8(_blur(g, sigma)),
                    int(p["threshold"] * 255),
                    int(min(1, p["threshold"] * 2) * 255),
                ).astype(np.float32)
                / 255
            )
        elif name == "steerable":
            b = _blur(g, sigma)
            t = math.radians(p["angle"])
            out = _pol(
                cv2.Sobel(b, cv2.CV_32F, 1, 0) * math.cos(t)
                + cv2.Sobel(b, cv2.CV_32F, 0, 1) * math.sin(t),
                polarity,
            )
        elif name == "hessian":
            b = _blur(g, sigma)
            xx = cv2.Sobel(b, cv2.CV_32F, 2, 0)
            yy = cv2.Sobel(b, cv2.CV_32F, 0, 2)
            xy = cv2.Sobel(b, cv2.CV_32F, 1, 1)
            root = np.sqrt((xx - yy) ** 2 + 4 * xy**2)
            l1 = (xx + yy + root) / 2
            l2 = (xx + yy - root) / 2
            out = _pol(-np.where(np.abs(l1) > np.abs(l2), l1, l2), polarity)
        elif name in ("frangi", "sato", "meijering"):
            out = _ridge(name, g, sigma, polarity)
        elif name == "gabor":
            out = (
                GaborBank(sigma=sigma, periods=[p["period"]], orientations=p["orientations"])
                .process([g], keep_intermediates=False)
                .primary
            )
        elif name == "tensor":
            out = (
                StructureTensor(
                    tensor_sigma=sigma,
                    orientation_smooth_sigma=sigma * 4,
                    output="orientation_residual",
                )
                .process([g], keep_intermediates=False)
                .primary
            )
        elif name == "directional_residual":
            out = (
                DirectionalResidual(length=size, orientations=p["orientations"])
                .process([g], keep_intermediates=False)
                .primary
            )
        elif name == "entropy":
            out = rank.entropy(_u8(g), disk(p["radius"]))
        elif name == "lbp":
            out = feature.local_binary_pattern(_u8(g), 8, p["radius"], method="uniform") / 9
        elif name == "local_std":
            mean = cv2.blur(g, (size, size))
            out = np.sqrt(np.maximum(cv2.blur(g * g, (size, size)) - mean * mean, 0))
        elif name == "local_contrast":
            kernel = np.ones((size, size), np.uint8)
            out = cv2.dilate(g, kernel) - cv2.erode(g, kernel)
        elif name == "preset_scratches":
            out = _ridge("sato", clahe(g), sigma, "dark")
        elif name == "preset_defects":
            enhanced = clahe(g)
            out = np.abs(enhanced - _blur(enhanced, sigma))
        else:
            raise AssertionError(name)
        if enhanced is not None:
            enhanced = np.clip(enhanced, 0, 1).astype(np.float32)
            if name != "preset_defects":
                out = _gray(enhanced)
        out = np.asarray(out, dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite response")
        intermediates = (
            {"enhanced": enhanced}
            if enhanced is not None and context.get("keep_intermediates", True)
            else {}
        )
        return ProcessingResult(out, intermediates, diagnostics=metadata)


_GROUPS = {
    "A Brightness": "brightness gamma clahe equalize stretch log lab_clahe",
    "B Smoothing": "gaussian median bilateral nlmeans unsharp",
    "C Background": "local_residual median_residual dog log_response shading retinex multiscale_retinex homomorphic",
    "D Morphology": "top_hat black_hat morph_gradient open_close_residual multiscale_top_hat multiscale_black_hat directional_top_hat directional_black_hat multiscale_residual",
    "E Edges and ridges": "sobel scharr laplacian canny steerable hessian frangi sato meijering",
    "F Directional and frequency": "gabor tensor vector_residual directional_residual fft_highpass fft_bandpass",
    "G Texture": "entropy lbp local_std local_contrast",
    "H Presets": "preset_scratches preset_defects",
}
for _id, _name, _purpose, _parameters in SPECS:
    _category = next(group for group, members in _GROUPS.items() if _id in members.split())
    _info = MethodInfo(
        "surface_" + _id,
        _name,
        _category,
        1,
        False,
        _purpose,
        recommended_use=_purpose,
        limitations="Exploratory response; scale and polarity depend on method. Not a defect decision.",
        parameters={
            key: (
                P["ridge_polarity"]
                if key == "polarity" and _id in {"frangi", "sato", "meijering"}
                else P[key]
            ).copy()
            for key in _parameters.split()
        },
    )
    surface_registry.register(
        type("Surface_" + _id, (_SurfaceMethod,), {"info": _info, "__module__": __name__})
    )

__all__ = ["normalize_input", "surface_registry"]
