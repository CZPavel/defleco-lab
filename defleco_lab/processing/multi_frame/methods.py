from __future__ import annotations

import cv2
import numpy as np

from ..base import MethodInfo, ProcessingMethod, ProcessingResult
from ..preprocessing import gray32


def _pair(frames, stride):
    if stride < 1 or len(frames) <= stride:
        raise ValueError("frame stride unavailable")
    return gray32(frames[-1 - stride]), gray32(frames[-1])


class FrameDifference(ProcessingMethod):
    info = MethodInfo(
        "frame_difference",
        "Absolute Frame Difference",
        "Multi frame",
        2,
        True,
        "Absolute intensity change.",
        parameters={
            "stride": 1,
            "blur_sigma": {
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 0.1,
                "units": "px",
            },
            "local_normalization": False,
        },
    )

    def _process(self, frames, **context):
        old, cur = _pair(frames, int(self.parameters["stride"]))
        s = float(self.parameters["blur_sigma"])
        if s > 0:
            old, cur = cv2.GaussianBlur(old, (0, 0), s), cv2.GaussianBlur(cur, (0, 0), s)
        out = np.abs(cur - old)
        mask = context.get("valid_mask")
        if self.parameters["local_normalization"]:
            out /= cv2.GaussianBlur(np.abs(cur), (0, 0), 8) + 1e-6
        if mask is not None:
            out = np.where(mask, out, 0)
        return ProcessingResult(out, {"difference": out}, valid_mask=mask)


class TemporalStatistics(ProcessingMethod):
    info = MethodInfo(
        "temporal_statistics",
        "Temporal Statistics",
        "Multi frame",
        2,
        True,
        "N-frame mean, median, standard deviation and ranges.",
        parameters={
            "window": {"default": 8, "minimum": 2, "maximum": 256, "units": "frames"},
            "stride": 1,
            "percentile_low": {
                "default": 10.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 1.0,
                "units": "%",
            },
            "percentile_high": {
                "default": 90.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 1.0,
                "units": "%",
            },
            "output": {
                "default": "std",
                "choices": ["mean", "median", "std", "range", "percentile_range"],
            },
        },
    )

    def _process(self, frames, **context):
        n = int(self.parameters["window"])
        stride = int(self.parameters["stride"])
        need = 1 + (n - 1) * stride
        if len(frames) < need:
            raise ValueError("temporal window unavailable")
        stack = np.stack([gray32(frames[-1 - i * stride]) for i in reversed(range(n))])
        lo = float(self.parameters["percentile_low"])
        hi = float(self.parameters["percentile_high"])
        output = self.parameters["output"]
        keep = context.get("keep_intermediates", True)

        def compute(name):
            if name == "mean":
                return stack.mean(0)
            if name == "median":
                return np.median(stack, 0)
            if name == "std":
                return stack.std(0)
            if name == "range":
                return np.ptp(stack, axis=0)
            if name == "percentile_range":
                return np.percentile(stack, hi, axis=0) - np.percentile(stack, lo, axis=0)
            raise ValueError(f"Unsupported temporal output: {name}")

        primary = compute(output).astype(np.float32)
        if not keep:
            return ProcessingResult(primary)
        maps = {name: compute(name).astype(np.float32) for name in (
            "mean", "median", "std", "range", "percentile_range"
        )}
        return ProcessingResult(primary, maps)


class TemporalMedianResidual(ProcessingMethod):
    info = MethodInfo(
        "temporal_median_residual",
        "Temporal Median Residual",
        "Multi frame",
        2,
        True,
        "Current frame minus temporal median.",
        parameters={
            "window": {"default": 5, "minimum": 2, "maximum": 256, "units": "frames"},
            "stride": 1,
        },
    )

    def _process(self, frames, **_):
        n = int(self.parameters["window"])
        stride = int(self.parameters["stride"])
        stack = np.stack([gray32(frames[-1 - i * stride]) for i in reversed(range(n))])
        med = np.median(stack, 0)
        signed = stack[-1] - med
        return ProcessingResult(
            np.abs(signed).astype(np.float32),
            {"median": med.astype(np.float32), "signed": signed.astype(np.float32)},
        )


class FarnebackFlow(ProcessingMethod):
    info = MethodInfo(
        "farneback",
        "Dense Optical Flow - Farneback",
        "Multi frame",
        2,
        True,
        "Dense two-frame optical flow.",
        parameters={
            "stride": 1,
            "pyr_scale": {"default": 0.5, "minimum": 0.01, "maximum": 0.99, "step": 0.05},
            "levels": {"default": 3, "minimum": 1, "maximum": 16},
            "winsize": {"default": 15, "minimum": 3, "maximum": 255, "step": 2, "units": "px"},
            "iterations": {"default": 3, "minimum": 1, "maximum": 100},
            "poly_n": {"default": 5, "choices": [5, 7]},
            "poly_sigma": {"default": 1.2, "minimum": 0.1, "maximum": 10.0, "step": 0.1},
            "flags": {"default": 0, "minimum": 0, "maximum": 1024},
            "scale": {"default": 1.0, "minimum": 0.1, "maximum": 1.0, "step": 0.1},
            "output": {
                "default": "magnitude",
                "choices": ["u", "v", "magnitude", "angle", "divergence", "curl", "local_residual"],
            },
            "residual_sigma": {
                "default": 8.0,
                "minimum": 0.1,
                "maximum": 500.0,
                "step": 0.5,
                "units": "px",
            },
        },
    )

    def _process(self, frames, **context):
        old, cur = _pair(frames, int(self.parameters["stride"]))
        scale = float(self.parameters["scale"])
        if scale != 1:
            old = cv2.resize(old, None, fx=scale, fy=scale)
            cur = cv2.resize(cur, None, fx=scale, fy=scale)
        p = self.parameters
        flow = cv2.calcOpticalFlowFarneback(
            old,
            cur,
            None,
            float(p["pyr_scale"]),
            int(p["levels"]),
            int(p["winsize"]),
            int(p["iterations"]),
            int(p["poly_n"]),
            float(p["poly_sigma"]),
            int(p["flags"]),
        )
        flow /= scale
        if scale != 1:
            flow = cv2.resize(flow, (frames[-1].shape[1], frames[-1].shape[0]))
        u, v = flow[..., 0], flow[..., 1]
        output = p["output"]
        keep = context.get("keep_intermediates", True)

        def selected_map(name):
            if name == "u":
                return u
            if name == "v":
                return v
            if name == "magnitude":
                return np.hypot(u, v)
            if name == "angle":
                return np.arctan2(v, u).astype(np.float32)
            if name == "divergence":
                return cv2.Sobel(u, cv2.CV_32F, 1, 0) + cv2.Sobel(v, cv2.CV_32F, 0, 1)
            if name == "curl":
                return cv2.Sobel(v, cv2.CV_32F, 1, 0) - cv2.Sobel(u, cv2.CV_32F, 0, 1)
            if name == "local_residual":
                sx = cv2.GaussianBlur(u, (0, 0), float(p["residual_sigma"]))
                sy = cv2.GaussianBlur(v, (0, 0), float(p["residual_sigma"]))
                return np.hypot(u - sx, v - sy)
            raise ValueError(f"Unsupported flow output: {name}")

        primary = selected_map(output)
        if not keep:
            return ProcessingResult(primary)
        maps = {
            "u": u,
            "v": v,
            "magnitude": selected_map("magnitude"),
            "angle": selected_map("angle"),
            "divergence": selected_map("divergence"),
            "curl": selected_map("curl"),
            "local_residual": selected_map("local_residual"),
            "flow": flow,
        }
        return ProcessingResult(primary, maps)


class DISFlow(ProcessingMethod):
    info = MethodInfo(
        "dis_flow",
        "DIS Optical Flow",
        "Multi frame",
        2,
        True,
        "Fast dense inverse-search optical flow.",
        parameters={
            "stride": 1,
            "preset": {"default": "medium", "choices": ["ultrafast", "fast", "medium"]},
            "output": {"default": "magnitude", "choices": ["u", "v", "magnitude", "angle"]},
        },
    )

    def _process(self, frames, **context):
        if not hasattr(cv2, "DISOpticalFlow_create"):
            raise RuntimeError("OpenCV DIS optical flow unavailable")
        old, cur = _pair(frames, int(self.parameters["stride"]))
        presets = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        flow = cv2.DISOpticalFlow_create(presets[self.parameters["preset"]]).calc(
            old.astype(np.uint8), cur.astype(np.uint8), None
        )
        u, v = flow[..., 0], flow[..., 1]
        output = self.parameters["output"]
        if output == "u":
            primary = u
        elif output == "v":
            primary = v
        elif output == "angle":
            primary = np.arctan2(v, u).astype(np.float32)
        else:
            primary = np.hypot(u, v)
        if not context.get("keep_intermediates", True):
            return ProcessingResult(primary)
        maps = {
            "u": u,
            "v": v,
            "magnitude": np.hypot(u, v),
            "angle": np.arctan2(v, u).astype(np.float32),
            "flow": flow,
        }
        return ProcessingResult(primary, maps)


class LocalPhaseCorrelation(ProcessingMethod):
    info = MethodInfo(
        "local_phase_correlation",
        "DIC-like Local Correlation",
        "Multi frame",
        2,
        True,
        "Grid of windowed local translations.",
        limitations="Local correlation, not full scientific DIC.",
        parameters={
            "stride": 1,
            "window_size": {"default": 32, "minimum": 8, "maximum": 512, "step": 2, "units": "px"},
            "grid_stride": {"default": 16, "minimum": 1, "maximum": 512, "units": "px"},
            "minimum_texture": {"default": 3.0, "minimum": 0.0, "maximum": 255.0, "step": 0.1},
            "minimum_q": {
                "default": 0.05,
                "minimum": -1.0,
                "maximum": 1.0,
                "step": 0.01,
                "decimals": 3,
            },
            "max_displacement": {
                "default": 20.0,
                "minimum": 0.0,
                "maximum": 10000.0,
                "step": 1.0,
                "units": "px",
            },
        },
    )

    def _process(self, frames, **_):
        old, cur = _pair(frames, int(self.parameters["stride"]))
        w = int(self.parameters["window_size"])
        gs = int(self.parameters["grid_stride"])
        win = cv2.createHanningWindow((w, w), cv2.CV_32F)
        sum_dx = np.zeros(old.shape, np.float32)
        sum_dy = np.zeros(old.shape, np.float32)
        sum_q = np.zeros(old.shape, np.float32)
        count = np.zeros(old.shape, np.float32)
        overlays = []
        for y in range(0, old.shape[0] - w + 1, gs):
            for x in range(0, old.shape[1] - w + 1, gs):
                a, b = old[y : y + w, x : x + w], cur[y : y + w, x : x + w]
                if min(a.std(), b.std()) < float(self.parameters["minimum_texture"]):
                    continue
                shift, response = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), win)
                valid = response >= float(self.parameters["minimum_q"]) and np.hypot(
                    *shift
                ) <= float(self.parameters["max_displacement"])
                if valid:
                    region = np.s_[y : y + w, x : x + w]
                    sum_dx[region] += shift[0]
                    sum_dy[region] += shift[1]
                    sum_q[region] += response
                    count[region] += 1
                    overlays.append(
                        {
                            "x": x + w / 2,
                            "y": y + w / 2,
                            "dx": shift[0],
                            "dy": shift[1],
                            "q": response,
                        }
                    )
        valid_pixels = count > 0
        dx = np.full(old.shape, np.nan, np.float32)
        dy = np.full(old.shape, np.nan, np.float32)
        q = np.full(old.shape, np.nan, np.float32)
        dx[valid_pixels] = sum_dx[valid_pixels] / count[valid_pixels]
        dy[valid_pixels] = sum_dy[valid_pixels] / count[valid_pixels]
        q[valid_pixels] = sum_q[valid_pixels] / count[valid_pixels]
        mag = np.hypot(dx, dy)
        return ProcessingResult(
            np.nan_to_num(mag),
            {"dx": dx, "dy": dy, "magnitude": mag, "quality": q},
            overlays=overlays,
            valid_mask=np.isfinite(mag),
        )


class TemporalFusion(ProcessingMethod):
    info = MethodInfo(
        "temporal_fusion",
        "Temporal Response Fusion",
        "Multi frame",
        2,
        True,
        "Fuses response maps from multiple pairs.",
        parameters={
            "pairs": {"default": 4, "minimum": 1, "maximum": 256},
            "stride": 1,
            "mode": {"default": "max", "choices": ["max", "mean", "median", "percentile"]},
            "percentile": {
                "default": 90.0,
                "minimum": 0.0,
                "maximum": 100.0,
                "step": 1.0,
                "units": "%",
            },
        },
    )

    def _process(self, frames, **context):
        n = int(self.parameters["pairs"])
        stride = int(self.parameters["stride"])
        cur = gray32(frames[-1])
        mode = self.parameters["mode"]
        keep = context.get("keep_intermediates", True)

        if not keep and mode in {"max", "mean"}:
            accumulator = None
            for i in range(1, n + 1):
                response = np.abs(cur - gray32(frames[-1 - i * stride]))
                if accumulator is None:
                    accumulator = response.copy()
                elif mode == "max":
                    np.maximum(accumulator, response, out=accumulator)
                else:
                    accumulator += response
            assert accumulator is not None
            if mode == "mean":
                accumulator /= n
            return ProcessingResult(accumulator.astype(np.float32, copy=False))

        maps = np.stack(
            [np.abs(cur - gray32(frames[-1 - i * stride])) for i in range(1, n + 1)]
        )
        out = {
            "max": lambda: maps.max(0),
            "mean": lambda: maps.mean(0),
            "median": lambda: np.median(maps, 0),
            "percentile": lambda: np.percentile(
                maps, float(self.parameters["percentile"]), axis=0
            ),
        }[mode]()
        intermediates = {"pair_responses": maps} if keep else {}
        return ProcessingResult(out.astype(np.float32), intermediates)
