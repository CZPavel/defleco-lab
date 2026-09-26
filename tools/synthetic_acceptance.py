from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defleco_lab.camera.synthetic_source import SyntheticConfig, SyntheticSource
from defleco_lab.motion import compensate_translation
from defleco_lab.processing import load_builtin_methods
from defleco_lab.processing.registry import registry


def main() -> int:
    load_builtin_methods()
    source = SyntheticSource(SyntheticConfig(width=480, height=320, shift_per_frame_px=3.0))
    frames = [source.next_frame().image for _ in range(10)]
    checks = {
        "structure_tensor": [frames[-1]],
        "gabor": [frames[-1]],
        "dog": [frames[-1]],
        "frame_difference": frames[-2:],
        "farneback": frames[-2:],
        "temporal_statistics": frames,
    }
    for method_id, inputs in checks.items():
        result = registry.create(method_id).process(inputs)
        assert result.primary.shape[:2] == frames[-1].shape[:2]
        assert np.isfinite(result.primary).all()
        print(f"PASS {method_id}: {result.processing_time_ms:.2f} ms")
    raw = np.mean(cv2.absdiff(frames[-2], frames[-1]))
    print(f"Synthetic sequence: {len(frames)} frames; uncompensated mean difference {raw:.3f}")
    base = np.random.default_rng(11).normal(127, 30, (240, 360)).astype(np.float32)
    dx = 7
    current = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]), (360, 240))
    defect = np.zeros(current.shape, bool)
    defect[90:135, 210:255] = True
    current[defect] += 35
    compensation = compensate_translation(base, 0, dx)
    valid = compensation.valid_mask
    unchanged = valid & ~defect
    raw_error = float(np.mean(np.abs(base[unchanged] - current[unchanged])))
    compensated_error = float(np.mean(np.abs(compensation.aligned[unchanged] - current[unchanged])))
    defect_residual = float(
        np.mean(np.abs(compensation.aligned[valid & defect] - current[valid & defect]))
    )
    assert compensated_error < raw_error * 0.2
    assert defect_residual > compensated_error * 5
    print(
        "PASS compensated moving-object case: "
        f"unchanged {raw_error:.3f} -> {compensated_error:.3f}; "
        f"localized residual {defect_residual:.3f}"
    )
    print("Hardware acceptance: outside this hardware-independent synthetic test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
