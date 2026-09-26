# Defleco LAB

Defleco LAB is an independent, experimental Windows desktop laboratory for comparing image-processing methods on reflected dynamic patterns. It is intended for research on local disturbances visible on glossy or specular surfaces using synthetic data, recorded sequences, image sequences, or an explicitly selected Basler camera.

> This is research software, not a calibrated measurement system, production inspection, or safety component. No metrological accuracy is claimed. Camera hardware behavior is **NOT TESTED** in this release.

![Defleco LAB with a generated synthetic fringe scene](docs/assets/synthetic-lab.png)

## Highlights

- Single-frame gradient, Laplacian, Difference of Gaussians, local residual, structure tensor, Gabor-bank, and experimental directional-residual views.
- Multi-frame difference, temporal statistics/median residual, Farneback and DIS optical flow, DIC-like local phase-correlation grid, and temporal response fusion.
- One-axis image-motion estimation from multiple dedicated Motion ROIs, quality gating, robust median fusion, cumulative position, and masked translation compensation.
- Deterministic synthetic fringes, grid, checkerboard, speckle, translation, and localized deformation.
- Raw-frame record/replay so the same sequence can be evaluated repeatedly with different algorithms.
- PySide6 engineering UI with zoom, pan, fit, 1:1, compare views, editable parameters, presets, Analysis ROIs, and Motion ROIs.
- CPU baseline; OpenCV CUDA is not required.

## Install on Windows

Requirements: Windows 10/11 x64, Python 3.11 x64. Hardware use additionally requires a compatible pylon Runtime and `pypylon 26.6`.

```powershell
git clone https://github.com/CZPavel/defleco-lab.git
cd defleco-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m defleco_lab
```

For optional Basler support:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[basler]"
```

The application never selects real hardware by connection order. Discovery is read-only and the operator must select an exact discovered camera. Supported temporary acquisition settings are capability-checked and restored on disconnect where practical. Persistent User Sets, Force IP, firmware, network persistence, and GPIO output are intentionally outside scope.

## Quick synthetic experiment

1. Start with `Synthetic` and `fringes`.
2. Compare `Structure Tensor`, `Gabor Filter Bank`, and `Difference of Gaussians`.
3. Start Live, choose a multi-frame method, and change frame stride.
4. Draw cyan Analysis ROIs and orange Motion ROIs. Select an ROI and use `+`/`-` to resize or `Delete` to remove it.
5. Record a short raw session, reload it, then select another method and replay the same frames.

Presets are editable starting points, not validated optimal settings.

## Architecture

```text
source -> immutable FramePacket -> bounded frame history -> processing method -> result/view
                         \-> bounded raw recorder -> session -> replay --------/
```

Algorithms implement a common metadata-rich interface; the parameter panel is generated from each method schema. Camera-specific objects do not enter processing code. See [architecture](docs/architecture.md), [methods](docs/methods.md), [motion compensation](docs/motion-compensation.md), and [session format](docs/session-format.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools\synthetic_acceptance.py
```

Normal CI needs no camera. A separate bounded manual smoke test is documented but hardware remains **NOT TESTED** until its result is recorded for an exact model/runtime/setup.

## Scientific context and limitations

The included algorithms are comparative tools. Reflected patterns, saturation, display behavior, motion blur, periodic ambiguity, surface curvature, viewing geometry, calibration, and synchronization can all dominate results. DIC-like local correlation is not presented as full scientific DIC. Directional residual is an experimental approximation, not a reproduction of a proprietary algorithm. See [references](docs/references.md).

## License and trademark notice

Source code is released under the [MIT License](LICENSE). Portions of the motion-estimation design were refactored from the MIT-licensed [basler-camera-encoder](https://github.com/CZPavel/basler-camera-encoder); see [NOTICE](NOTICE).

Basler and pylon are trademarks or product names of their respective owner. This independent project is not affiliated with, endorsed by, or supported by Basler. It does not redistribute pylon, vendor binaries, manuals, or proprietary images.
