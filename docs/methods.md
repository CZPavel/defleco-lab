# Processing methods

All numeric processing uses floating-point arrays. Visualization normalization and heat maps do not change stored raw frames.

## Single frame

- **Sobel / Scharr Gradient**: X, Y, and magnitude responses; useful for local slope/intensity changes but noise-sensitive. Raw magnitude is the default so response amplitudes remain comparable; display normalization is handled separately.
- **Laplacian**: signed and absolute second derivative; highlights narrow changes and noise.
- **Difference of Gaussians**: band-pass/high-pass residual at two configurable scales.
- **Local Background Residual**: subtracts a smooth Gaussian background.
- **Structure Tensor**: dominant local orientation, coherence, eigenvalue anisotropy, and orientation residual. The primary output is selectable; `orientation_residual` is intended for direct defect-enhancement experiments. Orientation is pi-periodic and smoothing uses `cos(2 theta)` / `sin(2 theta)`.
- **Gabor Filter Bank**: maximum frequency/orientation response, dominant orientation, first-period response, and orientation residual. The primary output is selectable. Its internal scale keeps configured period/sigma semantics consistent while reducing compute cost.
- **Directional Residual**: experimental discretized oriented-line background approximation, not a reproduction of a proprietary method.

## Multi frame

- Absolute frame difference with configurable stride.
- Temporal mean, median, standard deviation, range, percentile range, and temporal-median residual.
- Farneback dense optical flow with `u`, `v`, magnitude, angle, divergence, curl, and local-flow residual.
- DIS dense optical flow using supported OpenCV presets.
- DIC-like local phase-correlation grid with displacement, texture, quality, and vector diagnostics. It is not claimed to be full scientific DIC.
- Temporal response fusion by maximum, mean, median, or percentile.

Every method is scene-dependent. The outputs are research indicators, not calibrated surface height or pass/fail decisions.
