# Changelog

## Unreleased

- Fixed several PoC-processing correctness issues: invalid preset keys are now rejected instead of silently ignored, and all shipped presets are schema-valid.
- Structure Tensor and Gabor now expose defect-oriented primary outputs (including orientation residual) directly in the Processed view.
- Gradient keeps raw numerical response by default; display normalization remains a visualization concern, preserving cross-pattern comparability.
- Gabor internal downscale now scales its spatial kernel consistently; Local Phase Correlation exposes magnitude/dx/dy/quality outputs.
- Mono8 overlay rendering now preserves native camera brightness instead of per-frame min/max stretching.
- Added repository-level AGENTS.md guardrails: reuse proven Basler camera code first, keep hardware tests short, and prioritize the processing experiment.
- Aligned ExposureAuto/ExposureTime and GainAuto/Gain handling with the existing Basler test applications; automatic-loop runtime values are no longer restored/written as if they were manual settings.
- Made Original view a true camera/recording baseline with no continuous selected-method processing load.
- Added one-click local updater for the existing editable Windows installation and Desktop shortcut.
- Reworked live camera delivery to use a bounded mailbox instead of queueing full-resolution frame payloads into the Qt event loop.
- Coalesced heavy processing results so only the newest result can wait for the GUI; added result/camera-buffer drop diagnostics.
- Bounded live frame history to prevent multi-gigabyte growth on multi-megapixel streams.
- Made response visualization active-tab-only, debounced, and display-downscaled while keeping numerical processing at the selected processing scale.
- Reduced live memory/CPU cost for Structure Tensor, Gabor, temporal statistics, Farneback, DIS, and temporal fusion when intermediate maps are not requested.
- Fixed overlap handling in the DIC-like local phase-correlation map by averaging contributions instead of scan-order overwrite.
- Added temporary Basler free-run preparation (Continuous/FrameStart/Trigger Off where supported) with restore-on-close to recover from volatile trigger states left by other tools.
- Added camera-thread shutdown protection and reserved CPU headroom for the Qt event loop.
- Added idempotent Windows local setup, normal/debug launchers, and Desktop shortcut creation.
- Added a centralized Fusion dark theme using installed Qt system fonts.
- Added typed method editors and display-only range, heatmap, overlay, and shared compare controls.
- Fixed replay history reconstruction for strided and temporal multi-frame methods.
- Added explicit motion reset and correct cumulative integration across dropped processing requests.
- Hardened exact-target Basler node access, temporary ROI ordering, numeric increments, and sanitized labels.
- Bounded Windows hardware smoke test passed on one Basler a2A2448-23gmBAS with pypylon 26.6.

## 0.1.0 - 2026-09-25

- Initial experimental release.
- Modular single-frame and multi-frame processing registry.
- Synthetic translated/deformed pattern source.
- One-axis motion estimation and masked compensation.
- Raw session record/replay and engineering desktop UI.
- Optional exact-target Basler acquisition adapter for pypylon 26.6.
