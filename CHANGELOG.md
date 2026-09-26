# Changelog

## Unreleased

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
