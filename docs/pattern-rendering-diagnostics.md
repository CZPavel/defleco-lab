# Dynamic pattern artifacts and presentation diagnostics

## Current target and verified GPU routing

The 65-inch TV intentionally remains **1080x1920 portrait at 60 Hz**. There is no
resolution migration in this iteration. Period is in framebuffer pixels before
TV scaling; the operator-supplied 2160x3840 panel size implies ideal 2x scaling.

A subsequent controlled comparison confirmed a change from Intel to NVIDIA after
Windows Graphics Preference targeted the actual base `pythonw.exe` image rather
than only its venv launcher. The user observed ARTIFACT on Intel and CLEAN/clearly
improved on NVIDIA with identical sinusoidal-stripe settings. See
[nvidia-render-path-test.md](nvidia-render-path-test.md) for evidence, scope and limits.
Earlier observations below are retained as dated historical evidence.

## Distinguish three boundaries

1. **Rendered pixels:** wrong math, stale framebuffer regions, invalid GL state or
   context loss. Read back the image inside `paintGL`, before swap, and compare it
   with an independent reference. A clean sampled image does not rule out an
   intermittent problem in an unsampled frame.
2. **Motion sampling:** sharp edges can strobe, appear to reverse direction or form
   disturbing structures when rotation and breathing move fine patterns too far
   between refreshes. Spatial edge AA and VSync alone do not remove this aliasing.
3. **Presentation/scanout:** driver overrides, compositor, hybrid GPU transfer,
   cable/display processing and panel scanout are downstream of the framebuffer.
   A reported swap interval of 1 and 60 swaps/s do not prove zero physical tearing.

Continuous mode offers **Smooth movement over one display frame**, enabled by
default. Four midpoint temporal samples reduce strobing while retaining the exact
motion timeline. Moving edges become softer; disable the filter for sharp-frame
comparisons. Static/Stepped modes bypass it. It is not a camera exposure model or
a universally alias-free reconstruction.

## Reproduce the reported stripe case

Use binary/sharp stripes, period 50 **physical pixels**, rotation 20 deg/s,
breathing depth 20%, breathing frequency 0.5 Hz. These are separate settings:
50 px period is not 50% brightness or render scale.

```powershell
# Snapshot math, repeated draws, then six native pre-swap screen samples:
.\.venv\Scripts\python.exe tools\pattern_artifact_diagnostics.py --screen 1
# Same path with the production temporal filter:
.\.venv\Scripts\python.exe tools\pattern_artifact_diagnostics.py --screen 1 --motion-smoothing
# Additional 4K computational check (does not change display resolution):
.\.venv\Scripts\python.exe tools\pattern_artifact_diagnostics.py --width 2160 --height 3840 --frames 8 --motion-smoothing
```

The default computational sweep samples a full virtual 18-second rotation and
probes wrap boundaries. It does not run an 18-second hardware acquisition. Each
sample is compared to a separate float64 AA implementation; repeated draws must
be identical. Every pixel must be within 3 Mono8 levels. A small severe corrupt
patch fails even when its global area is tiny. `--output folder` saves at most
three failing images. No camera is opened and no driver setting is written.

Readback and reference evaluation are intrusive. Native captures are taken only
inside `paintGL`; CPU comparison happens after hiding the output. Interval metrics
exclude sampled swaps and their adjacent intervals. They remain software metrics,
not measurements of the panel's optical output.

## Notebook observations, 2026-10-26

- Windows 11 build 26200; Intel UHD driver 32.0.101.7085 and NVIDIA RTX A5000
  Laptop GPU driver 596.52.
- Qt/DxDiag reported the active external display as 1080x1920, 60 Hz, HDMI. A
  separate WMI adapter field reported 3840x2160; it was not used as the active
  window resolution. Internal display: 1920x1080, 60 Hz.
- DxDiag associated external scanout with NVIDIA, while a fresh external-only
  OpenGL window reported Intel as its renderer. This supports a hybrid transfer
  path; it does not establish that transfer as the artifact's cause.
- The old shader showed minor edge differences (maximum 7 Mono8 levels in one
  sampled native frame). Analytic gradients plus CPU orientation coefficients
  reduced checked frames to maximum 1 level. No random corrupt region was
  reproduced in the sampled framebuffers.
- The final four-sample FHD run produced 352 swaps in approximately six seconds;
  unsampled interval median 16.68 ms, p95 17.72 ms. The six sampled native frames
  matched the independent temporal reference within one Mono8 level.
- Read-only NVIDIA profile queries returned no explicit entries for the queried
  global/base VSync/presentation settings and no matching Python application
  profile. Missing entries do not establish effective driver values.

These observations are bounded local evidence. Physical tearing, intermittent
unsampled corruption and camera/display synchronization remain unverified.

## Targeted driver/display checks

1. Save a baseline with **Save display diagnostics** while the output is running.
   Confirm the renderer, physical dimensions and selected display.
2. For hybrid routing, use Windows **Settings > System > Display > Graphics** to
   test an application-specific high-performance NVIDIA preference for the actual
   interpreter executable; restart and verify that the renderer really changes.
   Python virtual-environment launchers may spawn a base interpreter: a preference
   for a shared base Python can affect other Python applications too. The program
   deliberately does not write this setting automatically.
3. If inspecting NVIDIA Control Panel, inspect the relevant application profile's
   **Vertical sync** setting. An app/context request is not a readback of every
   driver override. Do not interpret adaptive modes or global defaults as proof
   of synchronized presentation. Preserve other applications' settings.
4. Compare the same pattern with smoothing on/off and then with both rotation and
   breathing frequency reduced fourfold, keeping period unchanged. Disturbance
   that follows motion speed supports temporal aliasing; a stable corrupt patch
   in a frozen framebuffer points elsewhere.
5. If video also tears, compare an identical clip on each display and inspect the
   browser's GPU diagnostics. A shared symptom supports investigating the common
   Windows/driver/display path; it does not identify a particular bad setting.

Do not infer that disabling MPO/HAGS, installing a driver, changing global VSync,
or forcing a different GPU is a verified fix. None of those system changes were
performed during this investigation.

## Official references

- [Qt QOpenGLWindow](https://doc.qt.io/qt-6/qopenglwindow.html): swap-driven updates and framebuffer readback boundary.
- [Qt QSurfaceFormat](https://doc.qt.io/qt-6/qsurfaceformat.html): swap interval and context-reset requests.
- [NVIDIA Manage 3D Settings](https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/3D%20Settings/Manage_3D_Settings_%28reference%29.htm): application profiles and VSync controls.
- [NVIDIA GPU-selection precedence](https://nvidia.custhelp.com/app/answers/detail/a_id/5035): Windows GPU preference versus NVIDIA preference.
- [Microsoft application graphics preference](https://support.microsoft.com/en-us/windows/hardware/display-graphics/optimizations-for-windowed-games-in-windows-11): application GPU preference and restart. The page's separate windowed-game optimization concerns DirectX 10/11, not a claimed OpenGL fix.
- [NVIDIA NVAPI headers](https://github.com/NVIDIA/nvapi): read-only DRS query contracts and setting identifiers.
