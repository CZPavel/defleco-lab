# NVIDIA render-path A/B test — 2026-10-26

## Scope and outcome

**NVIDIA renderer confirmed. Physical artifact changed/improved in the controlled
test according to the user. Hybrid GPU routing is likely relevant.** This is not
a universal guarantee for every pattern, driver state, TV mode or optical setup.

The TV remained **1080x1920 portrait, 60 Hz**. No resolution switch, global NVIDIA
setting, driver update, HAGS/MPO/VRR change, shader/carrier/AA change or motion
sampling modification was performed.

## Actual executable, not an assumed launcher path

The production CMD and the test both launch `.venv\Scripts\pythonw.exe`.
In the measured application process:

- `sys.executable` identifies the venv `pythonw.exe`;
- `sys._base_executable` identifies the installed Python 3.11 `pythonw.exe`;
- `QueryFullProcessImageNameW(GetCurrentProcess())` identifies that same **base
  pythonw.exe** as the actual image, with a separate PID from the venv launcher.

The exact local paths and PIDs are saved in the local A/B diagnostics. The base
console `python.exe` was separately measured and is a different executable.
Only the actual base **pythonw.exe** received the preference in this iteration.
Consequently, console tests launched with python.exe must not be assumed to use
the same GPU. The preference can also affect other programs sharing that base
pythonw.exe; it is per executable, not per Python module or venv.

## Windows preference and restart

Used Windows **Settings > System > Display > Graphics > Add desktop app**, selected
the measured actual base `pythonw.exe`, then chose **High performance (NVIDIA RTX
A5000 Laptop GPU)**. The Python test process was completely exited and relaunched.
Read-only registry evidence changed from a missing value to `GpuPreference=2;`
for that exact image. The venv path itself still had no preference entry.

No registry-writing helper or global NVIDIA profile modification was used. To
revert the routing preference, select the same executable in Windows Graphics and
choose **Let Windows decide**, then fully restart the application. The original
configuration had no explicit entry; removing the app entry restores that absence.
Do not reset unrelated application or global options.

## Context and format audit

Lifecycle logs showed no context after window construction, nor after the target
screen was assigned. `initializeGL` ran only after `Beyond TV` had been assigned
and exposed. The application does not create a GL context on the internal display
and subsequently move an already initialized output there during this initial
show path.

The owned Qt context's `screen()` still reported the primary screen in both runs,
including the confirmed NVIDIA run. This metadata alone is not a GPU-selection
oracle. Qt 6.11.2's `QOpenGLWindow` constructs its context internally. No private
Qt workaround or replacement lifecycle was introduced without a demonstrated need.

Per-window format is set before context creation: OpenGL 3.3 Core, double buffer,
swap interval 1, 8-bit RGB, no MSAA. Runtime readback matched the requested context
properties. A global format change was unnecessary and was not claimed as a
routing fix. Swap interval 1 remains a context request/readback, not proof of
optical synchronization.

## Native export mechanism

The installed base Python console/windowed PE images have no export directory and
do not export `NvOptimusEnablement` or the AMD equivalent. A Python variable of
that name does not create a native PE export. A small launcher that merely spawns
the same separate Python process does not establish that the child's image exports
the symbol. No pseudo-fix, executable patch or speculative native bootstrapper
was added; the supported Windows preference already produced measured NVIDIA
rendering. AMD-specific code had no role on this Intel/NVIDIA machine.

## Reproducible A/B procedure

Primary preset: stripes, sinusoidal, period 50 **framebuffer px**, rotation
20 deg/s, breathing 20% at 0.5 Hz, motion smoothing OFF. Native panel size
2160x3840 is an operator declaration, not automatic EDID measurement. Framebuffer
1080x1920 therefore implies ideal scale_x=scale_y=2 and approximately 100 native
panel pixels per 50-framebuffer-pixel period.

```powershell
# Use the production windowed interpreter, not a different console executable.
.\.venv\Scripts\pythonw.exe tools\pattern_render_path_test.py --output work\gpu-A --label A --seconds 60 --expect-renderer Intel
# Change only the measured executable's Windows GPU preference, then restart:
.\.venv\Scripts\pythonw.exe tools\pattern_render_path_test.py --output work\gpu-B --label B --seconds 60 --expect-renderer NVIDIA
```

The tool creates the real `MainWindow` and production `PatternOutputWindow`; no
offscreen GL context is created. It enforces the existing 1080x1920@60 mode and
does not change it. Two images are captured inside `paintGL` before swap, with
readback and adjacent intervals excluded from the full-run cadence summary.
Physical observation remains UNCERTAIN unless a human supplies it. The optional
`--preset slow` uses rotation 5 deg/s, breathing 5%, 0.125 Hz with other parameters
unchanged; use paired Intel/NVIDIA slow runs if the primary difference is unclear.

## Evidence

Both 60-second primary runs rendered 1080x1920 on Beyond TV at reported 60 Hz,
DPR 1.0, requested/context swap interval 1. Both saved readbacks matched their
sinusoidal CPU reference within 1 Mono8 level. Readback does not establish panel
scanout; the physical results below are the user's direct observations.

| Test | Renderer | Display | FPS/swap | Physical observation | Result |
| --- | --- | --- | --- | --- | --- |
| A | Intel UHD, driver 32.0.101.7085 | 1080x1920@60 | 60.10 median-derived FPS; median 16.639 ms, p95 18.726 ms; max 73.480 ms | User saw tearing or other artifacts | ARTIFACT |
| B | NVIDIA RTX A5000 Laptop GPU, driver 596.52 | 1080x1920@60 | 60.02 median-derived FPS; median 16.662 ms, p95 17.114 ms; max 23.965 ms | User: artifacts disappeared or image clearly improved | CLEAN / improved |

1. **NVIDIA renderer confirmed** by GL_VENDOR/GL_RENDERER/GL_VERSION in the actual
   application output and readback of the exact executable preference.
2. **Physical artifact changed/improved** in this matched test, based on human
   observation. It is not inferred from screenshots or FPS.
3. **Hybrid routing likely relevant**: preference was the intentional variable,
   shader and display mode stayed unchanged. This remains a bounded practical
   result, not proof of a particular fault inside either driver.
4. **Next experiment:** retain the confirmed NVIDIA route and repeat the original
   sharp/binary case under the same 1080p mode. If artifacts persist despite a
   correct framebuffer and stable cadence, prioritize TV processing and physical
   scanout, then temporal aliasing and camera/display synchronization. Do not
   automatically return to shader changes. The primary difference was clear, so
   an additional paired slow physical A/B was not required in this iteration.

## References

- [Qt 6.11.2 QOpenGLWindow implementation](https://github.com/qt/qtbase/blob/v6.11.2/src/opengl/qopenglwindow.cpp).
- [Qt QOpenGLContext](https://doc.qt.io/qt-6/qopenglcontext.html): screen/format association before create.
- [NVIDIA Optimus rendering policies](https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf): native exported symbol mechanism.
- [NVIDIA Windows GPU-selection precedence](https://nvidia.custhelp.com/app/answers/detail/a_id/5035).
- [Microsoft Graphics Preference](https://support.microsoft.com/en-us/windows/hardware/display-graphics/optimizations-for-windowed-games-in-windows-11).
