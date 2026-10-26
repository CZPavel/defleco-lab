# Architecture

Defleco LAB separates acquisition, immutable frame data, numeric analysis, recording,
presentation, physical pattern output, and experiment orchestration. `FramePacket` is
camera-independent and owns its image buffer. A bounded chronological history
preserves source-frame spacing, while live processing may discard obsolete requests
to avoid unbounded latency. Raw recording has a separate loss-reporting path.

## Analysis pipeline

The numeric path is deliberately split into four layers:

```text
camera / replay frame
        |
        v
1. pre-processing
        |
        v
2. analysis method
        |
        v
3. response post-processing
        |
        v
numeric float response
        |
        +----------------------> saved/compared numeric data
        |
        v
4. display mapping / overlay
```

Pre-processing changes the method input. Analysis methods are the actual feature or
motion calculations (Scharr, Structure Tensor, Frame Difference, Farneback, etc.).
Response post-processing operates on the numeric response. Display range, colour map
and alpha overlay are presentation-only and are never fed back into numeric analysis.

The GUI is schema-driven: each processing plug-in publishes its name, category,
frame requirement, help, limitations, references, editable parameters, and optional
parameter-visibility dependencies. Changing an output therefore exposes only the
controls relevant to that output.

## Pattern output

The physical pattern generator is part of the same application but remains
architecturally separate from camera acquisition and analysis. A full-screen,
borderless output window can be placed on the external display. The integrated renderer reuses the established V02 pattern families: stripes,
checkerboard, circular/ring fringes, composite X+Y, nested squares, squircle,
spiral, counter-spiral, starburst, speckle and solid field. The first physical
screening still prioritises fine stripes, checkerboard, single spiral and rings.

Pattern state is deterministic in static/stepped mode. This makes a software-level
step-and-capture sequence possible without pretending that the display and camera
have hardware-level frame synchronization. Continuous rotation, phase sweep and
period breathing can be combined for visual screening.

`PatternOutputWindow` is a native `QOpenGLWindow` with an OpenGL 3.3 procedural
fragment shader (`pattern_shader.py`). Each analytic pattern is evaluated on the
GPU in physical display pixels. Only deterministic speckle is generated on the
CPU and uploaded as a cached texture. The offline `render_pattern` Mono8 API is
preserved as the reference renderer.

The window requests double buffering and swap interval 1. Continuous updates are
scheduled from `frameSwapped`, with motion advanced using monotonic elapsed time.
There is no software animation-FPS timer or intermediate downscaled pixmap.
Binary patterns use subpixel edge coverage; angular patterns use analytic
derivatives to avoid a false seam at the atan branch boundary. Rotation is
360-degree periodic; starburst phase is 720-degree periodic.

Affine stripe/checker/composite coverage uses analytic spatial gradients, and
orientation sine/cosine coefficients are computed once on the CPU. This avoids
observed Intel edge differences from GPU derivatives/trigonometric approximations.
The renderer resets clipping, stencil, culling, color writes and raster discard,
clears the full target, and checks current-context ownership, binding and GL errors.

Continuous motion optionally integrates four midpoint states over the preceding
reported display refresh interval. Sampling uses the original base period and
rotation/phase/breathing timeline; a missed frame does not extend the integration
window or accumulate old frames. Signals are averaged before Mono8 quantization.
Static/stepped cases bypass this filter. The last submitted temporal states are
available to the explicit artifact diagnostic for independent pixel verification.

The display driver/compositor ultimately controls scanout. A completed swap is a
software presentation boundary, not proof of optical settling or synchronized
camera exposure. The earlier standalone dynamic-pattern PoC remains the design
source for pattern-family semantics and future synchronization work.

## Automated screening

The screening runner separates acquisition from analysis:

```text
deterministic pattern cases
        |
        v
step pattern -> confirmed buffer swap -> settle -> acquire newer camera frame
        |
        v
lossless RAW image + capture_manifest.json
        |
        +------------------------------+
        v                              |
offline analysis recipes               |
        |                              |
        v                              |
numeric response (.npy)                |
        |                              |
        +--> grayscale / colour -------+
        +--> 50% overlay variants
        |
        v
results_manifest.json
```

This avoids reacquiring the vehicle for every processing or display variant and
prevents a blind Cartesian product of arbitrary parameters. Multi-frame recipes use
ordered states from the same pattern group.

Presentation is tracked by settings revision. A capture case must complete its
own buffer swap before its settle interval starts. Initialization/render failure,
loss of visible output, or a five-second initial presentation timeout stops
capture and marks its manifest as an error.

## Acquisition lifecycle

For real hardware, only the source worker may own pypylon objects. Normal camera
parameter configuration belongs in Basler pylon Viewer; Defleco LAB keeps only the
acquisition behavior it actually needs.

1. Discover and explicitly select the camera.
2. Open the camera and prepare temporary free-running acquisition state where
   required.
3. Acquire owned image copies into bounded history and recorder paths.
4. Feed the selected history window into the numeric pipeline.
5. Stop grabbing, restore temporary acquisition state where practical, report
   restore failures, and close.

Persistent camera/network operations are absent by design.

## Context and roadmap

The persistent implementation/status tracker is
[development-plan.md](development-plan.md). The operator-oriented first-use guide is
[operator-guide.md](operator-guide.md).
