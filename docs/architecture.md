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
have hardware-level frame synchronization. Continuous rotation remains available
for visual screening.

The earlier standalone dynamic-pattern PoC remains the design source for timing/sync,
breathing, centre-offset and other future pattern-output features; the integrated
application should continue reusing those established concepts rather than
rediscovering them.

## Automated screening

The screening runner separates acquisition from analysis:

```text
deterministic pattern cases
        |
        v
step pattern -> settle -> acquire newer camera frame
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
