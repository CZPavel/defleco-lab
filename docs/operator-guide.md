# Defleco LAB quick operator guide

This guide is written for the current physical PoC: one Basler camera, one external
pattern display and a stationary painted vehicle/body panel.

## What the four analysis sections mean

### 1. Pre-processing

Changes the camera image **before** analysis.

Start with:

- Processing scale: **50%**
- Contrast: **1.0**
- Brightness: **0**
- Gamma: **1.0**
- Blur: **0**
- CLAHE: **off**

Only change these when the raw image needs a specific experiment. Keeping the
baseline unchanged makes comparisons easier.

### 2. Analysis method

This is the actual numerical feature/response calculation.

For the first physical screening:

- **Sobel / Scharr Gradient**
  - Operator: Scharr
  - Output: magnitude
  - then compare vector_residual
- **Structure Tensor**
  - start with orientation_residual
  - compare linearity_suppressed and junction_response
- Multi-frame:
  - Frame Difference
  - Temporal Statistics

Only parameters relevant to the selected method/output should be visible.

### 3. Response post-processing

Changes the numeric response **after** the analysis method.

Start with everything neutral:

- Local background sigma: **0**
- Absolute: **off**
- Response smoothing: **0**
- Gain: **1.0**

A useful first experiment after Scharr is local-background subtraction plus
absolute response. Do not change many values at once when trying to understand why
a result improved.

### 4. Display / overlay

This changes only what the operator sees. It does **not** change the stored numeric
response.

Useful starting point from the first physical test:

- Colour map: **on**
- Range: **Percentile**
- Response alpha: **50%**

Turn colour map off for a grayscale view. A grayscale display may be easier to
judge for subsequent contrast processing, but colour can make small response
differences more obvious to a human.

## Pattern generator

The integrated Pattern generator opens a borderless pattern surface on the selected
display.

Recommended first candidates:

1. Fine stripes
2. Checkerboard
3. Single fine spiral
4. Concentric rings

The current physical test showed useful structure starting around **50 display
pixels**, so 50 px is now the manual default and Quick screening uses 48/64 px.

Modes / dynamics:

- **Static**: fixed pattern.
- **Stepped**: Next step can increment angle or phase.
- **Continuous**: rotation, carrier phase sweep and period breathing can be combined.
- **Virtual phase tilt X/Y** changes phase progressively across the screen and acts
  as a software-only two-axis warp.
- Radial/spiral families also expose X/Y centre position.

Esc closes the full-screen pattern output.

## Recording raw sessions

1. Start **Live** first. Record intentionally refuses to start when no new frames
   are arriving.
2. Press **Record** and choose a parent folder. Defleco creates its own timestamped
   session subfolder.
3. Confirm the visible red **REC** line: it shows elapsed time, frames actually
   written, queue depth, drops and the path.
4. Press **Stop record**. The status line reports the exact saved frame count/path.
   A 0-frame session is explicitly reported as such.

## Analysis ROI and Motion ROI

ROI controls are available both in the Input panel and the **ROI tools** toolbar.

- **Add Analysis ROI** immediately creates a cyan box in Original view. Drag the box
  to move it and a white corner handle to resize it. Enable *Process enabled Analysis
  ROIs only* when you want analysis restricted to those areas.
- **Add Motion ROI** creates an orange box. Auto motion compensation uses these
  regions to estimate vehicle translation.
- For a moving vehicle, put Motion ROI on texture/features that move with the car
  but are not dominated by the changing reflected display pattern. Start with a
  static display pattern for the first moving-vehicle test.
- Delete the selected ROI with the toolbar action or Delete key; Clear ROIs removes
  all boxes.

## Camera

Defleco LAB deliberately does not duplicate normal Basler camera parameter
configuration. Set exposure, gain, ROI, pixel format and related parameters in
**pylon Viewer**. Defleco LAB discovers/selects the camera and acquires frames.

## Recommended first comparison

Keep the vehicle, display and camera fixed.

For each of fine stripes, checkerboard and single spiral:

1. visually find a pattern fineness that makes the known defect visible;
2. record a short raw session;
3. compare Scharr magnitude;
4. compare **Fringe Line Geometry (vector)** first with Vector lines, then Geometry residual;
5. for a pattern sequence compare Frame Difference and Temporal Statistics;
6. treat the older tensor/vector-residual/directional outputs as optional experimental
   references rather than recommended defaults until real data proves otherwise.

Do not interpret a more colourful image as automatically being a better detector.
The useful result is the one where the known defect becomes more localised or
distinct while ordinary smooth curvature/pattern structure is suppressed.


## Automated screening

Use **Quick screening** first. It intentionally focuses on the pattern families and
responses that were most promising in the first physical test instead of multiplying
every possible parameter.

1. Select **Basler** and start **Live** acquisition.
2. Choose the external pattern display.
3. Open **Screening experiment** and choose a parent folder.
4. Keep the default settling delay for the first run.
5. Start **Run pattern capture sweep**.

The application then:

- shows one deterministic pattern state;
- waits for the configured settling time;
- waits for a newer camera frame;
- saves the lossless captured image and pattern/camera metadata;
- advances to the next pattern state.

After capture, automatic offline processing can produce named response recipes plus:

- grayscale full response;
- colour full response;
- grayscale 50% overlay;
- colour 50% overlay;
- numeric float response data.

The result folder also contains **screening_results.html**. Use **Open results** to
browse the generated combinations and filter them by pattern group or processing
recipe.

**Extended screening** additionally reuses more families from the standalone V02
generator and slower processing recipes. Run it only after Quick screening confirms
that the display/camera geometry and pattern scale are sensible.
