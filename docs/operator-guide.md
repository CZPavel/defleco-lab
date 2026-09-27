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

Start around a period of **20-30 display pixels** and adjust based on the real
reflection geometry.

Modes:

- **Static**: fixed pattern.
- **Stepped**: each Next step changes the angle/phase by a known increment.
  This is the preferred basis for reproducible automatic experiments.
- **Continuous rotation**: useful for visual screening.

Esc closes the full-screen pattern output.

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
4. compare Scharr vector residual;
5. compare Structure Tensor orientation residual;
6. compare the two line-suppressed tensor outputs;
7. for a pattern sequence compare Frame Difference and Temporal Statistics.

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
