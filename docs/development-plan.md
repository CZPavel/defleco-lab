# Defleco LAB development plan and context tracker

This file is the persistent implementation tracker for the laboratory application.
It records what is already implemented, what is intentionally deferred, and the
reasoning behind the next steps so a later development session does not have to
reconstruct the same context.

Status legend:

- [x] implemented in the repository
- [~] implemented as a usable first version, still needs physical validation/refinement
- [ ] planned / not implemented yet

## Current experimental objective

The immediate goal is not calibrated 3D metrology. It is to find combinations of
display pattern + image-processing response that make local paint/body-surface
deformations conspicuous and repeatable on a real stationary vehicle.

Observed during the first physical screening:

- fine rotating line/stripe patterns looked useful;
- fine checkerboard looked useful;
- circular / especially a visually clean single fine spiral looked useful;
- for single-frame processing, Sobel/Scharr gradient gave the clearest first visual impression;
- for multi-frame processing, Frame Difference and Temporal Statistics looked promising;
- colour response maps were easy to interpret visually;
- approximately 50% response overlay on the original image was useful;
- grayscale response is still important as the clean numerical/processing representation;
- the vehicle was stationary during these observations, so motion compensation was not required.

These observations are experimental notes, not yet a ranking of methods.

## Architecture decision

The analysis path is deliberately split into four layers:

1. **Pre-processing**: transforms the camera image before a method.
2. **Analysis method**: Scharr, Structure Tensor, Gabor, Frame Difference, etc.
3. **Response post-processing**: transforms the numeric method response.
4. **Display**: grayscale/colour/range/alpha presentation only.

The display layer must never be fed back into numerical processing. Colour maps and
alpha overlays are for a human observer only.

## Implemented foundation

### Acquisition and replay

- [x] bounded Basler live acquisition;
- [x] explicit camera selection;
- [x] raw session recording and replay;
- [x] image-sequence source;
- [x] latest-wins processing so heavy methods do not build unbounded GUI latency;
- [x] camera-specific processing code remains outside the processing methods.

### Camera UI simplification

- [x] camera exposure/gain/ROI/FPS editors removed from the Defleco LAB operator workflow;
- [x] camera parameters remain the responsibility of Basler pylon Viewer;
- [x] Defleco LAB keeps only camera discovery, acquisition and read-only runtime status;
- [ ] after physical validation, consider whether any single acquisition-only control is actually needed.

### Integrated pattern output

- [~] full-screen pattern output integrated into Defleco LAB;
- [~] selectable output display;
- [~] fine stripes;
- [~] checkerboard;
- [~] single spiral;
- [~] concentric rings;
- [~] binary and sinusoidal carrier;
- [~] period/fineness;
- [~] angle and phase;
- [~] brightness and inversion;
- [~] static mode;
- [~] reproducible stepped mode;
- [~] continuous rotation/phase mode;
- [ ] physical verification on the actual 4K display;
- [ ] verify Windows multi-monitor full-screen placement on the target PC;
- [~] record OS-reported display refresh rate and per-case pattern-to-capture delay; true VSync/photodiode timing measurement remains pending;
- [ ] optional photodiode/sync patch if timing uncertainty later matters.

### Pattern-generator source continuity

- [x] the standalone `GPixel_Deflecto_Dynamic_Pattern_Generator_PoC_V02.html`
      is treated as an existing design/source to reuse, not something to rediscover;
- [~] the integrated app currently covers the pattern families already most useful
      in the first physical test: stripes, checkerboard, spiral and rings;
- [ ] migrate additional V02 families only when useful for screening: composite X+Y,
      nested squares, squircle, counter-spiral, starburst, speckle and solid;
- [ ] consider V02 sync patch, breathing and auto-cycle concepts when timing or
      broader pattern screening becomes the actual experiment bottleneck.

### Contextual analysis UI

- [~] pipeline displayed as Pre-processing -> Analysis method -> Post-processing -> Display;
- [~] per-method parameter editor remains metadata-driven;
- [~] method parameters can declare conditional visibility dependencies;
- [~] irrelevant conditional parameters are hidden;
- [~] motion-compensation block is hidden for methods that do not support it;
- [~] frame stride is hidden when it is not relevant;
- [~] tooltips/help text added to the new controls;
- [ ] physical usability pass with the operator at the camera/display workstation;
- [ ] reduce or rename controls that still prove ambiguous during real use.

### Pre-processing

- [~] processing scale 100/50/25%;
- [~] contrast;
- [~] brightness;
- [~] gamma;
- [~] Gaussian blur;
- [~] optional CLAHE;
- [ ] decide from real data which controls deserve to remain exposed by default;
- [ ] optional saved pre-processing recipes.

### Single-frame methods

Existing:

- [x] Sobel / Scharr gradient;
- [x] Laplacian;
- [x] Difference of Gaussians;
- [x] Local Background Residual;
- [x] Structure Tensor;
- [x] Gabor bank;
- [x] Directional Residual.

New defect-oriented outputs:

- [~] Scharr/Sobel vector_residual: subtracts a locally smooth Gx/Gy field;
- [~] Scharr/Sobel orientation output;
- [~] Structure Tensor linearity_suppressed: suppresses pure one-direction line response;
- [~] Structure Tensor junction_response: requires significant energy in both tensor eigen-directions;
- [x] Structure Tensor orientation residual already available.

These outputs are hypotheses for physical screening. They are not claimed as
calibrated defect metrics until tested on real body panels.

### Multi-frame methods

- [x] Frame Difference;
- [x] Temporal Statistics;
- [x] Temporal Median Residual;
- [x] Farneback dense flow;
- [x] DIS dense flow;
- [x] local phase correlation / DIC-like grid;
- [x] temporal response helper.

Important follow-up:

- [ ] add true method-response sequence fusion (for example max/percentile of
      Farneback local-residual maps) instead of only fusing raw frame differences;
- [ ] evaluate Frame Difference and Temporal Statistics systematically against the
      same saved pattern sequences;
- [ ] keep motion compensation disabled for the stationary-car screening dataset;
- [ ] evaluate moving-vehicle compensation only after static pattern/processing
      combinations are understood.

### Response post-processing

- [~] local smooth-background subtraction;
- [~] absolute response;
- [~] response smoothing;
- [~] numeric response gain;
- [ ] decide which operations should become reusable ordered filter chains;
- [ ] optional morphology/thresholding only if real data shows a clear need.

### Display

- [x] grayscale display;
- [x] colour maps;
- [x] percentile/manual/automatic display range;
- [x] original/response alpha overlay;
- [~] default overlay changed to 50% based on the first physical observation;
- [x] display mapping remains separate from numeric response data.

## Automated screening experiment

The intended experiment is deliberately split into acquisition and offline analysis.

### Phase A - pattern acquisition

- [~] Experiment workspace / output folder selection;
- [~] deterministic case IDs;
- [~] pattern manifest stored with every captured case;
- [~] stepped capture: set known pattern state -> wait for display settling -> acquire a newer camera frame;
- [~] configurable settling time in milliseconds; refresh-count timing remains future work;
- [ ] first screening set:
  - fine stripes at several periods and orientations;
  - checkerboard at several cell sizes/orientations;
  - single spiral at several periods/phases;
  - rings at several periods/phases;
- [ ] short continuous-pattern recording for comparison with stepped acquisition;
- [~] preserve captured camera frames losslessly as RAW experiment PNG files plus camera metadata.

For the stationary-car PoC, stepped acquisition is preferred because it gives
known pattern state without requiring precise display/camera hardware triggering.

### Phase B - offline processing sweep

- [~] run saved raw cases through a bounded initial set of single-frame recipes;
- [~] run ordered pattern groups through initial multi-frame Frame Difference / Temporal Statistics recipes;
- [x] do not generate a blind Cartesian product of every numeric parameter;
- [~] use named, physically meaningful recipes/ranges;
- [x] compute each numeric response once;
- [x] render presentation variants from that response without recomputing analysis.

Initial display variants to export:

- [~] grayscale response;
- [~] colour response;
- [~] grayscale 50% overlay;
- [~] colour 50% overlay;
- [~] full response view through full grayscale/colour response exports.

### Phase C - result review

- [~] identifiable filenames and JSON manifests; CSV summary remains optional;
- [ ] gallery/filter view by pattern, method and recipe;
- [ ] compare 2/4 cases side by side;
- [ ] optional Defect ROI and Healthy Reference ROI;
- [ ] simple robust contrast score between defect and reference ROI;
- [ ] score is navigation/triage only, not an automatic declaration of the best method;
- [ ] allow manual favourites/notes so human visual judgement is retained.

## Scientific method extensions to consider after the first screening dataset

Do not implement these merely to increase method count. Add them when the captured
data makes the comparison useful.

- [ ] MID / multi-pattern modulated-intensity decoding;
- [ ] phase-shift / wrapped-phase response for sinusoidal patterns;
- [ ] Windowed Fourier / local phase-gradient analysis;
- [ ] optical-flow response fusion across a known pattern sweep;
- [ ] more complete DIC/speckle processing if speckle becomes promising;
- [ ] pattern-aware grid/lattice residual for checkerboard;
- [ ] pattern-aware phase/orientation curvature for stripe/spiral carriers.

## Definition of useful progress for this project

A change is useful when it improves one of these decisions:

- Which physical pattern reveals a known surface defect?
- Which numeric response suppresses predictable pattern structure while preserving
  the local disturbance?
- Which processing settings are stable enough to reproduce the observation?
- Can the same conclusion be reached from recorded data without repeating acquisition?
- Does a new method provide information not already visible in a simpler response?

A green CI run, commit SHA or successful camera connection is supporting evidence
only; it is not evidence that a deflectometry method is useful.

## Next recommended implementation step

1. Physically validate the integrated pattern window and the new pipeline controls.
2. Save a small real dataset for stripes, checker and single spiral on one known defect.
3. Compare:
   - Scharr magnitude;
   - Scharr vector residual;
   - Structure Tensor orientation residual;
   - Structure Tensor linearity-suppressed/junction response;
   - Frame Difference;
   - Temporal Statistics.
4. Use those observations to freeze the first Screening Recipe Set.
5. Implement the automated acquisition + offline processing runner against that
   recipe set, rather than guessing a huge parameter matrix first.
