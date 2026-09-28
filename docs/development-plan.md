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

## Reliability review 2026-09-27

Static review of the integrated laboratory workflow focused on acquisition ordering,
bounded buffers, processing hand-off, screening capture freshness, recorder behaviour
and clean shutdown. No new analysis features were added.

Findings / actions:

- [x] camera SDK delivery changed from `LatestImageOnly` to `OneByOne`; Defleco's
      own bounded mailbox now remains the observable place where backlog/drop
      handling occurs, which is safer for temporal methods and recording;
- [x] automated screening now waits through the configured pattern-settle interval,
      establishes a post-settle frame baseline, then requires one strictly newer
      camera frame before saving a case;
- [x] recorder drop count is reported when a manual recording is closed;
- [x] application shutdown now waits for an active offline screening worker instead
      of allowing a running QThread to be destroyed with the main window;
- [x] unexpected offline screening exceptions are surfaced to the operator instead
      of silently terminating the worker;
- [x] live camera mailbox, live history, processing requests/results and recorder
      queues remain bounded;
- [~] current raw-memory bounds are intentionally conservative rather than minimal:
      64 live frames, 16 camera-mailbox frames and up to 64 recorder copies. At
      multi-megapixel Mono8 resolution this can consume several hundred MB before
      method working buffers; physical monitoring on the target PC is still useful;
- [~] continuous 4K pattern animation remains GUI-thread rendered. The default 50%
      render scale reduces load, but physical testing should watch GUI responsiveness
      and camera-buffer drops during continuous patterns;
- [ ] do not tune buffer sizes or further optimise pattern rendering without evidence
      from the target workstation; the existing bounds are safe enough for the
      current ~6 FPS PoC and preserve useful temporal history.

## Camera connection reliability review 2026-09-28

Review driven by an older observed failure mode: live image could freeze after a
long run, and an immediate reconnect from Defleco could fail until the camera had
been opened/closed once in pylon Viewer.

Findings / actions:

- [x] Basler acquisition uses a single dedicated worker thread and a bounded mailbox;
- [x] pylon delivery is OneByOne so frame loss is handled/observed by Defleco rather
      than silently inside LatestImageOnly;
- [x] a 5 s no-frame watchdog converts a silent stream stall into a controlled
      acquisition error followed by camera teardown;
- [x] camera teardown now clears the InstantCamera reference even when transport
      calls such as IsGrabbing/IsOpen/Close raise after a device/network fault;
- [x] starting a new camera session is blocked if the previous worker did not stop
      within its timeout, preventing two workers from competing for the same device;
- [x] a newly opened Basler session clears live temporal history so multi-frame
      methods cannot combine frames from before and after reconnect;
- [x] finished acquisition workers release the GUI reference and stop the camera
      polling timer; reconnect therefore creates a fresh worker/device object;
- [x] late queued signals from an older worker are ignored once a newer camera
      session exists;
- [x] camera status exposes frame age, retrieve timeout count and application mailbox
      drops to distinguish processing load from acquisition/transport stalls;
- [~] manual reconnect should now be sufficient after a recoverable stream error;
      physical long-duration verification on the target GigE camera is still required;
- [ ] if the same symptom survives these changes, capture the exact pypylon error and
      GigE transport statistics before adding automatic reconnect/reset logic.

## GUI usability review 2026-09-28

Review driven by physical use on an FHD workstation where the application could
not be reduced enough vertically and the camera image could remain clipped after
window/maximize/fullscreen geometry changes.

Findings / actions:

- [x] initial window size is now derived from the available logical desktop size,
      so Windows DPI scaling cannot make the default 1500x900 logical window taller
      than an FHD work area;
- [x] Input/Camera, Pattern generator and Screening experiment are tabbed as
      alternative left-side workflows instead of being stacked vertically;
- [x] all three left workflows are scrollable, so their content no longer imposes
      a large minimum window height;
- [x] long camera/method/display combo-box contents may shrink instead of forcing
      oversized side docks;
- [x] the screening action buttons use a compact 2x2 layout instead of one very
      wide row;
- [x] ImageViewer Fit mode no longer relies on QGraphicsView.fitInView with
      automatic scrollbars; the scale is computed explicitly and fit-mode
      scrollbars remain off;
- [x] fitted viewers refit after viewport resize, tab activation and
      fullscreen/normal transitions, preventing the lower part of the image from
      remaining outside the viewport;
- [x] 1:1/manual zoom re-enables scrollbars and panning only when the image is
      actually larger than the viewport;
- [x] focused GUI regression tests cover fit-after-resize, FHD-sized layout,
      tabbed/scrollable left docks and DPI-scaled initial-size bounds;
- [~] physical verification is still required on the target Windows/FHD monitor
      because CI uses Qt's offscreen platform and cannot reproduce GPU/display-driver
      window-manager details.

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
      is treated as the existing design/source to reuse, not something to rediscover;
- [~] the integrated renderer now carries the V02 pattern families: stripes,
      checkerboard, circular/rings, composite X+Y, nested squares, squircle,
      spiral, counter-spiral, starburst, speckle and solid;
- [~] pattern-specific controls are contextual (duty/line width, squircle power,
      spiral arms/width, spokes, speckle size/seed) and the main first-test
      families remain visually prioritised through hints and Quick Screening;
- [~] Extended Screening includes a bounded subset of the additional V02 families
      rather than multiplying every pattern by every parameter;
- [ ] migrate V02 centre offsets only if optical setup adjustment needs them;
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
- [~] browsable HTML gallery with filtering by pattern group and processing recipe;
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
