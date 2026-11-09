# Defleco LAB: functionality overview

Defleco LAB is experimental Windows desktop software for comparing image-processing
responses on glossy or specular surfaces. It supports reflected dynamic-pattern
deflectometry experiments and single-frame Surface LAB enhancement under static
light. It is not a calibrated measurement instrument, a production inspection
system, an annotation editor or an automatic OK/NOK classifier.

## Installation and start

Use Windows 10/11 x64 and Python 3.11 x64. The normal Python dependencies are
NumPy, OpenCV, PySide6 and scikit-image; the optional Basler integration uses
`pypylon 26.6` and a compatible separately installed pylon Runtime. Full-screen
pattern rendering requires an OpenGL 3.3 graphics driver; OpenCV CUDA is not
required. Installation, setup scripts, the desktop shortcut, debug launcher and
the safe fast-forward local update are described in the [README](../README.md).

## Sources and acquisition

The **Input / Camera / Session** panel offers four sources:

| Source | Use |
|---|---|
| Synthetic | Generated fringes, checkerboard, grid or speckle for reproducible experiments without a camera. |
| Basler | Discover devices, explicitly select the desired camera and start/stop Live acquisition. |
| Recorded Session | Replay a saved raw session and compare processing methods on the same frames. |
| Image sequence | Load an ordered image folder for playback and analysis. |

Surface LAB also has **Open image** for a single file. Source data is kept separate
from derived results. Basler selection is explicit; the application does not pick
a device by connection order. Normal camera settings, such as exposure, gain, ROI
and pixel format, are configured in pylon Viewer. For live preview the application
requests free-running session acquisition where supported and attempts to restore
temporary settings on disconnect. It does not intentionally save User Sets, force
network settings, change firmware or drive GPIO. See the
[operator guide](operator-guide.md#camera) for the camera workflow.

Live acquisition uses a bounded frame mailbox and history. Processing uses a
latest-wins handoff so slow filters do not create an unbounded queue. The Original
view remains a baseline and does not continuously run the selected expensive
analysis. Raw recording uses a separate bounded path with a visible frame/drop
count. A session stores raw frames and metadata separately from results and can be
replayed; details are in [session format](session-format.md).

## Views and controls

The main image tabs are:

| Tab | Purpose |
|---|---|
| Original | Camera or recorded source baseline. |
| Processed | The selected analysis response after response post-processing and display mapping. |
| Intermediate | Available intermediate output for the selected method. |
| Compare | Linked baseline/reference and processed views for comparison. |
| Motion Debug | Motion-mask/compensation diagnostics when available. |
| Surface Compare | Original and Surface result side by side, with linked zoom and pan. |
| Surface LAB | Surface result alone. |
| GPixel Surface Fusion | Original/result comparison for seven configurable combined enhancement recipes. |

Surface Compare and Surface LAB share one source, selected method, parameters,
display settings and result; switching between them does not rerun a filter.
Viewers support fit, 1:1, zoom and pan. Analysis ROIs limit the applicable analysis;
separate Motion ROIs identify stable features for motion estimation. ROI placement,
editing and differences between them are described in the
[operator guide](operator-guide.md#analysis-roi-and-motion-roi).

## Deflectometry analysis

The processing pipeline is **pre-processing → analysis method → numerical response
post-processing → display**. Pre-processing provides scale, contrast, brightness,
gamma, Gaussian blur and optional CLAHE. Parameter editors are generated from the
selected method's schema and expose relevant controls. Response post-processing
includes local-background subtraction, absolute value, smoothing and gain. Display
mapping supplies grayscale/colormaps, auto/manual/percentile range, response alpha
and shared comparison ranges. Display mapping does not alter the numeric response.

Single-frame methods include Sobel/Scharr gradients, Laplacian, Difference of
Gaussians, local background residual, Structure Tensor, Gabor filter bank,
directional residual and Fringe Line Geometry. Multi-frame methods include frame
difference, temporal statistics, temporal median residual, Farneback and DIS optical
flow, local phase correlation and temporal fusion. Their parameters, outputs and
limitations are detailed in [methods](methods.md).

Optional one-axis motion compensation estimates translations from selected Motion
ROIs, rejects unreliable regions and fuses valid estimates. Historical frames are
aligned into the current image coordinates; validity masks prevent padded areas
from becoming apparent responses. It is experimental and depends on suitable
stable features; see [motion compensation](motion-compensation.md).

## Pattern generator

The **Pattern generator** opens an independent borderless full-screen output on a
selected display. Families include fine stripes, checkerboard, concentric rings,
composite X+Y, nested squares, squircle, single spiral, counter-spiral, starburst,
speckle and solid field. The controls expose appropriate spatial parameters,
binary/sinusoidal carrier, light/dark RGB endpoints, brightness and inversion.

Static mode presents a fixed pattern. Stepped mode advances angle or phase, while
Continuous mode can combine rotation, phase sweep and period breathing. Optional
temporal smoothing softens moving edges to reduce visible strobing. The OpenGL 3.3
renderer uses the selected screen's physical framebuffer and follows buffer swaps;
the CPU renderer remains available as an offline reference. GPU identity and
presentation diagnostics can be saved. A software buffer swap is not evidence of
optical settling, zero physical tearing or hardware camera/display synchronization.
See [operator guide: pattern generator](operator-guide.md#pattern-generator).

## Static-light Surface LAB

The Surface registry contains 50 single-frame methods across brightness/contrast,
smoothing, background removal, morphology, edges/ridges, directional/frequency,
texture and combined-preset families. See the [filter catalogue](surface-filters.md)
for the individual methods, parameters, normalization, polarity and limits.

The schema-driven panel offers enhanced image, grayscale response, heatmap and
overlay outputs. Heatmap settings control colormap, original/enhanced base, opacity,
polarity and auto/fixed/percentile display range. The enhanced base is the new
default; legacy presets without the base/colormap fields retain original/Turbo.
Display settings and the float32 numeric response are separate. Mono/BGR uint8,
uint16 and finite float inputs are normalized according to the documented Surface
contract; selected operations explicitly quantize internally to 8 bits.

The normal Surface preview processes at most one megapixel, while native ROI and
full-resolution processing are explicit actions. The full source is normalized
before ROI cropping. Native results stay pinned until **Resume preview**. A result
can be exported as an RGB8 PNG/TIFF display image with a JSON sidecar and optional
float32 NPY response. Method and display presets can be saved and loaded. Source
files are not overwritten.

## Screening workflows

**Quick screening** and **Extended screening** control the pattern display for a
deterministic set of cases. Each case waits for a presented pattern and settling
interval, then a newer camera frame; the raw image and metadata are saved. The
optional offline processing pass exports recipe responses and grayscale/colour
views and overlays, with a browsable HTML gallery. Extended adds pattern cases and
slower recipes. Pattern colours can optionally follow the selected endpoints.
Output filenames identify the case, recipe and processing method.

**Surface screening** captures one fresh Basler Live frame under the current static
light and processes all 50 standard Surface methods sequentially at full resolution.
**Surface screening Extended** additionally processes one versioned boosted recipe
per method. They produce 100 or 200 PNGs respectively: Enhanced and a heatmap
overlay per variant. The second image uses the shared Surface heatmap settings
snapshotted at run start; the standard response supplies the common display range
for standard/boosted comparisons. The original, JSON metadata and optional float32
NPY responses are saved separately. Each image filename identifies run, method,
variant and output. Stop preserves completed work and its gallery; an existing
Surface capture can be reprocessed offline in a new output directory. Full recipe
and failure details are in [Surface screening](surface-screening.md).

## GPixel Surface Fusion

K01–K07 combine existing filters with per-step switches, configurable parameters,
overall intensity and independent presets. K07 normalizes Hessian, directional
morphology and median residual branches before weighted-sum or maximum fusion.
Signed intermediates stay float; Fusion uses an exact float median and native-pixel
preview scaling. Clipping is reported. Its limited screening profile processes
the original selected in the Fusion tab through seven recommended configurations.
See [Fusion recipes and first test](gpixel-surface-fusion.md). These are exploratory
enhancements, not probabilities or validated defect decisions.

## Verification and remaining limits

The automated test suite and synthetic acceptance exercise the implemented
processing and UI contracts. On 2026-11-09, a bounded Surface-screening smoke test
with one Basler `a2A2448-23gmBAS` / `BaslerGigE`, pypylon 26.6 and pylon Runtime
12.2.0.1265 acquired one fresh 2448×2048 Mono8 image, processed three variants,
saved six PNGs, retained live acquisition and completed a cooperative stop and
clean shutdown. It did not test every method on hardware.

The response of synthetic fixtures, successful camera acquisition or a visually
plausible heatmap does not establish real scratch-detection efficacy, calibrated
accuracy, trigger/PTP behavior, optical synchronization or production throughput.
Evaluate settings on labelled representative real surfaces, lighting and defects.
