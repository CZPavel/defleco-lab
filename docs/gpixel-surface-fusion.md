# GPixel Surface Fusion

The **GPixel Surface Fusion** tab combines existing Surface filters to make weak
surface structures more visible to a human. It is experimental enhancement, not
automatic detection, probability, neural confidence, annotation or OK/NOK logic.
The seven versioned recipes have stable IDs `fusion_k01`–`fusion_k07` and version
`gpixel-surface-fusion.v1`. Their initial values are hypotheses informed by the
operator's selection of 91 Enhanced PNGs, not independently validated optima.

## First test on this Windows PC

1. Start **Defleco LAB** from the Desktop shortcut or `Start Defleco LAB.cmd`.
2. Open **GPixel Surface Fusion**, then **Open image**. Use an original image,
   preferably the saved screening original rather than an already enhanced PNG.
   Basler, replay, synthetic and image-sequence sources are also supported through
   the existing Input panel. **Use input source** returns from an opened file to
   that shared source; tab navigation preserves the explicitly opened file.
3. Choose **K01 – CLAHE + Directional Top-hat** and leave recommended defaults.
   Compare Original and Result, zoom/pan together, and adjust morphology angle
   to the structures. A line-shaped opening aligned with a scratch can preserve
   it and therefore remove its top-hat response; test other angles as well.
4. Draw one Analysis ROI, then **Process native ROI** for a native-pixel check.
   Select **K07 – GPixel Surface Fusion** to compare its default three branches.
   Toggle each Enabled checkbox or set its weight to zero to isolate contributions.
5. Use **Intensity** to reduce overly strong enhancement; zero returns the
   normalized original exactly. The status reports the final clipped-pixel fraction.
6. Use **Save preset** and **Export result**. Preview/native results remain pinned
   according to the existing Surface workflow; **Resume preview** releases the pin.

The two viewers, file loader, ROI controls, latest-wins worker, heatmap settings,
presets and export reuse the existing Surface implementation. The new tab has an
independent selection/configuration; only the visible workspace processes frames.
Enabled checkboxes hide unused step controls. **Restore recommended defaults**
resets the current recipe's parameters, including intensity.

## Recommended recipes

All included steps start enabled, polarity defaults to `both` where available,
and final intensity defaults to **0.5**, editable from 0 to 2.

| Recipe | Ordered calculation | Main defaults |
|---|---|---|
| K01 | CLAHE → directional white/black top-hat | clip 2, grid 8; size 31 px, angle 0° |
| K02 | CLAHE → multi-scale Hessian | clip 2, grid 8; sigmas [2, 4] px |
| K03 | Median-background subtraction → Hessian | median 31 px, correction strength 1; sigmas [2, 4] px |
| K04 | CLAHE → FFT high-pass | clip 2, grid 8; FFT sigma 4 px, signed response; absolute optional |
| K05 | Local-background subtraction → directional morphology | local sigma 4 px, strength 1; size 31 px, angle 0°, morphology strength 1 |
| K06 | CLAHE → unsharp mask | clip 2, grid 8; amount 2, sigma 4 px |
| K07 | Optional CLAHE → three independent branches → fusion | clip 2, grid 8; Hessian [2, 4] px / weight .40, morphology 31 px at 0° / .35, median residual 31 px / .25; Weighted Sum |

Hessian accepts **1–4 scales from 1 to 8 native pixels**. Its response selector
offers the fused result or one configured scale. Selecting a single scale does
not compute the others. Morphology uses odd sizes 3–63 px and angle 0–179°;
median uses odd sizes 3–63 px. Limits are enforced before processing. Each recipe
exposes only the steps it contains. K07 weights range from 0 to 2.

K03 uses `gray - correction_strength × median_background`. K05 analogously uses
`gray - local_strength × Gaussian_background` before morphology. These signed
float intermediate images are never passed back through source min/max
normalization or converted to an ordinary 8-bit image for the next stage. With
the final analysis step disabled, an enabled residual step can still be viewed.

## Numerical response, fusion and enhanced image

The source is normalized once using the existing Mono/BGR uint8, uint16 and
finite-float contract. Processing uses float32. CLAHE retains its documented
8-bit luminance histogram step. Fusion's median-background calculation instead
uses an exact float median (`scipy.ndimage.median_filter`, nearest boundary),
preserving weak higher-bit-depth differences and avoiding quantization bias on
uniform fields. The original Surface median implementation remains unchanged.

For normalized feature branches, calculate the 99th percentile of absolute
response. Divide by it and clip to [-1, 1]; for sparse maps whose percentile is
near zero use the absolute peak, and return zero when that is also negligible.
Do not subtract a new minimum or stretch a signed map into [0, 1]. Bright selects
positive response, dark selects negative response, and both retains both signs.
Top-hat/black-hat combine as white minus black. Hessian scales are normalized
individually and fused by maximum absolute response, retaining the selected sign.

K07 applies these normalizations independently to its branches. Disabled or
zero-weight branches are not computed. **Weighted Sum** uses the active weights
divided by their total. **Maximum** selects the largest absolute weighted
contribution with its sign, using the same normalized weights. All branches off
produce zero response. A weight change does not change another branch's raw
calculation. Exact normalization factors and effective pixel scales are recorded
in diagnostics. K06 uses its unsharp difference rather than a normalized ridge map.

The enhanced image blends the original toward the preprocessed base plus the
feature response using the final intensity. K06 adds its unsharp contribution
only once. Zero intensity returns the normalized original. Final display bounds
are [0, 1], with the clipped-pixel fraction explicitly reported and exported;
reducing intensity avoids saturation. Color input keeps a recognizable original
color basis. The primary numerical map remains separate from this display image.

## Preview, native resolution and performance

Parameters refer to **native-image pixels**. Preview remains bounded to 1 MP;
its worker supplies the actual raster scale so Fusion adjusts sigmas and kernel
sizes. Odd kernels have a minimum effective size of 3 px and sigmas a lower bound
of 0.2 px. Downsampling, rounding, filtering boundaries and image-dependent
normalization mean preview and native output need not be pixel-identical.
Full-source normalization happens before ROI cropping. A cropped ROI has its own
filter boundary context and need not equal the same crop of full-frame analysis.

Exact float medians in **K03/K07 are expensive**. At multi-megapixel native
resolution a large kernel may take tens of seconds or longer; start with a native
ROI. Calculations stay off the Qt UI thread. Pending jobs/results are bounded and
obsolete configurations are discarded. Cancellation/shutdown waits for the
current non-interruptible kernel; if it exceeds the shutdown timeout, closing is
postponed rather than destroying a running thread. No GPU framework is added.

## Presets, export and screening

Presets and JSON sidecars retain recipe ID/version, enabled and active steps,
all step parameters, branch weights, normalization policy and intensity.
Export also records visualization settings, source filename/frame when available,
ROI, source/processing dimensions, spatial scale and preview/native status.
PNG/TIFF is an RGB8 display image; optional NPY is the numerical float32 response.
Existing files and original photographs are not overwritten.

In **Screening experiment → Plan → GPixel Surface Fusion**, choose a parent folder
and click **Run Fusion screening**. This profile uses the full image currently
selected in the Fusion tab, without requiring a camera or changing patterns.
It evaluates only the seven recommended configurations, independently of manual
parameter edits, sequentially at native resolution. The existing exporter writes
**seven Enhanced PNGs plus seven heatmap-overlay PNGs**, JSON metadata and optional
NPY in a unique run directory. Heatmaps use the Fusion tab's display settings.
There is no Cartesian parameter sweep. Stop and Open results retain partial work;
saved captures can be reprocessed offline. Other screening profiles are unchanged.

## Evidence limits

Synthetic line/spot/gradient fixtures test polarity, finite outputs, shape, source
preservation, branch controls, weights, zero intensity, signed intermediates,
preset reproducibility, stale-result rejection and exports. Regression checks
cover the original Surface and deflectometry paths. These establish software
behavior only, not improved real scratch visibility, detection accuracy or
production throughput. Validate any preferred combination on representative real
originals and inspect noise, halos, clipping and lost weak structures.

Local verification on 2026-11-09: the full suite passed **391 tests**, with Ruff,
compileall, pip dependency checks and the existing synthetic acceptance passing.
An independent comparison of the 50 original Surface methods across float mono,
Mono16 and BGR inputs found agreement within numerical tolerance after shared
kernel extraction. A visible Windows Qt smoke test processed K01 and K07 on a
native 256×256 ROI of a saved real 2448×2048 Mono8 camera image, exported PNG/NPY/
JSON, preserved the selected file across tab changes and shut all workers down.
It reported about 22% and 19% clipping at the experimental default intensity;
that is a reason to reduce intensity on this scene, not evidence of better
defect visibility. No new live-camera or optical-accuracy validation was done.
