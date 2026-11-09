# Surface LAB filters

Implemented: **50 registry entries** in eight families. Sobel/Scharr X/Y/magnitude
and feature polarities are parameters rather than separately counted algorithms.
Unsharp/high-boost share one entry. Notch and wavelet filters are not implemented.

The existing Defleco filters supply DoG, local residual, gradients, Gabor,
Structure Tensor and directional residual. The read-only reference was
`CZPavel/PEKAT-Assistant-3.0`, `lab_enhancement.py`, `lab_candidates.py` and
`knowledge/curated/vision_algorithms`: directional morphology and LAB luminance
concepts are adapted to normalized/high-bit input; candidate classification,
threshold decisions and PEKAT runtime/context mutations are not imported.

Surface preview processes at most one megapixel. Native ROI/full-frame processing
is explicit and pinned until Resume preview. CPU scale limits bound kernels,
orientations and ridge scales, but a large native image may still take seconds.
The worker keeps one pending request and one completed result. Full-source
normalization/display buffers still consume memory proportional to source size.
ROI crops have their own filter boundary context and need not equal a crop from a
full-frame filter. Preview pixels are stretched to the shared source-coordinate
canvas for navigation; this display resampling does not add numerical detail.

Experimental single-frame exploration, not a calibrated defect classifier. Responses are float32 grayscale in method-specific units and may be signed; do not compare their raw amplitudes across methods. Enhancement methods also return `intermediates["enhanced"]` in [0,1] when intermediates are enabled. Input is never modified.

`normalize_input`: uint8 and uint16 use the full dtype range (255 / 65535), finite float [0,1] remains unchanged, other nonconstant float uses image min–max, and an out-of-range float constant is clipped. Diagnostics record dtype, original extrema and the normalization. For consistent float ROI/full-frame scaling normalize the full image before cropping. Integer normalization does not depend on ROI. Min–max on other floats is image-dependent and loses absolute scale. BGR inputs become gray for detection; LAB CLAHE works on luminance and preserves BGR output. All methods operate on normalized values, not sensor DN.

Parameters are validated at construction: `sigma` 0.2–20 px, `large` 0.4–40 px and greater than sigma, `size` odd integer 3–63 px, `radius` integer 1–15 px, `angle` 0–179 degrees, `orientations` integer 2–16, `period` 3–64 px, `amount` 0–4, `offset` -1–1, `low_cutoff`/`high_cutoff` 0.001–0.5 cycles/pixel (low < high), `gamma` 0.1–5, `clip` 0.1–10, `grid` integer 2–32, `threshold` 0.01–0.9, and `polarity` signed/absolute/bright/dark (ridge filters accept absolute/bright/dark). Signed maps default to signed; bright and dark produce nonnegative maps. `output` selects magnitude, gradient_x or gradient_y. Defaults are listed below. Scale is in pixels; performance depends strongly on image dimensions and chosen scale. Local/FFT boundary artifacts, saturated pixels, noise and illumination can create false responses. CLAHE/equalization/median/entropy/LBP quantize to 8 bits. Presets are starting points, not validated decisions.

Ridge filters use [scikit-image filters](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html) at sigma and twice sigma. Frangi is gated by the corresponding Sato polarity response because scikit-image 0.26 can produce a response for opposite polarity on an ideal line with zero smaller eigenvalue; it is therefore not an unmodified Frangi map. Hessian and directional derivative return curvature/gradient magnitude or selected sign, while Retinex and FFT high-pass are signed. Sobel and Scharr axis outputs are signed. `absolute` on ridge filters takes the maximum of both polarities.

| ID (prefix `surface_`) | Purpose | Parameters (defaults) | Method-specific limitation |
|---|---|---|---|
| `brightness` | Linear brightness and offset adjustment. | amount=1.0, offset=0.0 | Clips highlights. |
| `gamma` | Nonlinear tonal adjustment. | gamma=1.0 | Changes tone rather than detecting defects. |
| `clahe` | Local grayscale contrast. | clip=2.0, grid=8 | Tile boundaries and noise may be amplified. |
| `equalize` | Global grayscale contrast. | none | Global histogram may exaggerate noise. |
| `stretch` | Suppress extreme intensity outliers. | none | Uses 1st/99th percentiles; constant images become zero. |
| `log` | Compress highlights and lift dark areas. | amount=1.0 | Brightness compression, not a defect score. |
| `lab_clahe` | Color-preserving local luminance contrast. | clip=2.0, grid=8 | LAB conversion/8-bit luminance quantization can alter colors. |
| `gaussian` | Suppress fine random noise. | sigma=2.0 | Blurs small defects. |
| `median` | Suppress impulse noise. | size=15 | 8-bit conversion; broad kernels remove detail. |
| `bilateral` | Smooth while preserving stronger edges. | sigma=2.0, amount=1.0 | Can flatten weak texture. |
| `unsharp` | Enhance fine detail. | sigma=2.0, amount=1.0 | Amplifies noise and clips. |
| `local_residual` | Deviation from Gaussian background. | sigma=2.0, polarity=signed | Scale determines rejected background. |
| `dog` | Band-pass structures between two scales. | sigma=2.0, large=8.0, polarity=signed | Scale band can miss features outside it. |
| `log_response` | Second derivative at selected scale. | sigma=2.0, polarity=signed | Sensitive to noise and boundaries. |
| `shading` | Divide by slowly varying illumination. | sigma=2.0 | Division limited near black; clipped output. |
| `retinex` | Log-domain illumination suppression. | sigma=2.0 | Signed log residual; sensitive near black. |
| `homomorphic` | Frequency-domain illumination suppression. | sigma=2.0 | Periodic FFT boundary assumptions. |
| `top_hat` | Bright features smaller than footprint. | size=15 | Only features smaller than footprint. |
| `black_hat` | Dark features smaller than footprint. | size=15 | Only features smaller than footprint. |
| `morph_gradient` | Local bright-to-dark range. | size=15 | Responds to both edge polarities. |
| `directional_top_hat` | Bright elongated features at chosen angle. | size=15, angle=0.0 | The footprint angle is not necessarily the detected feature angle; a perpendicular footprint highlights a long line. |
| `directional_black_hat` | Dark elongated features at chosen angle. | size=15, angle=0.0 | The footprint angle is not necessarily the detected feature angle; a perpendicular footprint highlights a long line. |
| `multiscale_residual` | Maximum deviation across three scales. | sigma=2.0, polarity=signed | Maximum across scales amplifies noise. |
| `sobel` | First derivative edge strength or signed axis. | sigma=2.0, output=magnitude | Edge magnitude is not a defect label. |
| `scharr` | Rotationally balanced edge strength or signed axis. | sigma=2.0, output=magnitude | Edge magnitude is not a defect label. |
| `laplacian` | Second derivative edge response. | sigma=2.0 | Second derivative amplifies noise. |
| `canny` | Thresholded thin edges. | sigma=2.0, threshold=0.15 | Thresholded output discards response magnitude. |
| `steerable` | Gradient response at selected angle. | sigma=2.0, angle=0.0, polarity=signed | Sign depends on angle and edge direction. |
| `hessian` | Largest curvature with polarity. | sigma=2.0, polarity=signed | Single scale; responds to any curvature. |
| `frangi` | Multiscale vesselness/ridge response. | sigma=2.0, polarity=absolute | Sato polarity gate; not pure Frangi. |
| `sato` | Multiscale tubeness/ridge response. | sigma=2.0, polarity=absolute | Long structures need appropriate scales. |
| `meijering` | Multiscale neuriteness/ridge response. | sigma=2.0, polarity=absolute | Long structures need appropriate scales. |
| `gabor` | Oriented frequency-selective response. | sigma=2.0, period=12.0, orientations=8 | Response depends on period/orientation and kernel size. |
| `tensor` | Local orientation irregularity. | sigma=2.0 | Orientation residual can respond to normal bends. |
| `vector_residual` | Suppress smoothly varying gradient field. | sigma=2.0 | Suppresses slowly varying gradients, not all pattern lines. |
| `directional_residual` | Deviation from best oriented local line. | size=15, orientations=8 | Experimental orientation-binned approximation. |
| `fft_highpass` | Suppress low spatial frequencies. | sigma=2.0 | Periodic FFT boundary assumptions. |
| `entropy` | Local intensity complexity. | radius=4 | 8-bit local histogram; texture is not a defect. |
| `lbp` | Eight-neighbor texture code. | radius=4 | Uniform LBP codes are categorical, not a calibrated magnitude. |
| `local_std` | Local intensity variation. | size=15 | Noise also increases variation. |
| `local_contrast` | Local intensity range. | size=15 | Outliers can dominate range. |
| `preset_scratches` | CLAHE followed by multiscale dark ridge. | clip=2.0, grid=8, sigma=2.0 | Assumes dark elongated scratches. |
| `preset_defects` | CLAHE followed by local residual. | clip=2.0, grid=8, sigma=2.0 | Texture and CLAHE noise also respond. |
| `nlmeans` | Suppress repeated-pattern and random noise. | amount=1.0 | 8-bit conversion and bounded fixed search window; can remove fine texture. |
| `median_residual` | Deviation from local median. | size=15, polarity=signed | 8-bit median background; signed response. |
| `multiscale_retinex` | Log residual averaged over three scales. | sigma=2.0 | Three log scales; signed response and sensitivity near black. |
| `open_close_residual` | Signed deviation from morphological background. | size=15, polarity=signed | Signed difference from average opening and closing; footprint controls features. |
| `multiscale_top_hat` | Bright features over three footprint scales. | size=15 | Maximum response at three bounded scales; noise can increase. |
| `multiscale_black_hat` | Dark features over three footprint scales. | size=15 | Maximum response at three bounded scales; noise can increase. |
| `fft_bandpass` | Select spatial frequency band in cycles per pixel. | low_cutoff=0.02, high_cutoff=0.15 | Hard radial cutoffs can ring; periodic FFT boundary assumptions. |
