# Surface screening

Static-light batch enhancement uses one fresh, immutable live Basler frame at
full resolution. It does not cycle physical patterns or classify defects.
`surface` includes all 50 registry entries once; `surface_extended` includes
standard and boosted variants of each. Parameters come from the registry and
versioned screening recipes, never from manual Surface LAB controls.

## Recipe version 1

Standard recipes use registry defaults. Boosted recipes override only parameters
present on the method:

| Parameter/method | Boosted value |
|---|---|
| sigma, large | Twice the standard value within schema bounds; large remains above sigma |
| size, radius | 31 px, 8 px |
| clip, amount | 4, 2 |
| Brightness | amount 1.4, offset 0.05 |
| Gamma | gamma 0.6 |
| Log compression | amount 3 |
| Canny | threshold 0.075 |
| Gabor, directional residual | additionally orientations 16; period and angle unchanged |
| FFT band-pass | low cutoff 0.01, high cutoff 0.25 cycles/pixel |
| Equalize, stretch | Same numerical output; enhanced-to-original difference multiplied by 1.5, explicitly display-only |

Other parameters remain unchanged. Changing spatial support is an exploratory
recipe rather than a guaranteed improvement in defect visibility. LBP outputs
categorical texture codes; its heatmap is not defect strength.

## Data and display

The capture workspace preserves the original bit depth and frame metadata.
Every processing pass has a unique result directory, incremental manifest and
HTML gallery with family/method/variant filters. Each completed variant contains
Enhanced and Enhanced + heatmap PNGs plus JSON metadata; optional NPY retains
the unmodified float32 numerical response. Result PNGs use RGB8 display mapping.

The heatmap uses the Surface LAB/Surface Compare settings captured at the start of
the run: overlay base (enhanced or original), colormap, opacity, polarity, range
mode and associated values. In auto and percentile modes, resolve the display
range from each method's standard response; use that same fixed range for its
boosted response. Fixed mode uses the specified range. Constant maps receive a
finite nonzero display interval. Alpha is response-dependent and capped by the
selected opacity. When enhanced is selected as the base, it matches the standalone
Enhanced export, including any display-only enhancement gain. Signed numerical
response values remain separate from the display mapping.

The enhanced overlay base is the default for a new Surface LAB session. Older
presets without a base or colormap retain original/Turbo. When the backend processes
an older Surface capture without stored settings and no explicit settings are
provided, it uses the earlier absolute response, 1st/99th percentile, 50% opacity,
enhanced-base rendering. A GUI offline rerun snapshots the settings selected in
Surface LAB/Compare when it starts, including for an older capture.

Each derived-result filename includes the run identifier, method ID, standard/boosted
variant and output type. The two PNGs, optional NPY and JSON sidecar can therefore
be moved into a single folder while retaining their identities. Deflectometry
screening filenames likewise retain case, recipe and method identification.

## Execution and failure behavior

Only one screening job runs at a time. A live run waits up to five seconds for
a newly acquired frame; RAW encoding and sequential full-resolution processing
run on a worker. Stop is cooperative between variants, including a check after
the current calculation/export. Closing the application cancels and joins the
worker; it never destroys a running QThread.

Method failures are recorded and processing continues. Acquisition or file-write
failure stops the run. Terminal outcomes are completed, completed with errors,
cancelled or error; completed files and their gallery remain available.
Reprocessing a Surface capture can select standard/extended and NPY again without
changing its original or previous results.

Synthetic fixture checks and a bounded camera smoke test establish only the
paths actually exercised. Defect visibility, optical quality and production
performance require a separate evaluation on labelled real surface images.

## Verification on 2026-11-09

The automated tests exercise complete standard/extended sweeps on synthetic
Mono16/BGR fixtures, with 100/200 result PNGs, shared ranges, lossless input,
optional response arrays, method/IO errors, reruns, cancellation and Qt shutdown.

A bounded physical smoke test used one explicitly matched Basler
`a2A2448-23gmBAS` (`BaslerGigE`), pypylon 26.6 and pylon Runtime 12.2.0.1265.
Through the existing MainWindow acquisition path it captured a fresh
2448×2048 Mono8 frame, processed three full-resolution variants and saved six
PNGs. Live acquisition and a Qt heartbeat continued; cooperative Stop, partial
gallery availability and clean camera/worker shutdown passed. The test did not
run all 50 methods on hardware, assess actual scratches or establish throughput.
