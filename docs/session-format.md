# Session format

```text
session_YYYYMMDD_HHMMSS/
  session.json
  metadata.csv
  frames/frame_000000.png
  results/
```

Raw frames are stored separately from derived results. `metadata.csv` carries frame index, host timestamp, optional camera timestamp and BlockID, exposure, gain, dimensions, pixel format, estimated one-axis motion, motion quality, cumulative position, and relative filename.

Unique camera identifiers and IP addresses are excluded by default. A session is written with a `.partial` suffix and renamed only after clean completion. Replay reconstructs camera-independent `FramePacket` values and runs raw frames through the currently selected algorithm.
