# Architecture

Defleco LAB separates acquisition, immutable frame data, processing, recording, and presentation. `FramePacket` is camera-independent and owns its image buffer. A bounded chronological history preserves source-frame spacing, while live processing may discard obsolete requests to avoid unbounded latency. Raw recording has a separate loss-reporting path.

The GUI is schema-driven: each processing plug-in publishes its name, category, frame requirement, help, limitations, references, and editable parameters. New methods therefore do not require method-specific controls in the main window.

For real hardware, only the source worker may own pypylon objects. The adapter performs exact runtime selection, bounded grabs, capability checks before temporary writes, a baseline snapshot, and best-effort rollback. Unique hardware identity is held only in memory and is excluded from sample configuration, sessions, and public reports.

## Lifecycle

1. Discover and explicitly select a source.
2. Open and snapshot allowed temporary camera parameters.
3. Apply explicit session changes with readback.
4. Acquire owned image copies into bounded history and recorder paths.
5. Stop grabbing, restore temporary parameters, report restore failures, and close.

Persistent camera/network operations are absent by design.
