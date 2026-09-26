# AGENTS.md — Defleco LAB development guardrails

## Project priority

Defleco LAB is primarily an experimental image-processing laboratory for evaluating reflected-pattern / deflectometry methods. The novel work is the processing and evaluation layer, not basic Basler camera plumbing.

When working on this repository, optimize first for:
1. reliable live acquisition;
2. responsive GUI;
3. reproducible record/replay;
4. correct and inspectable processing results;
5. fast comparison of pattern + algorithm combinations.

Do not spend substantial time re-solving camera control that already exists elsewhere.

## Reuse-first rule for Basler hardware

Before changing or adding Basler camera behavior, inspect and reuse the proven implementation patterns from:

- https://github.com/CZPavel/basler-camera-encoder
  - camera discovery and exact identity selection;
  - CameraReadback;
  - ExposureAuto / ExposureTime distinction;
  - GainAuto / Gain distinction;
  - resulting FPS;
  - read-only runtime snapshot;
  - acquisition worker patterns.

- https://github.com/CZPavel/basler-ace2-gige-tester
  - defensive GenICam helpers;
  - trigger handling;
  - camera profile snapshot / restore;
  - ROI, binning and decimation ordering;
  - temporary profile changes;
  - rollback;
  - low-load/live preview behavior.

Do not invent a new camera abstraction or new parameter semantics if one of those repositories already contains a working implementation that can be adapted.

When copying/refactoring MIT-licensed logic, preserve attribution in NOTICE where appropriate.

## Exposure / gain semantics

Never interpret ExposureTime or Gain alone as the complete camera configuration.

Always consider:
- ExposureAuto
- ExposureTime
- GainAuto
- Gain

When ExposureAuto is not Off, ExposureTime may be the current output of the automatic control loop rather than a manually configured constant.
When GainAuto is not Off, treat Gain analogously.

Do not write ExposureTime while ExposureAuto is active unless the user explicitly requests a transition to manual exposure.
Do not write Gain while GainAuto is active unless explicitly requested.

## Camera safety

Do not modify persistent camera state unless the user explicitly asks for it.

Do not:
- write persistent User Sets;
- use Force IP;
- update firmware;
- change persistent network configuration;
- drive GPIO outputs.

Temporary session changes must be restored on clean shutdown where practical.

## Processing/runtime rules

The GUI must remain responsive even if processing cannot keep up.

Use:
- bounded acquisition buffers;
- latest-wins processing;
- bounded result delivery;
- bounded live frame history;
- explicit processing/display scale.

Dropping stale processing frames is acceptable.
Accumulating unbounded latency is not.

Do not compute hidden heavy views or intermediate maps unless they are needed.

The Original view is a camera/recording baseline. Do not run the selected processing method continuously while only the Original tab is active.

## Definition of done

A task is not complete just because the GUI starts, CI is green, or a camera can be opened.

For processing work, acceptance must exercise the actual processing path:
- produce a non-empty numerical response from the changed method;
- verify a known synthetic disturbance changes the response in the expected region where practical;
- verify the Processed view can display the selected primary output;
- use recorded real frames/replay for algorithm comparisons once such data exist.

For camera work, reuse the proven camera layer first and test only the changed camera behavior. Do not spend the task budget re-validating unrelated camera functions.

Prefer end-to-end evidence for the project's primary purpose over broad peripheral testing.

## Testing scope

Default hardware validation should be short and targeted.

Unless the user explicitly requests a soak/stress test:
- do not run long GUI torture tests;
- do not repeatedly move/resize windows for minutes;
- do not cycle every processing method;
- keep a normal hardware smoke test around 1–2 minutes;
- test only the functionality changed by the current task.

For algorithm development, prioritize recorded real data and replay over repeated live-camera retesting.

## Token/time discipline

Before implementing:
1. inspect existing code and related repositories;
2. identify what is already solved;
3. reuse it;
4. change only the missing layer;
5. run the smallest acceptance test that proves the change.

Do not broaden the task with speculative cleanup or unrelated benchmarks without explicit user request.
