# AGENTS.md — Defleco LAB development guardrails

## Project priority

Defleco LAB is an experimental image-processing laboratory for reflected-pattern /
deflectometry research. The main value is fast, reliable comparison of processing
methods on real and recorded data.

## General execution contract

Work delta-first, not from scratch.

Before implementing anything:
1. identify what the requested outcome actually changes;
2. identify existing code, prior decisions, referenced repositories, scripts, data,
   tests, or documentation that already solve part of it;
3. reuse those parts instead of recreating them;
4. implement the missing delta while preserving already-solved behavior;
5. validate at a depth proportional to the uncertainty, consequence, and decision value of the change.

If the task explicitly points to an existing implementation, treat it as the
default source of truth for that functionality unless there is a demonstrated
reason not to reuse it.

Do not create a parallel abstraction, helper, wrapper, test harness, or rewritten
implementation merely because it is convenient for the current task.

## Context and data reuse

Do not repeatedly re-read unchanged material without a concrete reason.

Prefer:
- previously established findings when the source has not changed;
- targeted reads of changed files or unresolved code paths;
- diffs and exact call sites over repository-wide rescans;
- recorded/replay data over repeated hardware acquisition when the same frames are
  sufficient for the question.

Re-read or re-test only when:
- the relevant source changed;
- the previous result is uncertain;
- a dependency changed;
- the new task exercises a different code path;
- or the user explicitly asks for fresh verification.

## Validation economy

Tests are evidence, not ceremony.

A commit SHA, green CI, successful import, GUI startup, or device discovery proves
only that specific fact. Do not present those as proof that the application's
primary function works.

For each change, choose the test or analysis with the highest decision value.
Small targeted tests are preferred when they answer the question, but broader or
longer validation is appropriate when the unresolved uncertainty, failure mode, or
consequence requires it. Do not optimize for minimum test count at the expense of
confidence in the requested behavior.

Do not, unless explicitly requested:
- rerun unchanged hardware validation;
- run long soak, stress, resize, click, or GUI-torture loops;
- cycle unrelated methods or parameters;
- benchmark unchanged code;
- regenerate screenshots or reports unrelated to the requested behavior;
- repeat tests whose result is already established and whose dependencies did not
  change.

When a failure is found, test the shortest path from cause to user-visible effect.


## GitHub and CI workflow hygiene

Treat remote pushes and CI runs as externally visible project events, not as a
scratchpad. This repository runs CI on pushes and pull requests, and failed runs
can generate user notifications.

For multi-step implementation, debugging, or review work:
- group related edits into a coherent change set before pushing to `main`;
- prefer a temporary branch / pull request when several dependent edits or
  corrective iterations are expected;
- local or branch commits may be frequent, but avoid chains of small remote pushes
  to `main` that repeatedly trigger redundant CI;
- do not knowingly push an intermediate lint/test failure to `main`;
- when CI fails, inspect the exact failed step and its log before making another
  corrective change;
- distinguish lint/static-analysis failures from compile, unit-test, runtime,
  hardware, and acceptance-test failures when diagnosing or reporting them;
- do not interpret a failed workflow headline as evidence that every test failed;
  identify which step actually failed and which later steps were skipped;
- after a sequence of changes, verify that the latest relevant `main` CI run is
  green before treating the repository state as settled.

Do not rerun CI merely to obtain a green badge when the failure has a deterministic
code cause. Fix the cause first, then use one relevant run as confirmation.

When several files must change together but the available GitHub tooling writes
them as separate commits, use a working branch where practical and merge the
coherent result instead of exposing each intermediate state on `main`.

## Definition of done

Completion is defined by the requested behavior, not by activity volume.

For processing changes, acceptance should exercise the actual processing path and
show that the selected output responds meaningfully to a known synthetic or
recorded disturbance where practical.

For GUI changes, verify only the affected interaction and its user-visible result.

For camera changes, reuse the established camera path already present in the
project and test only the changed camera behavior.

For documentation-only changes, do not run runtime or hardware tests unless the
documentation change itself depends on them.

## Runtime rules

The GUI must remain responsive even if processing cannot keep up.

Use bounded acquisition/history/result paths and latest-wins processing where
appropriate. Do not compute hidden expensive outputs that are not being used.

The Original view is a camera/recording baseline; selected heavy processing should
not run merely because live acquisition is active.

## Camera safety

Do not modify persistent camera state unless explicitly requested.

Do not write persistent User Sets, force network configuration, update firmware,
or drive hardware outputs as part of ordinary testing. Temporary session changes
should be restored on clean shutdown where practical.

## Reporting

Report:
- what behavior changed;
- what existing implementation was reused;
- the minimum meaningful validation performed;
- any remaining uncertainty that matters to the user's next decision.

Do not pad reports with exhaustive file lists, SHAs, benchmark tables, or repeated
test inventories unless they materially help the task or the user asks for them.
