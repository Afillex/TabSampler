# ADR 0001: Python 3.13 for the core, basic-pitch isolated on Python 3.11

Status: accepted (2026-09-27)

## Context

This decision is **not** from a spec cue. It was forced by a dependency constraint
discovered while planning Phase 0, and it sets `requires-python` for the whole repository,
so it needs to be written down before any code depends on it.

Spec 6 (Phase 0) requires running Basic Pitch on GuitarSet. Spec 4 requires development
and inference on an M4 MacBook Air (macOS arm64). Those two requirements collide:

`basic-pitch` 0.4.0 declares, in its package metadata:

```text
coremltools; platform_system == "Darwin"
tensorflow-macos<2.15.1,>=2.4.1; platform_system == "Darwin" and python_version > "3.11"
```

and `tensorflow-macos` has wheels only for `cp39`, `cp310` and `cp311`.

Measured consequences (`uv pip compile`, 2026-09-27, macOS arm64):

| Target | Result |
|---|---|
| basic-pitch 0.4.0 on Python 3.12 / 3.13 / 3.14 | **unsatisfiable** — no `tensorflow-macos` wheel |
| basic-pitch 0.4.0 on Python 3.11 | resolves, **no TensorFlow at all**: `coremltools 9.0`, numpy 2.4.6, librosa 0.11.0 |
| One shared env on Python 3.12 | resolves, but degrades to basic-pitch **0.3.0**, TensorFlow 2.16.2, numpy **1.26.4**, librosa **0.11.0** |
| Core stack without basic-pitch on 3.13 / 3.14 | resolves clean: numpy 2.5.3, scipy 1.18.1, librosa 1.0.0, numba 0.67.0, mirdata 1.0.0, mir_eval 0.8.2, torch 2.14.0 |

So a single shared environment costs an older transcriber, numpy 1.x, an older librosa,
and ~400 MB of TensorFlow that the CoreML path does not need.

## Decision

Two environments.

1. **Core package:** `requires-python = ">=3.13,<3.14"`, pinned with `uv python pin 3.13`,
   locked in `uv.lock`. This is what `src/tabsampler` is type-checked and tested against.
2. **Transcriber:** `basic-pitch==0.4.0` on Python 3.11, installed separately with
   `uv tool install --python 3.11` (see `scripts/setup_transcriber.sh`), deliberately
   **outside** `uv.lock`, and reached only through a subprocess boundary
   (`transcribe/basic_pitch_cli.py`) plus an on-disk note-event cache.

3.13 rather than 3.14 because both resolved identically on the day, 3.13 was already
present in the local uv Python store, and it has a year more wheel maturity — which
matters at Phase 2 (torch) and Phase 5 (onnxruntime, coremltools).

## Alternatives considered

- **One environment on Python 3.11.** Simplest, and genuinely tempting: one lockfile, no
  subprocess. Lost because it pins the entire project — including the Phase 2 training code
  and every future dependency — to a Python that reaches end of life in October 2027, in
  order to accommodate one stage of the pipeline.
- **One environment on Python 3.12.** Lost on every axis at once: older basic-pitch, numpy
  1.x, older librosa, plus TensorFlow.
- **Vendor the Basic Pitch ONNX/CoreML model and run it directly with `onnxruntime`,
  dropping the `basic-pitch` package.** Attractive, and it may become the right answer at
  Phase 5. Rejected for now because it means reimplementing the model's post-processing
  (note creation from onset/frame/contour posteriorgrams, the melodia trick), which is
  exactly the part we want to take as given while reproducing a published baseline
  ("reproduce before you innovate").

## Consequences

**Easier.** The core keeps numpy 2.x and librosa 1.x. The transcriber's dependency rot is
quarantined: it cannot constrain anything else. The cache makes Phase 1's experiment loop
fast, because transcription stops being re-run, and gives spec 4's determinism requirement
almost for free. `Transcriber` being a Protocol means swapping the implementation touches
one adapter file.

**Harder.** Two environments to install, documented in the README. A process boundary that
needs real error handling: a missing executable or a non-zero exit must raise, never return
an empty note list, because an empty list scores as note F1 = 0.0 and reads as a result
rather than a bug. A cache-key scheme that must include the transcriber's parameters and
version, or stale results will be silently reused. Crossing the boundary via CSV also
quantises `confidence` to `velocity/127`, since the CSV writes an integer MIDI velocity
where the in-process API returns a float amplitude — harmless in Phases 0-1, where nothing
consumes `confidence`, but Phase 3's acoustic term will care.

**Revisit when:** Phase 2 needs torch features unavailable on 3.13; Phase 5 replaces the
transcriber; or `basic-pitch` ships a release without the `tensorflow-macos` constraint.
