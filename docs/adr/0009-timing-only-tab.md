# ADR 0009: v1 renders time-positioned tab, not rhythmic notation

Status: accepted (2026-09-27). What the MusicXML and Guitar Pro exporters do about
a format that cannot be written without durations is decided separately in
[ADR 0017](0017-rhythm-for-exports.md); v1 still transcribes no rhythm.

Decides spec D5 — the largest piece of hidden scope in the project.

## Context

Real tabs have measures, beats and note values. Producing them needs beat tracking and
quantisation: tempo estimation, downbeat detection, and a grid to snap onsets to. That
is a second MIR problem with its own evaluation, its own failure modes, and no overlap
with the fingering question this project is actually about.

## Decision

v1 spaces notes **proportionally to time**. No bar lines, no note values, no beat grid.
Onsets carry the rhythm implicitly through horizontal position.

Quantised notation is its own mini-project after M3, not a stretch of Phase 1.

## Alternatives considered

- **Quantise to a beat grid now** (e.g. with `beat_this`). Rejected: it puts an
  unmeasured estimator between the decoder and the output, so a wrong-looking tab could
  be the decoder's fault or the beat tracker's, and we would not know which.
- **Full notation with durations.** Rejected outright for v1; it needs the beat grid
  plus voice separation.

## Consequences

**Easier.** The renderer is a function of onsets and positions only, and is trivially
testable with a golden file. Nothing in the pipeline depends on tempo.

**Harder.** The output does not look like a tab from a tab site, and cannot be imported
into Guitar Pro as rhythmically meaningful. Anyone reading it has to infer rhythm from
spacing. This is the single most visible v1 limitation and belongs in the README.

**Revisit at:** after M3.
