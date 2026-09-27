# ADR 0002: v1 accepts isolated guitar recordings only

Status: accepted (2026-09-27)

Decides spec D1.

## Context

The input we accept sets the difficulty of every downstream stage, which datasets are
usable, and how honestly we can evaluate. Accepting full mixes from day one would mean
source separation sits in front of the pipeline before there is any evidence the pipeline
works at all, and every error would be ambiguous between separation and transcription.

## Decision

v1 accepts **isolated guitar audio only** — one guitar, no other instruments. Full-song
input is Phase 6, where `htdemucs_6s` produces a guitar stem and results are reported
separately from isolated mode.

## Alternatives considered

- **Isolated guitar plus a best-effort full-song mode.** Rejected: "best effort" with no
  measurement is the kind of claim this project exists not to make, and it doubles the
  evaluation surface before the ground-zero number exists.
- **Full songs from day one.** Rejected: it puts an unmeasured separation stage in front of
  everything and makes every metric uninterpretable.

## Consequences

**Easier.** GuitarSet, GOAT, GAPS and Guitar-TECHS are all usable as-is. Transcription
errors are attributable to the transcriber rather than to separation. The Phase 0 and
Phase 1 gates stay small enough to actually reach.

**Harder.** The app is less immediately useful to someone holding a finished recording.
Phase 6 must report isolated and full-song results as separate tables, never merged, or
the comparison becomes dishonest.

**Revisit at:** Phase 6.
