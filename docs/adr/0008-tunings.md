# ADR 0008: User-chosen tuning and capo from day one; no automatic detection

Status: accepted (2026-09-27)

Decides spec D3.

## Context

Tuning is baked into candidate generation: it decides which (string, fret) pairs can
produce a pitch at all. Retrofitting it later would mean touching every stage.

## Decision

The `Tuning` type (open pitches, `n_frets`, `capo`) is a parameter everywhere from the
start, and fret numbers are measured **relative to the capo**. Standard tuning is the
default, not an assumption. Automatic tuning detection is out of scope for v1.

`Tuning.pitch_at` is the only place pitch arithmetic happens, so candidate generation
and the E4 pitch-validity metric cannot disagree about what a fret sounds.

## Alternatives considered

- **Standard tuning only, generalise later.** Rejected: supporting arbitrary tunings
  costs almost nothing now (one indexed lookup) and would be invasive later.
- **Automatic tuning detection.** Rejected for v1: it is a separate estimation problem
  with its own failure modes, and getting it wrong corrupts every downstream stage in a
  way that is hard to diagnose. A user knows their own tuning.

## Consequences

**Easier.** Drop tunings, capos and non-standard setups work without new code paths.
Users with a capo get correct fret numbers rather than absolute ones.

**Harder.** Every function that places a note needs the tuning threaded through it, and
tests have to cover the capo case explicitly or the off-by-`capo` bug hides.

**Revisit:** only if users turn out to mis-declare their tuning often enough to matter.
