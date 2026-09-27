# ADR 0013: ASCII and JSON for Phase 1; MusicXML and Guitar Pro on the app track

Status: accepted (2026-09-27)

Decides spec D4.

## Context

Phase 1 needs an output a human can check and an output a machine can consume. It does
not need interchange with notation software yet.

## Decision

- **ASCII tab** — six lines, low E at the bottom, time-proportional spacing (ADR 0009),
  with a marker on notes whose posterior falls below a threshold. This is the format I
  read, and the one I can play from to check the decoder against a real guitar.
- **JSON** — every `TabNote` with its posterior and its alternatives. This is what the
  later web UI (spec D14) needs to show uncertainty on hover, and what makes a
  regression test possible.

MusicXML and Guitar Pro export (`PyGuitarPro` writes gp3-gp5) belong to the application
track after M1. MIDI is not an output: it cannot express string assignment, which is
the thing this project produces.

## Alternatives considered

- **MusicXML in Phase 1.** Rejected: it wants rhythmic notation to be worth anything,
  and ADR 0009 defers that.
- **MIDI.** Rejected: it would silently discard the fingering, which is the output.

## Consequences

**Easier.** Both formats are pure functions of the decoder's output, testable with a
golden file. Uncertainty is visible in ASCII and machine-readable in JSON.

**Harder.** No path into notation software until the app track, so the output cannot yet
be edited in Guitar Pro or MuseScore.

**Revisit at:** the app track, after M1.
