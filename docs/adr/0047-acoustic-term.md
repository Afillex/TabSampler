# ADR 0047: The acoustic term in the decoder

Status: accepted (2026-10-04)

Carries out Phase 3's Task 4 (`docs/plans/2026-10-04-phase-3-audio.md`) for ADR 0046's evidence:
how a string probability per note reaches the decoder's cost.

## Decision

- **`HandSetScorer` takes the evidence beside its weights**: `evidence`, a mapping from a note —
  the `NoteEvent` itself — to six log-probabilities, low string first. A note placed on string
  `s` adds `acoustic × −log P(s | audio)` to its shape's emission cost (spec §2.2).
- **A note with no evidence adds nothing, and nor does any note when `acoustic` is zero**: the
  decoder is then Phase 2's exactly, which is what the ablation compares against.
- **Keyed by the note itself**, because the evidence is computed for exactly the notes the decoder
  places — the reference notes in oracle mode, the transcriber's end to end — and a note group
  holds those same events.
- **The `FingeringScorer` protocol does not change**: the term lives in the emission cost, which
  the decoder, forward-backward and the brute-force oracle already call, so the oracle checks it
  with no new code of its own.
- **Not fitted**: the fitter's features leave the term out, and its weight is fixed by a rule
  committed before it is measured (Task 5).

## Alternatives considered

- **Evidence on `NoteEvent`.** A contract change (ADR 0007) to carry a field that only Phase 3
  fills, through every stage.
- **A scorer that wraps `HandSetScorer`.** It would duplicate the cost model's plumbing for one
  term.

## Consequences

**Easier.** The ablation is one weight set to zero, and every guarantee the oracle gives the
decoder covers the new term.

**Harder.** A scorer now carries data as well as weights, so it is built per track.
