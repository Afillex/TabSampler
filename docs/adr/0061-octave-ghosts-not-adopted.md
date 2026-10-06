# ADR 0061: Dropping octave ghosts is not adopted; nor a loudness floor, nor a new uncertainty threshold

Status: accepted (2026-10-06)

Reports Tasks 2 and 3 of `docs/plans/2026-10-06-optimise-current.md`, every rule committed before
its run (89e93e3, 1025b0c).

## Context

Before Phase 6, Ege chose three improvements to what exists: a faster app (Task 1, done without a
decision: the tuning check now overlaps transcription), uncertainty marks that point at wrong
notes (Task 2), and fewer of Basic Pitch's extra notes on electric guitar (Task 3). At Phase 5's
thresholds Basic Pitch writes 1,951 notes on Guitar-TECHS's player 3 where 1,629 were played; 566
are extra, 109 of them an octave from a played note.

## Result

**Task 2 — the uncertainty threshold stays 0.6.** End to end on player 3, 61.5% of tab notes are
wrong. The rule's threshold (the largest marking at most a quarter of the notes) is 0.50; marked
notes are wrong 0.794 of the time, unmarked 0.558 — a lift of 1.42, short of the 1.5 required. At
today's 0.6 the lift is 1.12. Of the 1,194 wrong notes, 566 are extra notes the decoder's posterior
cannot see and 639 are heard notes on the wrong string, so the posterior ranks even its own string
errors weakly.

**Task 3a — a loudness floor: not adopted.** Basic Pitch's amplitudes never fall below 0.244 at
its thresholds, so the pre-registered floors (0.10–0.40) dropped at most 21 notes; best +0.0025.

**Task 3b — octave ghosts: adopted on validation, rejected by the test look.** Dropping a note
exactly an octave above a note at least as loud that starts within 50 ms:

| | E2 before | E2 after | E1 before | E1 after | notes dropped |
|---|---|---|---|---|---|
| player 3 (validation, chosen on) | 0.4185 | 0.4434 (+0.0249 [−0.0253, +0.0993]) | 0.7737 | 0.7675 | 101 (51 extra, 50 played) |
| **EGDB, 240 clips (test)** | **0.4547** | **0.4537 (−0.0010)** | **0.7416** | **0.7136** | 4,909 of 38,790 |

On player 3 the gain came from the decoder's context, not from cleaner notes — half the dropped
notes were played. On EGDB the filter removed 13% of the notes, many of them played octaves, and E2
did not rise. The adoption rule (EGDB's change above 0, GuitarSet's above −0.01) failed on EGDB;
**GuitarSet's test look was not run**, since it could not change the outcome. Predictions: EGDB's E2
change fell inside −0.01 to +0.03; its E1 fall (0.028) exceeded the predicted 0 to 0.01.

## Decision

- The pipeline keeps every transcribed note; `transcribe/cleanup.py` stays, tested and unused.
- `uncertainty_threshold` stays 0.6; the page's wording (a ranking, uncalibrated) stays.
- Player 3 — 12 takes, one player — has now chosen Phase 5's thresholds and failed to predict
  this filter. A rule chosen on it alone should be expected to shrink or vanish on EGDB.

## Consequences

The measured gains of this round are in speed only (a 3-minute file: 9.05 s → 5.17 s). Accuracy
is where Phase 5 left it. The largest measured loss is string choice (EGDB: 0.32 of E2 even on the
true notes), and the only measured lever on it is Phase 4's electric audio evidence (ADR 0054),
which needs publishable weights, fresh validation data and PyTorch at run time. EGDB has now been
read four times (`experiments/test_set_access.log` lines 31, 32, 33 and 35); each further look should
be spent as sparingly.
