# ADR 0016: The D10 headline target

Status: **proposed (2026-09-27) — awaiting Ege's sign-off on X**

Completes ADR 0004, which fixed the headline *metric* and deferred the *target* until a
baseline existed. One exists now. ADR 0004 is not superseded: the metric is unchanged.

## Context

ADR 0004 chose E2, Exact Tab F1 on GuitarSet under our protocol, and said the target would
be set after M1 as "baseline + X points", because setting one before a baseline exists is
inventing a number. Phase 1.5 then moved the baseline twice (ADR 0018, ADR 0019), so the
figures below are the current ones, not M1's.

**Baseline**, commit `a05e1e0`, all 360 GuitarSet tracks, hand-set weights:

| | oracle | end-to-end |
|---|---|---|
| E1 note F1, transcriber (raw) | 1.0000 | 0.7437 |
| E1 note F1, pipeline (placed) | 1.0000 | 0.7452 |
| **E2 exact tab F1** | **0.6599** | **0.4318** |
| E3 playable groups / transitions | 0.9971 / 0.9134 | 0.9897 / 0.9164 |
| E4 pitch validity | 1.0000 | 1.0000 |
| E5 calibration error | 0.1652 | 0.3851 |

### Where the headroom actually is

Three facts decide the shape of the target, and none of them is a preference.

1. **End-to-end E2 is bounded above by end-to-end E1.** Every E2 match is also an E1
   match — E2 requires onset and pitch *and* string — over the same note counts, so
   `E2_e2e <= 0.7452` until Phase 5 replaces the transcriber. A target naming only an
   end-to-end number would be a target on `basic-pitch`, not on the fingering work.
2. **The fingering question lives in oracle mode.** Conditional on hearing a note, we place
   it right 66.0% of the time given perfect notes and 57.9% end to end (0.4318 / 0.7452).
   Phases 2 and 3 change that conditional rate, and it is visible in oracle mode first and
   undiluted.
3. **End-to-end moves at roughly three-quarters of oracle.** An oracle gain Δ can reach at
   most about `0.74 Δ` end to end, and less in practice because a mis-transcribed
   neighbour also degrades the decode of a correct note.

### What counts as a real move

The note-level standard error on E2 = 0.66 over 62 476 notes is 0.0019. Notes inside a
track are correlated, so the effective sample is smaller; at a design effect of 10 the
standard error is about 0.006. **Half a point is noise. Two points is real but small** —
ADR 0018 was worth +2.3 points and it was a bug fix, not a model.

## Decision (proposed)

**X = 10 points of oracle E2, at M2.**

| | target | from | this is |
|---|---|---|---|
| **E2 oracle** (the target) | **≥ 0.760** | 0.6599 | baseline + 10.0 points |
| E2 end-to-end (expected, not a gate) | ≥ 0.480 | 0.4318 | baseline + ~5 points |

with these guardrails, from ADR 0004's rule that E2 is never reported alone:

- **E4 = 1.0000 exactly.** Not a target; anything less is a bug that blocks reporting.
- **E3 group and transition rates must not fall** below the baseline above. A model that
  raises E2 while producing unplayable tab has not improved anything.
- **E5 end-to-end must improve on 0.3851.** It is the weakest number in the table and
  spec 1's fourth goal — being honest about uncertainty — is not met at 0.39.

**Why ten and not five or twenty.** Ten points removes 29% of the 34% of oracle placements
currently wrong. Five points is only about 4.5 standard errors clear of ADR 0018's +2.3,
which was a bug fix rather than a model, so it would let a marginal result count as
success. Twenty points would require
halving the error, which is a claim about learned fingering nobody can support before the
data is in hand. Ten is the smallest number that cannot be reached by tightening the
existing hand-set model, which is the thing the target is meant to distinguish.

**Not derived from anyone else's figure.** Spec 3.4's reference points were measured under
other protocols and this target is not calibrated against them, above them or below them.
It comes from our own baseline and our own error structure.

## Alternatives considered

- **A target on end-to-end E2 alone.** Rejected on fact 1: it is a target on the
  transcriber, and it would be unreachable for reasons unrelated to the fingering work.
- **A target on the conditional string-accuracy rate** (E2 / E1) rather than on E2.
  Tempting, because it isolates fingering exactly. Rejected: it is not the metric ADR 0004
  chose, and changing the headline metric needs a superseding ADR, not a target ADR. It is
  reported as a diagnostic instead.
- **No target, just "report the numbers".** Rejected: ADR 0004 already argued this, and
  spec 1 requires accuracy to be stated as a number on a named benchmark.
- **A target per phase.** Rejected as premature: Phases 3–5 have no plans yet, and a target
  for a phase whose data has not arrived would be invented.

## Consequences

**Easier.** M2 has a pass/fail condition that can be checked with one `make eval-m1`, and
one that a marginal result cannot sneak through.

**Harder.** Ten points is a real bar. If the learned model lands at +6, that is a result to
report as a result — "a learned fingering model on DadaGP beats hand-set weights by 6
points but missed the target" — not a reason to move the target afterwards. **Moving this
target after seeing a number requires a superseding ADR that says so in as many words.**

**Revisit** only if the baseline moves again for reasons unrelated to the fingering model
(another Phase 1.5-style defect fix), in which case the *points* stay and the absolute
number is recomputed from the new baseline.

## What Ege is being asked to decide

Only **X = 10**. Everything above it is derivation; the number is a judgement about what
would count as the learned model having worked. Say a different number and the rest of the
ADR still stands.
