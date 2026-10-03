# ADR 0028: The hand window stays after missing its validation check

Status: accepted (2026-10-03) — **the pre-registered check was missed, and Ege kept the
window**

Follows [ADR 0025](0025-hand-window.md) and changes nothing in it. ADR 0025 records the
window's first check — its acceptance gate on human tab — and Ege's first decision to keep
it; this record holds the second check and the second decision. It was first drafted as an
addendum to ADR 0025 on the unmerged branch and moved here, because an accepted ADR is not
edited.

## Context

`docs/plans/2026-10-02-hand-window-and-calibration.md` (Task 6) pre-registered a check in
commit 40f8504, before any window-model run: *the hand window does not lower the hand-set
weights' recovery of human fingerings on clean or distorted artist-disjoint validation
parts.* It set no tolerance.

| hand-set weights, 300 artist-disjoint validation songs | clean (133,604 notes) | distorted (372,243 notes) | NLL per group |
|---|---|---|---|
| point model (840ca52) | 0.8217 | **0.5728** | 0.7489 |
| hand window (3e40e31) | 0.8257 | **0.5705** | **0.6937** |

**The check failed: distorted recovery fell by 0.0023.** Clean recovery rose by 0.0040, and
the window raised the likelihood the model gives human fingerings.

What the miss does and does not show: the comparison is paired — the same notes under both
models — so even a small difference can be real. No uncertainty was estimated, and the rule
set no tolerance, so whether 0.0023 is noise is unknown. A song-level paired bootstrap on
DadaGP validation would answer it; none has been run.

## Decision

**Ege kept the window (2026-10-03).** It stays in the decoder, the cost model, E3 and the
CRF fitter exactly as ADR 0025 describes.

## Alternatives considered

- **Return to the point model.** Offered to Ege and declined. The case for keeping the
  window, as put to Ege: it fixes a measured flaw (ADR 0022), fits human fingerings better
  by likelihood, and gains on clean parts.
- **Re-judge the miss with a tolerance chosen now.** Rejected: a tolerance chosen after the
  result is not a pre-registration.

## Consequences

Both of the window's pre-registered checks were missed — ADR 0025's gate and this one — and
both decisions to keep it were Ege's, made with the numbers in hand. The lesson is about
pre-registration: the next comparison states, before it runs, a tolerance and how its noise
will be estimated, so that a result of this size can be read either way.

**Revisit** if a paired bootstrap on validation shows the distorted-part loss is real and
material, or when task C3 fits clean and distorted playing separately.
