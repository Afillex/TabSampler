# ADR 0023: Cost weights fitted on DadaGP — measured, recorded, not yet the default

Status: accepted (2026-10-01)

Supersedes ADR 0012's premise that tuning is blocked: it is no longer. Does **not** change the
default decoder: `configs/phase1_baseline.yaml` (hand-set) stays the default, for the reason
in *Decision*. The fitted decoder is `configs/fitted_dadagp.yaml`.

## Context

ADR 0012 shipped M1 with hand-set weights because no legal tuning data existed. DadaGP now
does (ADR 0021), and Phase 1.5 had shown the hand-set ratio `move : high` = 120 : 1 was
probably wrong (ADR 0018). Every step below was pre-registered in a script committed before
it ran, and each prediction is reported whether or not it held.

## What was done and measured

**Fit** (`scripts/fit_cost_weights.py`). The decoder is linear in its four weights, so it is a
linear-chain CRF; the weights were fitted by maximum likelihood with exact gradients
(`fingering/fit.py`, checked against the oracle-verified `log_partition`, a brute-force sum
over paths, and finite differences). 600 training songs, 469,363 chord shapes, L-BFGS
converged in 11 iterations:

| | move | span | high | open_reward |
|---|---|---|---|---|
| hand-set | 1.0 | 1.0 | 0.1 | +0.25 |
| **fitted** | **0.5772** | **0.7377** | **0.3767** | **−1.1483** |

Two disjoint training halves differ by up to 28% on `high` but recover the same share of
validation fingerings to within 0.3 points, so the spread is song-to-song style, not noise.
`open_reward` is negative; the features are correlated (an open string also lowers mean fret
and span), so that sign is a correction given the other three terms, not a standalone claim.

**DadaGP validation** (300 songs, 529,932 notes, identical lattices for both weight sets):

| | NLL / group | recovery | clean parts | distorted parts |
|---|---|---|---|---|
| hand-set | 0.7676 | 0.6196 | **0.8097** | 0.5422 |
| fitted | **0.4976** | **0.6456** | 0.6963 | 0.6251 |

Prediction "`move : high` falls by an order of magnitude": **missed** — 120 : 1 → 18.4 : 1,
a 6.5× fall. The pooled gain is entirely distorted parts (71% of the notes); on clean parts
the fitted weights are 11 points worse.

**Temperature** (`scripts/calibrate_temperature.py`, DadaGP validation): **T = 1.721**.
Calibration error, computed as E5 computes it: hand-set 0.2689 → fitted 0.1431 → fitted and
calibrated **0.0562**. Prediction "T is close to 1": **wrong** — path-level likelihood leaves
per-note posteriors overconfident. Prediction "most of the gain is the weights": **held**,
59% weights, 41% temperature.

**GuitarSet, evaluated once** (`configs/m2_fitted_eval.yaml`, committed before the run):

| | hand-set | fitted + T | |
|---|---|---|---|
| E2 oracle | 0.6599 | **0.6988** | +0.0389 |
| E2 end-to-end | 0.4318 | **0.4603** | +0.0285 |
| E3 chord shapes, oracle / e2e | 0.9970 / 0.9896 | 0.9961 / 0.9883 | **below the guardrail** |
| E3 transitions, oracle / e2e | 0.9134 / 0.9164 | 0.8955 / 0.9004 | (not a playability measure, ADR 0022) |
| E4 | 1.0000 | 1.0000 | |
| E5 end-to-end | 0.3851 | **0.1411** | −63% |
| E5 oracle | 0.1652 | **0.1286** | −22% |

The original prediction — E2 rises, the M2 target of 0.760 is not reached — **held**. The
revision made before the run from DadaGP's clean parts — E2 falls — **was wrong**: clean
DadaGP parts are not a good proxy for acoustic GuitarSet. ADR 0016's E3 chord-shape
guardrail is **breached** in both modes, by 0.0009 and 0.0013.

## Decision

1. **The fitted weights and temperature are recorded and reproducible**, in
   `configs/fitted_dadagp.yaml` and `experiments/results.csv`.
2. **They are not the default.** That decision was made on DadaGP validation, before
   GuitarSet was run, because the fitted weights are 11 points worse on clean guitar parts
   and a typical user records clean or acoustic guitar.
3. **GuitarSet may not be the reason to reverse it.** The GuitarSet numbers favour the fitted
   decoder, and adopting it *because of them* would be selecting on the test set (ADR 0003).
   There are two legitimate routes, and choosing between them is Ege's call:
   - **Honour the rule as first pre-registered.** `scripts/fit_cost_weights.py` committed
     "adopt iff pooled validation recovery improves", which it did. Declining was a post-hoc
     deviation made from a validation breakdown, and it is defensible to say the
     pre-registered rule should govern. The cost: it means accepting a model that breaches
     the E3 guardrail by a tenth of a point.
   - **Earn it on validation.** Pre-register a validation proxy for acoustic playing chosen
     on principle, fit weights for it, and adopt only if they win there.
4. **The E3 breach stands as a breach.** It is small, but ADR 0016 does not grade guardrails
   by size, and E3's chord-shape rate is now a validated playability measure (ADR 0022).

## Alternatives considered

- **Adopt now, since GuitarSet improved.** Rejected: test-set selection, the one thing ADR
  0003 exists to prevent.
- **Report only the pooled validation number.** Rejected: it hides an 11-point loss on the
  kind of playing most users will submit.
- **Fit on clean parts only, now.** Not done: it is a new experiment with its own
  hypothesis, and choosing "clean" because it resembles GuitarSet must be argued before the
  run, not after.

## Consequences

**Easier.** Tuning is no longer blocked, and the machinery — the CRF fitter, the protocol,
the pre-registration habit — is in place for every later fit, including Phase 2's.

**Harder.** The project now has two credible decoders and a decision that the test set is
not allowed to make.

**Revisit** when Ege chooses between the two routes in decision 3, or when the hand-window
movement model (ADR 0022) changes the features these weights multiply.
