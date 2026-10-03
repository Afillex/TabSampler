# ADR 0038: The default decoder, re-decided on the validation player

Status: proposed (2026-10-03) — pre-registration; accepted with its result

Follows [ADR 0032](0032-style-decoders.md), which made weights fitted on clean DadaGP parts
the default, and [ADR 0037](0037-guitarset-validation-player.md), which made GuitarSet's
player 00 validation data. Decided in principle by Ege on 2026-10-03.

## Context

ADR 0032 chose the clean-fitted weights on DadaGP validation (+3.6 points on clean parts).
On all 360 GuitarSet tracks the default then lost 1.9 points of oracle E2 and breached both
chord-shape guardrails. DadaGP has been a poor guide to GuitarSet three times; player 00
is the first validation data that is GuitarSet's own kind of playing.

## Decision, fixed before the runs

**Candidates**, both with the hand window and its stretch (ADRs 0025, 0030):

- **A — the hand-set weights** (`configs/phase1_baseline.yaml`: move 1, span 1, high 0.1,
  open_reward 0.25), the default for longest and set from guitar knowledge, not a proxy.
- **B — the clean-fitted weights** (`configs/decoder_clean.yaml`, today's default).

The temperature changes neither E2 nor E3, so it plays no part in the choice.

**Runs:** `tabsampler eval-m1 --split validation --per-track-out ...` on player 00's 60
tracks, oracle and end to end, once per candidate, under `configs/m2_validation_redecide.yaml`;
compared with `scripts/compare_validation.py` (track-level paired bootstrap, exact counts).

**Rule, oracle mode:** B stays the default iff its oracle E2 is higher than A's with the
95% paired interval wholly above zero, **and** its oracle chord-shape rate is not lower than
A's by more than 0.0005. Otherwise `configs/decoder_clean.yaml` takes A's weights, with a
temperature calibrated for them on DadaGP's clean validation parts by ADR 0033's method.
A is the incumbent because B's only evidence came from the proxy that failed; B has to win
on the data we now trust. End-to-end figures are recorded and do not enter the rule. The
distorted decoder is untouched: GuitarSet has no distorted guitar.

**Prediction: B loses**, as it did on all 360 tracks — which included player 00, so this
prediction leans on a test-set result (the contamination ADR 0037 describes). The rule
decides on player 00's numbers alone, whatever the prediction says.

## Alternatives considered

- **Keep B without a test.** Its only support is a proxy that has failed three times.
- **Revert to A without a test.** That would be choosing by the 360-track test result.
- **A symmetric rule, highest point estimate wins.** With 60 tracks a small difference is
  noise; a challenger that cannot show a clear win does not replace the incumbent, as in
  ADRs 0027 and 0032.

## Consequences

Whichever wins is the first default chosen on GuitarSet-like playing. Its test figure on
players 01–05 is still not an independent check (ADR 0037); the next decision is.
