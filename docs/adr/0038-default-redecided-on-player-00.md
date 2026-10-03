# ADR 0038: The default decoder, re-decided on the validation player

Status: accepted (2026-10-03) — **the default returns to the hand-set weights**

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

## Result (2026-10-03)

Player 00, 60 tracks, 13,223 reference notes; track-level paired bootstrap, exact counts:

| | A, hand-set | B, clean-fitted | B − A, 95% interval |
|---|---|---|---|
| **E2 oracle** | 0.8065 | 0.8297 | +0.0232 [−0.0030, +0.0513] |
| **E3 chord shapes, oracle** | 6543 / 6607 | 6538 / 6607 | drop 0.00076 (0.0005 allowed) |
| E2 end to end | 0.4861 | 0.4830 | −0.0031 [−0.0259, +0.0212] |
| E3 transitions, oracle / e2e | 0.9997 / 0.9952 | 0.9998 / 0.9948 | |
| E5, oracle / e2e (their own temperatures) | 0.2336 / 0.1309 | 0.0249 / 0.2715 | |

**B fails both conditions**: its gain's interval includes zero, and it lowers the
chord-shape rate by 0.00076, more than the 0.0005 allowed. So `configs/decoder_clean.yaml` holds the hand-set
weights again, with **T = 1.5728**, calibrated for them on DadaGP's clean validation parts
(calibration error 0.2153 at 2.9974 → 0.0941). On player 00 that default scores oracle /
end-to-end E5 0.1178 / 0.2100, and E2 as A above. The clean-fitted weights are kept in
`configs/fitted_clean_dadagp.yaml` so their published GuitarSet figures stay reproducible.

**The prediction held only through the rule.** On player 00 the clean fit's point estimate
was *better* by 2.3 points — the opposite of its −1.9 over all 360 tracks — so the other
five players must favour the hand-set weights more strongly. Players differ, which is one
more reason a single held-out player is a noisy guide, and why the rule asks a challenger
for a clear win.
