# ADR 0045: Phase 2's test evaluation — M2 missed by 7.8 points; the comparison on the test players

Status: accepted (2026-10-04) — **Phase 2 closed on this record by Ege's sign-off**

Reports task C6 of `docs/plans/2026-10-01-phase-2.md`, which Ege chose to run after ADR 0044:
Phase 2's one evaluation on GuitarSet's test players 01–05, pre-registered in 65fa8ae
(`configs/m2_c6_eval.yaml`). Two logged looks, nothing chosen from either.

## Today's default on the test players

300 tracks, `audio_mic`; the default decoder `configs/decoder_clean.yaml` — the hand-set weights
plus ADR 0039's open-string cost, T = 1.2934 — beside the hand-set default measured on the same
tracks on 2026-10-03:

| | oracle | end to end | hand-set default |
|---|---|---|---|
| E1 transcriber (raw) | 1.0000 | 0.7493 | 1.0000 / 0.7493 |
| **E2 exact tab F1** | **0.6819** | **0.4418** | 0.6559 / 0.4277 |
| E3 chord shapes | 0.9978 | 0.9906 | 0.9980 / 0.9912 |
| E3 transitions | 0.9996 | 0.9981 | 0.9997 / 0.9984 |
| E4 pitch validity | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| E5 calibration error | 0.0792 | 0.2439 | 0.0794 / 0.2189 |

**ADR 0016's values:** E4 held; chord shapes held (≥ 0.9970 / 0.9896); transitions held end to
end and, in oracle mode, sit exactly on the guardrail at the four decimals it is stated in
(0.9996) — more digits would take a further look; end-to-end E5 held (< 0.3851). **The M2
target, oracle E2 ≥ 0.760, is missed by 7.8 points.** Every prediction for this run held.

**An independent check.** ADR 0039 chose the open-string cost on player 00 alone, so this is
the first test figure since ADR 0037 that checks a choice rather than restating earlier runs:
+0.0260 on the test players, after +0.0209 on player 00. It held up.

## The spec's comparison

Oracle mode, on exactly the groups the default decodes:

| | player 00 (validation) | players 01–05 (test) |
|---|---|---|
| (a) the Viterbi decoder (today's default) | 0.8273 | 0.6819 |
| (b) the learned term alone | 0.4652 | 0.4514 |
| (c) the learned model, decoded by Viterbi | 0.7970 | 0.6856 |

On player 00 the learned model was 0.0303 below the decoder and not preferred (ADR 0044); on the
test players it is 0.0037 above. That prediction failed. No interval is computed on the test
players and nothing is chosen from them: the reversal is one more sign that a single held-out
player is a noisy guide (ADR 0038), not evidence for the learned model. Its weights stay
unpublished in any case (ADR 0042).

## What it means for Phase 2

The spec's Phase 2 gate — "the comparison table exists; a result where the transformer does
not beat Viterbi is still a valid result" — is met, on validation and on test. The plan's own
goal, M2, is not: the decoder improved by 2.6 points in Phase 2's last chunk and stands 7.8
short. Closing the phase on that record, with the miss stated, is Ege's decision.

## Consequences

**Easier.** Phase 2 ends with a measured default, an independent confirmation, and a negative
result for a DadaGP-trained learned model written down.

**Harder.** The test players have now been read for this default and this model; a further
symbolic-only attempt at M2 needs a new pre-registered look, and the record says DadaGP is the
limit, which points at the data of later phases rather than at more of the same.
