# ADR 0063: The electric string classifier is adopted as the app's electric-guitar option

Status: accepted (2026-10-06)

Reports Tasks 2–3 of `docs/plans/2026-10-06-electric-audio-evidence.md`; every rule was committed
before its run (4239c9c, dea1d3f, 2701afc, 5d13720). Data: ADR 0062.

## Result

**The classifier** (`cache/acoustic/electric`): Phase 4's network trained from scratch on
Guitar-TECHS's three players and EGFxSet's clean notes — CC BY 4.0 data only — best at epoch 10 of
13. Per-note string accuracy on notes more than one string can sound: EGSet12 0.4665 → 0.5159,
IDMT 0.4201 → 0.4242, against Phase 4's classifier trained on players 1–2 alone; pooled 0.4360 →
0.4555. Calibrated on IDMT's licks: temperature **6.7977** (it is badly overconfident on unfamiliar
audio: NLL 2.24 → 1.15), weight **0.5**.

| E2 | without | with the evidence | change | |
|---|---|---|---|---|
| EGSet12, oracle (validation, judges) | 0.7250 | 0.7926 | +0.0676 [+0.0129, +0.1441] | |
| EGSet12, end to end | 0.4150 | 0.4693 | +0.0543 [−0.0132, +0.1286] | |
| IDMT licks, end to end (chose the calibration) | 0.3166 | 0.3426 | +0.0260 | optimistic |
| **EGDB, oracle (test)** | **0.6762** | **0.7229** | **+0.0467** | predicted +0.02 to +0.07 |
| **EGDB, end to end (test)** | **0.4547** | **0.4992** | **+0.0445** | predicted +0.01 to +0.05 |

Intervals are over takes, on validation only. EGDB's chord-shape rate is unchanged end to end
(19,567 against 19,577 of 19,726). On true notes the open-data classifier slightly beats Phase 4's
SynthTab-started one on EGDB (0.7229 against 0.7200, ADR 0054).

## Decision

- **`configs/decoder_electric.yaml` is the app's electric-guitar option**: the default decoder
  plus the evidence at weight 0.5, temperature 6.7977. By the rule fixed before the look.
- **The default stays `decoder_clean.yaml`.** The evidence was trained, calibrated and judged on
  electric guitar only; on acoustic GuitarSet, Phase 4's audio evidence lost on player 00 (ADR
  0054). GuitarSet was not read for this ADR.
- **Using it needs PyTorch** (the `model` group) **and the trained weights.** Whether those weights
  are published, with attribution to Guitar-TECHS and EGFxSet under CC BY 4.0, is D15: Ege's call.

## Consequences

The electric path is the project's largest end-to-end gain so far (EGDB +0.0445; Phase 5's
thresholds gave +0.0296). It costs a PyTorch dependency and a model on disk, and the user must say
the guitar is electric. EGDB has now been read five times (`experiments/test_set_access.log` lines
31, 32, 33, 35, 36); EGSet12 is one player on twelve takes, and IDMT eleven licks.

**Revisit** when more electric music with string labels exists (GOAT, if access and its licence
fit), or if a single model must serve acoustic and electric guitar alike.
