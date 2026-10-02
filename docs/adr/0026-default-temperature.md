# ADR 0026: The default decoder gets a calibrated temperature

Status: proposed (2026-10-02) — pre-registration; accepted with its result

## Context

E5, the calibration error of the per-note confidences, is the default decoder's weakest
number: 0.3851 end to end on GuitarSet. On 2026-10-01 a temperature calibrated on DadaGP
validation cut the *fitted* decoder's calibration error there by 61% (ADR 0023), but the
default — the hand-set weights — still runs at temperature 1. The temperature changes only
the confidence numbers, never a fingering: the Viterbi path does not depend on it.

## Decision (pre-registered before the run)

- **Hypothesis:** temperature scaling on the artist-disjoint DadaGP validation side
  (ADR 0024) lowers the default decoder's per-note calibration error there. No direction is
  predicted for the temperature itself: the hand window (ADR 0025) spreads probability
  over windows of equal cost, so whether the posteriors come out over- or under-confident
  is not known in advance.
- **Single variable:** the temperature. Weights (hand-set), the window model, songs,
  lattices and span bounds are fixed.
- **Method:** `scripts/calibrate_temperature.py --split artist --decoder-config
  configs/phase1_baseline.yaml`, minimising per-note NLL of the human position; the reported
  metric is calibration error computed as E5 computes it.
- **Adoption:** the fitted temperature is written into `configs/phase1_baseline.yaml` if
  validation calibration error falls; GuitarSet plays no part.
