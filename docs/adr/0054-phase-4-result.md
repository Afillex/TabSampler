# ADR 0054: Phase 4's result — on electric guitar the audio evidence helps; on acoustic it does not

Status: proposed — Phase 4 closes on this record if Ege signs off

Reports Phase 4 (`docs/plans/2026-10-06-phase-4-electric.md`), clean electric first (ADR 0049).
The spec's "done when": results on both test sets, including the gap between electric and
acoustic guitar.

## Result

The evidence: ADR 0046's string classifier fine-tuned on Guitar-TECHS's players 1–2 (real
electric direct input, ADRs 0051–0052) from SynthTab's weights, calibrated on player 3
(temperature 0.6957, acoustic weight 0.25). On its own it picks the right string for 0.3739 of
player 3's ambiguous notes (chance 0.2657) — no better than SynthTab's classifier (0.3628), nor than
the same network trained from scratch (0.3739).

Oracle E2 of the default decoder without the evidence and with it:

| | sound | without | with | change |
|---|---|---|---|---|
| Guitar-TECHS player 3 (validation; string recovery, not E2) | electric | 0.5795 | 0.6231 | +0.0436, on the takes the weight was chosen on |
| GuitarSet player 00 (validation) | acoustic | 0.8273 | 0.8044 | −0.0230 [−0.0468, −0.0035] |
| **GuitarSet players 01–05 (test)** | acoustic | **0.6819** | **0.6876** | **+0.0057** |
| **EGDB, 240 clips (test)** | electric | **0.6762** | **0.7200** | **+0.0438** |

No interval is computed on either test set, and nothing is chosen from them.

**The electric-to-acoustic gap, for the decoder alone, is small**: 0.6762 on EGDB against 0.6819
on GuitarSet's test players. End to end on EGDB: Basic Pitch's E1 (raw) 0.7230, E2 0.4251
(530 of 43,700 transcribed notes not placed), beside GuitarSet's 0.7493 and 0.4418 (ADR 0045).

**With the evidence the sound matters.** On electric guitar — the sound it was trained and
calibrated on — it raised the test set's oracle E2 by 4.4 points, the first gain from audio in
this project; on acoustic guitar it lost on player 00 and was within noise on the test players,
where the sign reversed, as Phase 2's learned model's did (ADR 0045). Of Phase 4's 13
pre-registered predictions, 5 failed, each stated in `results.csv`: both classifiers' accuracy on
player 3, the temperature (below 1), GuitarSet's test players (a small gain where a loss was
predicted) and EGDB's gain (above the predicted −0.02 to +0.04, by 0.004).

## Decision

- **The default decoder is unchanged**: hand-set weights, acoustic weight zero. Whether
  electric input should get the audio term — a decoder config of its own, as distorted guitar
  has (ADR 0032) — is Ege's decision, and the only electric validation figure for it is the
  calibration's own, so it would need a fresh validation measurement first.
- **The spec's Phase 4 deliverable exists**: both test sets reported, with the electric-to-acoustic
  gap. Closing the phase on it is Ege's decision.

## What it does not settle

Why the classifier helps the decoder on electric guitar while its per-note accuracy is low: its
errors may fall where the decoder is already sure, and its right answers where the decoder is
not. Unmeasured. EGDB is one player on one guitar, so its +0.044 says little about electric
guitar in general; GOAT, whose access was unclear, would be a second electric corpus.

## Consequences

**Easier.** An electric path with audio evidence is measured, and every part of it — loaders,
label checks, the split, the calibration — is in place for more electric data.

**Harder.** Both test sets have now been read with this evidence. Another attempt needs its own
pre-registered look at each.
