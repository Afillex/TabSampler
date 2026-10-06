# ADR 0060: Automatic tuning correction is not adopted

Status: accepted (2026-10-06)

Reports the app plan's Task 6 (`docs/plans/2026-10-06-app-track.md`), pre-registered in 30f071b.

## Context

Ege's own recording failed the app track's gate: the tab "didn't resemble the original sound". The
recording sits 0.43–0.45 semitone off A440, and shifting it back more than doubled the notes Basic
Pitch found (29 → 71). Ege chose automatic correction: estimate the offset from the recording's
harmonic part (`librosa.estimate_tuning`), and when |offset| ≥ 0.1 semitone pitch-shift the audio
back before transcribing (`tabsampler/audio/tuning.py`).

## Result

Guitar-TECHS's player 3 (validation, 12 takes, direct input), each take pitch-shifted by d to
simulate a detuned guitar; the default decoder end to end; Basic Pitch at `CHOSEN_PARAMS`.

| d (semitones) | E2, correction off | E2, on | on − off [95% interval] |
|---|---|---|---|
| 0 | 0.4173 | 0.4046 | −0.0127 [−0.0374, +0.0014] |
| +0.25 | 0.4226 | 0.4080 | −0.0146 [−0.0442, +0.0251] |
| +0.45 | 0.3044 | 0.3603 | +0.0559 [−0.0683, +0.1494] |
| −0.45 | 0.1141 | 0.0996 | −0.0145 [−0.0832, +0.0772] |

**Every hypothesis failed, and so did the adoption rule fixed before the run** (corrected ≥
uncorrected at every d, and no worse than −0.01 at d = 0):

- **H1 (no harm in tune) failed:** −0.0127. Player 3's guitar itself reads −0.03 to −0.17, so four
  of twelve in-tune takes crossed the threshold and were shifted; Basic Pitch had handled them.
- **H3 failed:** at +0.25 Basic Pitch loses nothing uncorrected (0.4226), and correction costs.
- **H2 failed:** at +0.45 correction helps (+0.056), but the interval includes zero and it stays
  0.057 below the in-tune figure.
- **The boundary risk is real.** At d = −0.45 the takes, already about −0.07, sit past −0.5: the
  estimator read eight of twelve as +0.42 to +0.49 sharp — the wrong way — and correction moved
  them towards a whole semitone off. Uncorrected is as bad (0.1141): a guitar half a semitone off
  has no right answer at standard pitch.

The detuning is synthetic (a phase vocoder), and the corrected arm processes the audio twice. The
uncorrected in-tune figure, 0.4173, is 0.0012 below Phase 5's 0.4185 for the same setting because
this script re-writes each take as a mono WAV before transcribing.

## Decision

**The pipeline does not correct tuning.** `estimate_offset` and `retune` stay in the package,
tested, for the record and for any later use; nothing in the pipeline calls `retune`.

## Consequences

Ege's recording, about +0.45 semitone off, stays outside what the system handles well. What the app
does about such recordings — warn, or a narrower correction measured afresh — is Ege's decision.
Any new rule must be measured on fresh validation data or declared as chosen on these 12 takes;
choosing a threshold from this table and scoring it on the same takes would flatter it.
