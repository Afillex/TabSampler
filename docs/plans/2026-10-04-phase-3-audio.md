# Phase 3 — Audio Conditioning → M3

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Status:** started 2026-10-04, after Ege closed Phase 2 (ADR 0045) and chose Phase 3.

**Goal (spec §6):** pick the form of audio evidence (D13), pretrain on SynthTab, add the acoustic
term to the decoder. **Done when:** a measured change in oracle-mode Exact Tab F1 against Phase 2,
with an ablation that removes the audio input. The ablation is the deliverable, not the gain.

## Where this starts

- The decoder's cost has an acoustic weight, `CostWeights.acoustic`, identically zero and unused
  until now (spec §2.2: −log P(string | audio)).
- **Data:** SynthTab's development set, downloaded 2026-10-04 to `data/synthtab/` (gitignored):
  169 songs from DadaGP, each rendered once in one of 57 guitar tones — 8.9 hours of mono
  22.05 kHz audio, 21 songs and 0.93 hours of it acoustic — with per-string JAMS labels (string
  and fret per note). CC BY-NC 4.0. The full set is 2 TB; one of its archives needs more disk
  than the 33 GB free, so more SynthTab is a later decision for Ege, made only if the
  development set proves too small.
- **Evaluation:** GuitarSet's player 00 (validation, ADR 0037) for every choice; the test players
  once, at the end, pre-registered. `audio_hex` stays forbidden as input (ADR 0005): the
  acoustic model hears `audio_mic` only.

## Global constraints

- Players 01–05 are read once, at the end, pre-registered.
- Weights trained on SynthTab, which is derived from DadaGP, stay unpublished like ADR 0042's.
- One variable per experiment; the acoustic weight is fixed in advance or chosen on player 00 by a
  rule committed first.

## Task 1: D13 — the form of the audio evidence (ADR)

Per note: the probability of each string, from a small CNN over a constant-Q window around the
note's onset, given the note's pitch. That is D13's cue (b) — string probabilities that plug into
the acoustic term — computed per note rather than per frame, because the decoder scores notes.

- [x] Write the ADR; index it. *(ADR 0046.)*

## Task 2: SynthTab as notes and audio (`src/tabsampler/data/synthtab.py`)

JAMS `note_tab` annotations per string → notes with onset (seconds), pitch, string and fret; the
times are in ticks with a tempo annotation, so their unit must be **verified against the audio**
(onsets should line up with energy rises), not assumed.

- [x] Tests on a synthetic JAMS fixture; a check script on the real files; commit.
  *(168 usable tracks, 146,877 notes; 3 skipped — two seven-string, one without labels.
  Ticks are 960 to the quarter note. **SynthTab's audio lags its labels**: measured at
  5.8 ms frames, the onset-strength peak trails the labelled onsets by 29.0 ms on electric
  tracks and 40.6 ms on acoustic ones, against 11.6 ms on GuitarSet's player 00, whose
  onsets are trusted — so about 17 ms and 29 ms of real latency. Task 3 corrects for it
  when it cuts the audio windows.)*

## Task 3: The note windows and the string classifier (`src/tabsampler/audio/`, `model/`)

**The window, centred on the note's pitch.** A constant-Q transform of the whole track (22,050 Hz,
hop 256 — 11.6 ms; two bins per semitone from MIDI 28 to 124), cropped per note to the 48
semitones from an octave below its pitch to three octaves above, and to the frames from 35 ms
before its onset to 300 ms after. Centring on the pitch leaves what differs between strings —
the harmonics' balance, the attack — in the same place for every note, so the network learns
timbre rather than pitch; the pitch itself goes in beside it. Log magnitude, normalised per
window.

**SynthTab's latency is corrected where the windows are cut**: onsets move later by 29 ms on
acoustic tracks and 17 ms on electric ones (Task 2's measurement); GuitarSet's onsets are not
moved.

**The classifier.** A small CNN over the window, the pitch beside it, six logits, and a softmax
over only the strings that can sound the pitch in the track's tuning. Trained on SynthTab's
development set, all four families, split by track (about 85% to train, 15% to stop on), Adam
1e-3, early stopping on the held-out tracks' negative log-likelihood, seed 0.

**What it is judged by, before it touches the decoder:** per-note string accuracy, among the
notes with more than one possible string, on the held-out SynthTab tracks and on player 00's
notes from `audio_mic` — against two floors, chance among the possible strings and always
picking the string the decoder's cost model alone would prefer.

- [ ] Tests (offline, synthetic audio): the window's shape and centring, the latency shift, the
  mask over possible strings; the classifier's masked softmax.
- [x] Training script, checkpointing as `scripts/train_model.py` does (`scripts/train_strings.py`).
- [x] **The run, pre-registered:** `scripts/train_strings.py data/synthtab/SynthTab_Dev --run
  cache/acoustic/dev`, the script's settings. Hypotheses: (1) among notes more than one string
  can sound, per-note string accuracy on SynthTab's held-out tracks is well above chance; (2)
  on player 00's notes, heard through `audio_mic`, it is above chance too, though the distance
  from rendered to real guitar will cost much of the gap. Predictions: held-out SynthTab 0.60
  to 0.85 against a chance of about 0.3 to 0.4; player 00 0.40 to 0.60 against a similar
  chance. The player-00 figure is measured once, after training, by a script committed
  before it runs.
  *(SynthTab held out: 0.5946 against chance 0.2884, best epoch 1 of 4 — the range missed by
  0.005. Player 00: the classifier 0.4736, the default decoder 0.8181, chance 0.2758 — both
  hypotheses held.)*

## Task 4: The acoustic term in the decoder (ADR)

`−acoustic × log P(string | audio)` added to each note's cost, the probabilities supplied per note
without changing what the decoder does when the weight is zero. Oracle-checked like every other
term.

- [x] ADR 0047; `HandSetScorer.evidence`; scorer tests and an oracle test with the term on.

## Task 5: The ablation on player 00, then the test players once

Pre-registered: oracle E2 with the acoustic term against the same decoder without it (Phase 2's
default), on player 00; then both on players 01–05, once.

- [x] **The run on player 00, pre-registered:** `scripts/evaluate_acoustic.py --run
  cache/acoustic/dev --weight 1.0 --out cache/validation/p3`, then `compare_validation.py` on
  `a.json` and `c.json`. The weight is fixed at **1.0**, not chosen: log-probabilities in nats are
  already on the cost model's temperature-1 scale, and choosing it on player 00 and then judging
  on player 00 would be circular. **Hypothesis:** the audio evidence raises oracle E2 — the 95%
  track-level interval wholly above zero, and no clear chord-shape drop (ADR 0039's rule).
  **Prediction:** uncertain, between −0.02 and +0.03: the classifier hears the right string 47%
  of the time where the decoder's context gets 82%, so it helps only where the decoder is
  unsure, and a confident wrong string can hurt. Whatever the sign, the measured change is the
  M3 deliverable.
  *(Measured: 0.8273 → 0.7001, −0.1273 [−0.1834, −0.0729], with a clear chord-shape drop:
  the audio hurts at weight 1.0. The hypothesis and the prediction failed.)*
