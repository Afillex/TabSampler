# Phase 4 — Real-Data Fine-Tuning, Clean Electric First

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every run that produces a
> reported number is pre-registered in a commit made before it runs.

**Status:** started 2026-10-06, after Ege closed Phase 3 (ADR 0048) and decided D2: clean
electric guitar first (ADR 0049), Guitar-TECHS first.

**Goal (spec §6):** train on real recorded guitar; evaluate on GuitarSet and EGDB, both held out.
**Done when:** results are reported on both test sets, including the gap between electric and
acoustic guitar.

## Where this starts

- **Phase 3's pipeline**: note windows (`audio/windows.py`), the string classifier
  (`model/strings.py`), temperature scaling, the acoustic term (ADR 0047), and the scripts that
  train, calibrate and ablate it. Trained on SynthTab it cost the decoder on GuitarSet (ADR 0048),
  with rendered-against-real and electric-against-acoustic confounded.
- **Data:** Guitar-TECHS (Zenodo 14963133, CC BY 4.0, 4.1 GB), downloaded 2026-10-06 to
  `data/guitar-techs/` (gitignored). Three players on electric guitars; per recording a
  direct-input WAV, an amp-microphone WAV, two room microphones (MP3) and one MIDI file from a
  multi-track pickup, **one MIDI track per string** (`e B G D A E`, 960 ticks per beat). Players
  1 and 2 recorded single notes, techniques, scales and chords; player 3 recorded musical excerpts.
- **Evaluation:** GuitarSet's player 00 stays the acoustic validation player (ADR 0037); its
  players 01–05 and EGDB are read once, at the end, pre-registered.

## Global constraints

- One variable per experiment; the hypothesis committed before the run.
- A multi-track MIDI pickup tracks pitch from each string and can mistrack — ghost notes, octave
  slips, late onsets. **The labels are checked against the audio before anything trains on them**,
  and whatever is dropped is reported.
- Weights trained only on Guitar-TECHS are CC BY 4.0 data's; whether to publish them is D15, Ege's.
  Weights that start from SynthTab's stay unpublished (ADR 0046).

## Task 1: D2 (ADR)

- [x] Clean electric first; acoustic measured separately. *(ADR 0049, Ege.)*

## Task 2: Guitar-TECHS as notes and audio (`src/tabsampler/data/guitartechs.py`)

Each string's MIDI track → notes with onset (seconds, under the file's tempo map), pitch, string
and fret = pitch − the string's open pitch in standard tuning.

- [x] Tests on a synthetic MIDI fixture; commit.
- [x] A check script on the real files: pitch ranges per string; overlapping notes on one string;
      the onset lag against the DI audio, measured as Phase 3 measured SynthTab's, with GuitarSet's
      player 00 as the control; and the share of labelled pitches the audio confirms. Report;
      correct for a constant lag, drop what fails the check, and say how much.
      *(Labels 23/16/15 ms late for players 1/2/3; overlaps a 3–4 ms note-off artefact; most
      unconfirmed notes are glitches under 60 ms. Rule and figures: ADR 0051.)*

## Task 3: The split (ADR)

Players 1 and 2 train; **player 3 is validation** — player-disjoint, as ADR 0037 keeps GuitarSet's
players apart, and its musical excerpts are the closest thing here to what a user plays. The DI
signal is the clean-electric input; the amp microphone is kept for a later robustness check.

- [x] Write the ADR; index it. *(ADR 0051.)*

## Task 4: EGDB as the second test set

- [ ] Obtain EGDB; a loader; add it to `data/splits.py` with the snapshot-and-guard mechanism
      GuitarSet has — every EGDB track is test, refused by `assert_tuning_allowed`, every look
      logged. The guard is not weakened to fit a second test set.

## Task 5: The string classifier on real electric audio

Same architecture and windows as Phase 3. **One variable: the training data** — Guitar-TECHS
players 1–2 (DI) instead of SynthTab — trained from scratch. A second run, pre-registered on its
own, starts from SynthTab's weights instead (the variable: initialisation).

- [x] Pre-register: per-note string accuracy on player 3 and on GuitarSet's player 00, against
      SynthTab's classifier on the same notes.

**Pre-registered (run 1, from scratch):** `scripts/train_strings.py data/guitar-techs --corpus
guitartechs --run cache/acoustic/gt`, everything else as Phase 3's run (Adam 1e-3, batch 256,
patience 3, seed 0); best epoch by NLL on player 3. Measured on notes more than one string can
sound: on player 3, the new classifier and SynthTab's (`cache/acoustic/dev`) through
`scripts/evaluate_strings.py --examples cache/acoustic/gt/examples.npz`; on GuitarSet's player 00
(`audio_mic`), the new one through `scripts/evaluate_strings.py`, beside SynthTab's 0.4736.

- **Hypothesis 1:** on player 3, the classifier trained on real electric audio beats SynthTab's.
  Prediction: new 0.55–0.75, SynthTab's 0.35–0.55. Player 3 also chooses the epoch, so the new
  figure is slightly optimistic; SynthTab's is not.
- **Hypothesis 2:** on GuitarSet's acoustic microphone recordings, it does worse than SynthTab's
  classifier, which saw some acoustic tones: prediction 0.30–0.47, against 0.4736.

Together they put numbers on the two halves of Phase 3's confound: rendered against real
(hypothesis 1) and electric against acoustic (hypothesis 2).
- [ ] Train; measure; record.

## Task 6: Calibrate on player 3

Phase 3's rule (35a14b2), on player 3 instead of SynthTab's held-out tracks: the temperature by
NLL on player 3's notes, the acoustic weight by the decoder's recovery of the labelled string,
smaller on a tie.

- [ ] Pre-register; run; record.

## Task 7: The ablation on GuitarSet's player 00

The decoder with and without the calibrated evidence, oracle E2, ADR 0039's rule. Set beside
Phase 3's −0.0408, it measures electric-to-acoustic transfer with rendered-against-real removed.

- [ ] Pre-register; run; record.

## Task 8: One look at both test sets

- [ ] Pre-register: GuitarSet players 01–05 and EGDB, oracle E2 with and without the evidence;
      end to end beside it. Run once; write Phase 4's result (ADR) with the electric-to-acoustic
      gap; README, HANDOFF, devlog. Phase 4 then goes to Ege for sign-off.
