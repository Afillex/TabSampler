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

- [ ] Tests on a synthetic JAMS fixture; a check script on the real files; commit.

## Task 3: The note windows and the string classifier (`src/tabsampler/audio/`, `model/`)

A constant-Q window around each onset, the classifier, and its training on SynthTab's
development set split by song, stopping on the held-out songs.

- [ ] Tests (offline, synthetic audio); training script; per-note string accuracy on held-out
  SynthTab songs and — the real question — on player 00's notes.

## Task 4: The acoustic term in the decoder (ADR)

`−acoustic × log P(string | audio)` added to each note's cost, the probabilities supplied per note
without changing what the decoder does when the weight is zero. Oracle-checked like every other
term.

## Task 5: The ablation on player 00, then the test players once

Pre-registered: oracle E2 with the acoustic term against the same decoder without it (Phase 2's
default), on player 00; then both on players 01–05, once.
