# Electric guitar: audio evidence for string choice

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every measured choice is
> pre-registered in a commit before its run.

**Status:** planned 2026-10-06. Ege chose this path after the optimisation round (ADR 0061), over
Phase 6.

**Goal:** better string choice on electric guitar — the largest measured loss (EGDB: 0.32 of E2
even on the true notes) — with weights that may be published and an app that can use them.

## Where this starts

- **The one measured gain:** Phase 4's string classifier, fine-tuned from SynthTab, raised EGDB's
  oracle E2 0.6762 → 0.7200 (ADR 0054). Not the default: its weights start from SynthTab's
  (unpublishable, ADR 0046's terms), its only electric validation figure is its own calibration's,
  and it was never measured end to end.
- **A publishable candidate exists:** the same network trained from scratch on Guitar-TECHS players
  1–2 (CC BY 4.0), `cache/acoustic/gt`: 0.3739 per-note accuracy on player 3, as good as the
  fine-tuned one, never calibrated or put in the decoder.
- **The bottleneck is data.** Training: 14,788 notes of drills from two players; the network
  overfits after one epoch. Validation: player 3 is Guitar-TECHS's only music, and it has chosen
  Phase 4's calibration, Phase 5's thresholds and Task 3b's filter.
- PyTorch is an optional group (`model`, ADR 0041); the app has no audio-evidence path.

## Task 1: Find more real electric guitar with string labels (needs web research)

Search with the self-hosted crawler for datasets that have: real recorded electric guitar (direct
input or amplified), per-note string and fret labels, a licence that allows training **and**
publishing weights, and a working download. Known by name, unverified: GOAT (access unclear),
IDMT-SMT-Guitar, others the search finds. Every claim gets its source.

- [x] A table: dataset, audio, labels, size, licence, access, source link. To Ege (2026-10-06).

| dataset | audio | string labels | size | licence, access | could serve as |
|---|---|---|---|---|---|
| **GOAT** (ISMIR 2025) | real electric, **direct input**, several guitars and players | tablature (Guitar Pro) and aligned MIDI | 5.9 h unique (29.5 h with amp renders) | **CC BY-NC 4.0**, restricted: by request, "research purposes only", "not intended for use in any commercial product" | training (weights then non-commercial) |
| **EGFxSet** (ISMIR 2022) | real Stratocaster, direct input; also through 12 hardware effects | string–fret per file, every note in standard tuning, 5 pickup settings | 8,970 five-second notes in all; clean set 431 MB | **CC BY 4.0**, open | training (isolated notes only) |
| **EGSet12** (DAFx 2024) | real solo electric music, Telecaster through an amp, **microphone** | Guitar Pro and JAMS per piece | 12 pieces, 380 s, one player | **CC BY 4.0**, open | validation (music, another player and guitar) |
| **IDMT-SMT-Guitar** (DAFx 2014) | real electric (and acoustic), mostly direct input | XML with `stringNumber`/`fretNumber` per a third-party parser — **to confirm on the files** | subset 1: ~4,700 notes of licks, 3 guitars | **CC BY-NC-ND 4.0**, open, "for evaluation purpose" | validation only (no-derivatives rules out training) |

Sources: github.com/JackJamesLoth/GOAT-Dataset and Zenodo record 15690894 (licence via Zenodo's API);
egfxset.github.io and Zenodo 7044411; Zenodo 11406378; idmt.fraunhofer.de/en/publications/datasets/guitar.html,
Zenodo 7544110 and github.com/bakerbass/HarmonicsClassifier (`idmt_parser_prompt.md`).
Ruled out: hegelespaul/Electric-Guitar-Dataset (no licence, no download); EGDB and GuitarSet (our
test sets); SynthTab (rendered, unpublishable); SCORE-SET and DadaGP (no audio).
- [x] **Ege picked open data only (2026-10-06):** train on Guitar-TECHS players 1–2 + EGFxSet
      (CC BY 4.0, so the weights may be published with attribution); validate on EGSet12 +
      IDMT-SMT-Guitar. GOAT not requested.
- [ ] Download; check each dataset's labels against its audio (as `check_guitartechs.py` did);
      confirm IDMT's string and fret fields; an ADR for the datasets, their roles and splits.

## Task 2: Retrain and calibrate the classifier on publishable data only

- [ ] Train from scratch on Guitar-TECHS players 1–2 plus Task 1's training part.
- [ ] Calibrate temperature and acoustic weight on the new validation set (not player 3).

## Task 3: The audio term end to end

- [ ] The classifier heard at transcribed notes' onsets, inside `transcribe_path`, behind an
      electric decoder config; PyTorch imported only on that path.
- [ ] Pre-registered measurement on the new validation set, oracle and end to end; then one
      pre-registered look at EGDB.

## Task 4: The app

- [ ] An "electric guitar" choice on the page and the CLI (`--config`).
- [ ] D15 for these weights (Ege): publish with attribution, or keep local.

## Not in this plan

- Phase 6 (full songs); Basic Pitch fine-tuning (ADR 0056).
