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
- [x] Download; check each dataset's labels against its audio (as `check_guitartechs.py` did);
      confirm IDMT's string and fret fields; an ADR for the datasets, their roles and splits.
      *ADR 0062, proposed: loaders in `data/electric.py`, `scripts/check_electric.py`.*

## Task 2: Retrain and calibrate the classifier on publishable data only

- [x] Tools: `scripts/train_strings.py --corpus electric`, `scripts/evaluate_strings.py --electric`,
      `data/electric.py`'s `aligned` (each take's label delay from its audio).
- [x] **Baseline, measured before this pre-registration:** Phase 4's classifier trained on
      Guitar-TECHS players 1–2 (`cache/acoustic/gt`), per-note string accuracy on notes with more
      than one possible string — EGSet12 0.4665 (1,419 notes; decoder alone 0.6963, chance 0.2809),
      IDMT licks 0.4201 (2,730; decoder alone 0.4234, chance 0.2961).
- [x] **Pre-registered training run** (4239c9c, before it started): the same network and
      settings as Phase 4 (Adam 1e-3, batch 256, patience 3, at most 30 epochs, seed 0), trained from
      scratch on ADR 0062's data — Guitar-TECHS players 1–3 and EGFxSet's clean notes, stopping on a
      hashed 15% — into `cache/acoustic/electric`. *Single variable:* the training data.
      *Metric:* the per-note accuracy above, per set and pooled. *Prediction:* above the baseline
      by at least 0.03 on each set; pooled between 0.47 and 0.55. *Rule:* Task 3 uses the new
      classifier if its pooled accuracy beats the baseline's, else Phase 4's (also publishable).
      *Result:* 14,214 training notes, 2,751 to stop on; best epoch 10 of 13 (stop-set accuracy
      0.7754 — same guitars as training, so not a measure of anything else). **EGSet12 0.5159**
      (+0.0494, held), **IDMT licks 0.4242** (+0.0041, the +0.03 prediction failed); pooled
      **0.4555** against 0.4360 (1,890 against 1,809 of 4,149), below the predicted 0.47–0.55.
      **By the rule, Task 3 uses `cache/acoustic/electric`.** On IDMT neither classifier beats the
      decoder alone (0.4234); on EGSet12 the decoder alone (0.6963) is far ahead of both.
- [x] **Pre-registered calibration and judgement** (dea1d3f): calibrate on IDMT's licks,
      judge on EGSet12, so the set that chooses is not the set that judges.
      `scripts/calibrate_acoustic.py --corpus idmt` fits the temperature by the NLL of IDMT's notes
      and picks the acoustic weight among 0, 0.1, 0.25, 0.5, 1.0 that recovers the most labelled
      strings when the default decoder decodes IDMT's true notes (the smaller on a tie); `--judge`
      then decodes EGSet12's true notes without and with the evidence. *Single variable:* the
      audio evidence. *Metric:* EGSet12's oracle string recovery (oracle E2), change with a
      bootstrap interval over its 12 takes. *Prediction:* a weight above 0 is chosen; EGSet12
      changes by 0 to +0.04 (EGDB gained +0.044 in Phase 4, but here the decoder alone already
      recovers 0.70 of the ambiguous notes and the classifier 0.52, and EGSet12 is microphone on
      an amplifier, not direct input). *Rule:* Task 3's end-to-end work goes ahead if EGSet12's
      change is above 0; otherwise this path stops and is reported.
      *Result:* temperature **6.7977** (IDMT NLL 2.2398 → 1.1544: the classifier is badly
      overconfident on audio unlike its training), weight **0.5** (IDMT 0.4401 → 0.4554).
      **EGSet12: 0.7250 → 0.7926, +0.0676 [+0.0129, +0.1441]** — the prediction's upper bound
      failed; the rule's condition held. Task 3 goes ahead.

## Task 3: The audio term end to end

- [x] The classifier heard at transcribed notes' onsets, inside `transcribe_path`, behind an
      electric decoder config; PyTorch imported only on that path. *`model/evidence.py`, the
      config's `evidence` block, `configs/decoder_electric.yaml` (differs from the default only in
      the evidence, checked).*
- [x] **Pre-registered end-to-end measurement** (2701afc): `scripts/evaluate_electric.py` —
      Basic Pitch at `CHOSEN_PARAMS`, each take decoded with `decoder_clean` and `decoder_electric`.
      *Single variable:* the evidence. *Metric:* end-to-end E2 on EGSet12 (judges), change with a
      bootstrap interval over takes; IDMT reported, optimistic. *Prediction:* EGSet12 gains +0.01
      to +0.05 — less than oracle's +0.068, since the classifier now hears transcribed notes, some
      of them wrong. *Rule:* one pre-registered look at EGDB if EGSet12's change is above 0.
      *Result:* **EGSet12 0.4150 → 0.4693, +0.0543 [−0.0132, +0.1286]** — the interval includes
      zero; the prediction failed narrowly (above). IDMT, optimistic: 0.3166 → 0.3426, +0.0260
      [−0.0084, +0.0601]. The rule's condition held: the EGDB look goes ahead.
- [x] **The EGDB look, pre-registered** (5d13720) (`configs/electric_test_eval.yaml`):
      `scripts/evaluate_egdb.py --config configs/electric_test_eval.yaml --run cache/acoustic/electric
      --weight 0.5 --temperature 6.7977 --e2e-audio` — oracle and end to end, each without and with
      the evidence, in one run. *Predictions:* oracle +0.02 to +0.07 from 0.6762; end to end +0.01 to
      +0.05 from 0.4547. *Adoption rule:* if the end-to-end change is above 0, `decoder_electric`
      becomes the app's electric-guitar option (the default stays acoustic-safe); otherwise it is
      not offered. GuitarSet is not read. An ADR either way.
      *Result (log line 36):* **oracle 0.6762 → 0.7229 (+0.0467); end to end 0.4547 → 0.4992
      (+0.0445)** — both predictions held; adopted as the electric option (ADR 0063).

## Task 4: The app

- [ ] An "electric guitar" choice on the page and the CLI (`--config`).
- [ ] D15 for these weights (Ege): publish with attribution, or keep local.

## Not in this plan

- Phase 6 (full songs); Basic Pitch fine-tuning (ADR 0056).
