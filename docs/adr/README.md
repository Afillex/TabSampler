# Architecture Decision Records

One record per decision that is hard to reverse or would need explaining later.
Template: `0000-template.md`.

**Never edit an accepted ADR.** Supersede it with a new one and update the status line
of the old record to point at its replacement.

| ADR | Decision | Spec ref | Status |
|---|---|---|---|
| [0001](0001-python-version-and-isolated-transcriber.md) | Python 3.13 core; basic-pitch isolated on 3.11 behind a subprocess and cache | — (forced) | accepted |
| [0002](0002-v1-input-scope.md) | v1 accepts isolated guitar audio only | D1 | accepted |
| [0003](0003-evaluation-protocol-guitarset-held-out.md) | **GuitarSet is test-only**, enforced in `data/splits.py` | D8 | accepted; amended by 0037 |
| [0004](0004-headline-metric.md) | Headline = end-to-end Exact Tab F1 on GuitarSet; target set after M1 | D10 | accepted |
| [0005](0005-guitarset-audio-channel.md) | Evaluate on `audio_mic`; hex channels forbidden as input | — (gap) | accepted |
| [0006](0006-reference-pitch-rounding.md) | Round float MIDI to int for the reference tab; keep floats for E1 | — (gap) | accepted |
| [0007](0007-immutable-collections-in-contracts.md) | Contract collections are tuples; `NoteGroup`/`ChordState`/`Context` defined | 2.1 (deviation + gap) | accepted |
| [0008](0008-tunings.md) | User-chosen tuning and capo from day one; no auto-detection | D3 | accepted |
| [0009](0009-timing-only-tab.md) | Time-positioned tab for v1, not rhythmic notation | D5 | accepted |
| [0010](0010-chord-level-decoder-states.md) | Decoder states are chords, with span pruning | D6 | accepted |
| [0011](0011-playability-rules.md) | Concrete E3 playability rules (validation pending DadaGP) | D9 | accepted; finger rule superseded by 0019; validated by 0022; transition rule superseded by 0025, 0029, 0031 |
| [0012](0012-cost-weights-hand-set-for-m1.md) | **M1 ships hand-set weights**; tuning gated on legal data | D7 | accepted; premise lifted by 0023 |
| [0013](0013-output-formats.md) | ASCII + JSON in Phase 1; MusicXML/GP on the app track | D4 | accepted |
| [0014](0014-brute-force-oracle.md) | **The brute-force oracle is the decoder's specification** | 2.2 (gap) | accepted |
| [0015](0015-no-cpp-decoder.md) | No C/C++ decoder port — decode is 3% of runtime | D16 | accepted |
| [0016](0016-headline-target.md) | D10 target: **oracle E2 >= baseline + 10 points** at M2, with E3/E4/E5 guardrails | D10 | accepted |
| [0017](0017-rhythm-for-exports.md) | Exports use a fixed 120 BPM grid and carry "rhythm is not transcribed" **in the file** | D5 (gap) | accepted |
| [0018](0018-hand-position-carry.md) | Carry the hand position across all-open shapes; `transition_cost_from` added to the scorer contract | 2.1/2.2 (defect) | accepted; hand position superseded by 0025 |
| [0019](0019-barre-chords-in-e3.md) | E3 counts **fingers**, not fretted notes, so barre chords are playable (supersedes ADR 0011's fourth rule) | D9 (defect) | accepted |
| [0020](0020-licensing-and-publication.md) | **MIT for the code, public repo**; weights decided per training corpus | D15 (partial) | accepted |
| [0021](0021-dadagp-training-protocol.md) | **DadaGP v1.1 is the training corpus**: shipped split frozen by hash; clean standard 6-string songs only | D8 | accepted; split superseded by 0024 for fitting |
| [0022](0022-playability-rules-validated.md) | ADR 0011 validated on 16.8M human shapes: **chord rules hold (99.86%), the speed rule does not (88%)** | D9 | accepted; its window implemented in 0025; caveat lifted by 0031 |
| [0023](0023-fitted-cost-weights.md) | Weights fitted on DadaGP: **+3.9 E2 and −63% E5 on GuitarSet, but not the default** — worse on clean guitar | D7 | accepted; open question settled by 0027 |
| [0024](0024-artist-disjoint-split.md) | **Artist-disjoint DadaGP split** for every fit: 519 artists / 2,607 songs to validation, frozen by hash | D8 | accepted |
| [0025](0025-hand-window.md) | **The hand is a 4-fret window**: decoder, cost model and E3 (gate missed at 0.9795, kept by Ege's decision) | 2.2 / D9 | accepted; second check in 0028; wide-shape anchoring superseded by 0030 |
| [0026](0026-default-temperature.md) | Default decoder's temperature calibrated on artist validation: **T = 2.9974, ECE −62%** | D7 | accepted |
| [0027](0027-fitted-weights-fair-test.md) | Fair test for fitted weights: **failed** (clean and E3 lose) — hand-set stays the default | D7 | accepted |
| [0028](0028-hand-window-kept.md) | The hand window stays after **missing its validation check** by 0.0023 on distorted parts (Ege's decision) | 2.2 / D9 | accepted |
| [0029](0029-e3-move-timing.md) | E3 times a hand move from the **last fretted group**, not from an open group in between: human tab 0.9795 → 0.9823 | D9 (defect) | accepted |
| [0030](0030-hand-stretch.md) | A chord wider than the window **stretches the hand** instead of anchoring it (supersedes one sentence of 0025); recovery unchanged | 2.2 / D9 | accepted; a claim corrected by 0036 |
| [0031](0031-e3-speed-limit.md) | E3's speed limit set from human tab: **48 frets/s**; 0.9988 of unseen artists' transitions pass | D9 | accepted; wording corrected by 0036 |
| [0032](0032-style-decoders.md) | Clean and distorted guitar get **a decoder each**; clean-fitted weights win on clean parts (0.8257 → 0.8616) and become the default; distorted keeps hand-set (chord shapes −0.00065) | D7 | accepted; clean weights re-decided by 0038 |
| [0033](0033-per-style-temperature.md) | Each style's decoder gets **its own temperature**: clean T = 1.1975, distorted T = 4.2982 | D7 | accepted |
| [0034](0034-c3-feature-groups.md) | Two richer feature groups — **per-string preference** and **fret regions**: only regions, for distorted, kept; not adopted (chord shapes −0.00062) | D7 | accepted |
| [0035](0035-e3-transition-baseline.md) | ADR 0016's **transition guardrail gets a baseline** under the new E3 rule: 0.9996 oracle, 0.9969 end to end | D10 | accepted |
| [0036](0036-corrections-to-0030-and-0031.md) | Corrections after review: **48 frets/s passes 98.6% of real hand moves** (99.88% of transitions); the stretch relaxes | D9 | accepted |
| [0037](0037-guitarset-validation-player.md) | **GuitarSet's player 00 becomes validation data**; players 01–05 stay test-only (amends 0003) | D8 | accepted |
| [0038](0038-default-redecided-on-player-00.md) | The default decoder **re-decided on player 00**: the clean fit did not clearly win (+0.023, interval [−0.003, +0.051]) — **hand-set again**, T = 1.5728 | D7 | accepted |
| [0039](0039-open-strings-up-the-neck.md) | **Open strings played up the neck cost extra**: one fitted weight, 0.7615; on player 00 oracle E2 0.8065 → 0.8273 (+0.021, interval [+0.007, +0.036]) — **adopted**, T = 1.2934. Chord-shape rule: no clear drop | D7 | accepted |
| [0040](0040-regions-and-strings-on-player-00.md) | ADR 0034's **fret regions and per-string preferences, re-judged on player 00**, each on top of the default: +0.0009 and +0.0010, intervals include zero — **neither adopted** | D7 | accepted |
| [0041](0041-framework-and-compute.md) | The learned model's **framework and compute**: PyTorch 2.14.1 in a `model` group, CPU-only on Linux, the M4 CPU first | D11 | accepted |
| [0042](0042-dadagp-trained-weights.md) | **Weights trained on DadaGP stay unpublished**; code and results public; fitted cost weights stay | D15 | accepted |
| [0043](0043-learned-model-design.md) | **The learned model**: today's decoder plus a learned cost from a bidirectional GRU over the groups; the CRF stays the output layer | D12 | accepted; measured by 0044 |
| [0044](0044-learned-model-measured.md) | The learned model on player 00: DadaGP NLL 0.4327 → 0.3136, but oracle E2 0.8273 → 0.7970 (−0.030, interval [−0.086, +0.020]) — **not preferred**; Phase 2's comparison exists on validation | D12 | accepted |
| [0045](0045-phase-2-test-evaluation.md) | **Phase 2's test evaluation**: today's default 0.6819 oracle E2 on players 01–05 (+0.026, confirming ADR 0039) — **M2 missed by 7.8 points**; the comparison on the test players: (a) 0.6819, (b) 0.4514, (c) 0.6856 | D10 | accepted |
| [0046](0046-audio-evidence.md) | **The audio evidence**: a string probability per note, from a small CNN over a constant-Q window, through the acoustic term; pretrained on SynthTab | D13 | accepted |
| [0047](0047-acoustic-term.md) | **The acoustic term**: `HandSetScorer` carries per-note string log-probabilities; a note on string s adds `acoustic × −log P(s)`; zero weight is Phase 2's decoder | D13 | accepted |
| [0048](0048-phase-3-result.md) | **Phase 3's result**: the SynthTab-trained audio evidence costs the decoder on GuitarSet — test oracle E2 0.6819 → 0.6073 (−0.075), player 00 −0.041 calibrated — and helps on SynthTab's held-out tracks, where it was calibrated; the weight stays zero | D13 | accepted |
| [0049](0049-primary-guitar-sound.md) | **The primary guitar sound is clean electric** (D2): Phase 4 trains on real electric audio, Guitar-TECHS first; GuitarSet (acoustic) stays the test set, reported separately with EGDB | D2 | accepted |
| [0050](0050-egdb-second-test-set.md) | **EGDB is the second test set**: all 240 clips, direct input, in a committed snapshot under the same guard as GuitarSet; no validation part | §3.3 | accepted |
| [0051](0051-guitar-techs-split-and-labels.md) | **Guitar-TECHS**: players 1–2 train, player 3 validates; direct input; pickup glitches under 60 ms dropped, overlaps trimmed, bends and harmonics left out; per-player label delay corrected at the window | D13 | accepted |

Decisions still open, each due at the phase that needs it: D14 (app surface — **now due, M1 is met**), and D15 for corpora other than DadaGP
and SynthTab (code and publication settled by ADR 0020; DadaGP-trained weights by ADR 0042).
D2, D11, D12 and D13 are decided by ADRs 0049, 0041, 0043 and 0046.

**D10 is settled** by ADR 0016: oracle E2 must reach baseline + 10 points (0.760) at M2.

D16 (C++ decoder) is closed as will-not-do by ADR 0015.
