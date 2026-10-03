# Architecture Decision Records

One record per decision that is hard to reverse or would need explaining later.
Template: `0000-template.md`.

**Never edit an accepted ADR.** Supersede it with a new one and update the status line
of the old record to point at its replacement.

| ADR | Decision | Spec ref | Status |
|---|---|---|---|
| [0001](0001-python-version-and-isolated-transcriber.md) | Python 3.13 core; basic-pitch isolated on 3.11 behind a subprocess and cache | — (forced) | accepted |
| [0002](0002-v1-input-scope.md) | v1 accepts isolated guitar audio only | D1 | accepted |
| [0003](0003-evaluation-protocol-guitarset-held-out.md) | **GuitarSet is test-only**, enforced in `data/splits.py` | D8 | accepted |
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
| [0030](0030-hand-stretch.md) | A chord wider than the window **stretches the hand** instead of anchoring it (supersedes one sentence of 0025); recovery unchanged | 2.2 / D9 | accepted |
| [0031](0031-e3-speed-limit.md) | E3's speed limit set from human tab: **48 frets/s**, 0.9988 of unseen artists' moves pass | D9 | accepted |
| [0032](0032-style-decoders.md) | Clean and distorted guitar get **a decoder each**; clean-fitted weights win on clean parts (0.8257 → 0.8616) and become the default; distorted keeps hand-set (chord shapes −0.00065) | D7 | accepted |
| [0033](0033-per-style-temperature.md) | Each style's decoder gets **its own temperature**, calibrated on that style's validation parts | D7 | proposed |
| [0034](0034-c3-feature-groups.md) | Two richer feature groups — **per-string preference** and **fret regions** — each kept only if it raises a style's validation recovery | D7 | proposed |

Decisions still open, each due at the phase that needs it: D2 (primary guitar sound,
Phase 4), D11 (framework and compute, Phase 2), D12 (fingering model design, Phase 2),
D13 (form of audio evidence, Phase 3), D14 (app surface — **now due, M1 is met**),
D15 (licensing — **code and publication settled by ADR 0020**; weight release still open, per corpus).

**D10 is settled** by ADR 0016: oracle E2 must reach baseline + 10 points (0.760) at M2.

D16 (C++ decoder) is closed as will-not-do by ADR 0015.
