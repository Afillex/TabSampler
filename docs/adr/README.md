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
| [0011](0011-playability-rules.md) | Concrete E3 playability rules (validation pending DadaGP) | D9 | accepted; finger rule superseded by 0019 |
| [0012](0012-cost-weights-hand-set-for-m1.md) | **M1 ships hand-set weights**; tuning gated on legal data | D7 | accepted |
| [0013](0013-output-formats.md) | ASCII + JSON in Phase 1; MusicXML/GP on the app track | D4 | accepted |
| [0014](0014-brute-force-oracle.md) | **The brute-force oracle is the decoder's specification** | 2.2 (gap) | accepted |
| [0015](0015-no-cpp-decoder.md) | No C/C++ decoder port — decode is 3% of runtime | D16 | accepted |
| [0016](0016-headline-target.md) | D10 target: **oracle E2 >= baseline + 10 points** at M2, with E3/E4/E5 guardrails | D10 | accepted |
| [0017](0017-rhythm-for-exports.md) | Exports use a fixed 120 BPM grid and carry "rhythm is not transcribed" **in the file** | D5 (gap) | accepted |
| [0018](0018-hand-position-carry.md) | Carry the hand position across all-open shapes; `transition_cost_from` added to the scorer contract | 2.1/2.2 (defect) | accepted |
| [0019](0019-barre-chords-in-e3.md) | E3 counts **fingers**, not fretted notes, so barre chords are playable (supersedes ADR 0011's fourth rule) | D9 (defect) | accepted |
| [0020](0020-licensing-and-publication.md) | **MIT for the code, public repo**; weights decided per training corpus | D15 (partial) | accepted |

Decisions still open, each due at the phase that needs it: D2 (primary guitar sound,
Phase 4), D11 (framework and compute, Phase 2), D12 (fingering model design, Phase 2),
D13 (form of audio evidence, Phase 3), D14 (app surface — **now due, M1 is met**),
D15 (licensing — **code and publication settled by ADR 0020**; weight release still open, per corpus).

**D10 is settled** by ADR 0016: oracle E2 must reach baseline + 10 points (0.760) at M2.

D16 (C++ decoder) is closed as will-not-do by ADR 0015.
