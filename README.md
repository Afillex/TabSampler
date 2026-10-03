# Tab Sampler

![CI](https://github.com/Afillex/TabSampler/actions/workflows/ci.yml/badge.svg)

Turn a guitar recording into guitar tablature.

```text
audio -> note events -> string/fret candidates -> fingering scorer -> decoder -> tab
```

Status: **Milestone M1 reached; Phase 2 (learned fingering) in progress.**

## Results

All 360 GuitarSet tracks, `audio_mic`. The default decoder is now the one for **clean or
acoustic guitar** (ADR 0032): cost weights **fitted on clean DadaGP playing**, a
**temperature calibrated for that style** (T = 1.1975, ADR 0033), and the hand modelled as a
4-fret window that a wide chord can stretch (ADRs 0025, 0030). Distorted guitar has its own
decoder, `configs/decoder_distorted.yaml`. Both modes per spec 3.2: *oracle* feeds
reference notes to the fingering stage and so measures fingering alone; *end-to-end* feeds
Basic Pitch's notes and is what a user gets.

| | metric | oracle | end-to-end | previous default (oracle / e2e) |
|---|---|---|---|---|
| E1 | note F1, transcriber (raw) | 1.0000 | 0.7437 | 1.0000 / 0.7437 |
| E1 | note F1, pipeline (placed) | 1.0000 | 0.7452 | 1.0000 / 0.7452 |
| **E2** | **exact tab F1 (headline)** | **0.6689** | **0.4439** | 0.6878 / 0.4396 |
| E3 | playable groups | 0.9957 | 0.9889 | 0.9967 / 0.9904 |
| E3 | playable transitions | 0.9996 | 0.9969 | a different rule |
| E4 | pitch validity | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| E5 | calibration error (ECE) | **0.1008** | 0.3027 | 0.1651 / 0.1272 |

*Previous default*: hand-set weights, T = 2.9974, the same window without the stretch. 62 476
reference notes; 62 476 placed in oracle mode, 62 758 end to end. End to end, 158 notes lay
outside the guitar's range, 127 more were dropped as unfingerable, and 64 groups (1 in
oracle mode) were decoded with the span bound relaxed. **The transcriber costs 0.2251 of
E2** (0.6689 → 0.4439).

### What changed, and what GuitarSet said

Decided on DadaGP before GuitarSet was run (artist-disjoint validation, ADR 0024), every
comparison with a song-level paired bootstrap:

- **Clean and distorted guitar got a decoder each** (ADR 0032). Weights fitted on clean
  playing alone beat the hand-set ones on clean validation parts, 0.8257 → 0.8616 (95%
  interval +0.013 to +0.062): the first fitted weights to win on clean validation parts. Weights fitted
  on distorted playing gained even more there (+0.077) but decoded slightly more chord
  shapes that E3 calls unplayable (0.00065 more, against an allowance of 0.0005), so the
  distorted decoder keeps the hand-set weights.
- **Each decoder has its own temperature** (ADR 0033): 1.1975 for clean, 4.2982 for
  distorted, from one pooled 2.9974.
- **E3's transition rule became a measure** (ADRs 0029, 0031, 0036): a move is timed from
  the last fretted note, and the speed limit is 48 frets per second, set from human tab.
  On artists it was not taken from, human tab passes it on 99.88% of transitions — and on
  98.6% of the moves where the hand actually shifts, since most transitions move nothing.
  At the old 12 frets per second only 77% of those moves passed.
- **Two richer feature groups were tried** (ADR 0034): a per-string preference and fret
  regions. Neither earned a place in a shipped decoder.

| ADR 0016 guardrail | required | measured | |
|---|---|---|---|
| E4 | 1.0000 | 1.0000 / 1.0000 | held |
| E3 groups, oracle | ≥ 0.9970 | 0.9957 | **breached, by 0.0013** |
| E3 groups, end to end | ≥ 0.9896 | 0.9889 | **breached, by 0.0007** |
| E3 transitions | baseline set by this run (ADR 0035) | 0.9996 / 0.9969 | — |
| E5 end to end | < 0.3851 | 0.3027 | held |
| **M2 target**, E2 oracle | ≥ 0.760 | 0.6689 | **missed, by 9.1 points** |

**The headline went the wrong way.** The prediction committed before the run
(`configs/m2_style_eval.yaml`) expected oracle E2 to rise, because the clean decoder had
won by 3.6 points on clean DadaGP validation; **on GuitarSet it fell by 1.9 points**. Its
chord-shape prediction missed too: "within 0.001 of 0.9967 / 0.9904" came out 0.9957 (on
the boundary) and 0.9889 (0.0015 off), breaching both guardrails. End-to-end E2 rose by
0.4 points, as predicted, though that is within what ADR 0016 counts as noise; oracle
calibration error fell from 0.1651 to 0.1008 and end-to-end calibration error rose from
0.1272 to 0.3027, both as predicted — a sharper decoder is more confident about notes the
transcriber got wrong.

Two of these decisions were prompted in part by earlier GuitarSet runs — the per-style
temperature by oracle calibration error not moving last time, the clean default by
GuitarSet being acoustic. Both were decided on DadaGP, so ADR 0003 holds, but this run
confirms them less independently than a fresh prediction would. ADR 0016's guardrails now
also rest on two different decoders: the chord-shape baseline is the hand-set decoder's
from Phase 1.5, the transition baseline this run's (ADR 0035).

This is the third time DadaGP's clean parts have been a poor guide to GuitarSet, and the
first time they got the direction wrong: validation said better, the test set says worse. **The default is
not reverted on this evidence**: choosing by GuitarSet, in either direction, would be
selecting on the test set (ADR 0003), exactly as adopting fitted weights on a GuitarSet
gain would have been in ADR 0023. What it does show is that the project needs validation
data that predicts GuitarSet before the next model choice can mean much; that is the open
question this leaves.

### The hand window (ADRs 0024–0028)

Before this, the hand became a 4-fret window rather than a point (ADR 0025): human tab's
transition pass rate under the old 12 frets/s limit went from 88.35% to 97.95% (98.23% once
moves were timed correctly, ADR 0029), oracle E2 rose 2.8 points on GuitarSet (0.6599 →
0.6878), and both of the window's pre-registered checks were missed and recorded (ADRs 0025,
0028). Fitted weights lost a fair test fixed in advance (ADR 0027), and a pooled temperature
was calibrated (ADR 0026). `docs/devlog/2026-10-02.md` has the detail.

### Phase 1.5

Phase 1.5 closed two of M1's defects, each measured on its own (one variable per run):

- **ADR 0018** — an all-open chord used to be free to move to and from, so
  `fret 2 → open → fret 10` was charged no hand movement. Now it is.
- **ADR 0019** — E3 counted fretted *notes*, so every full barre chord was scored
  unplayable. It counts fingers now, with the barre free at the lowest fret only.

| metric | M1 | after ADR 0018 | after ADR 0019 | |
|---|---|---|---|---|
| E2 oracle | 0.6366 | **0.6599** | 0.6599 | +0.0233 |
| E2 end-to-end | 0.4231 | **0.4318** | 0.4318 | +0.0087 |
| E3 groups oracle | 0.9835 | 0.9832 | **0.9970** | +0.0135 |
| E3 groups end-to-end | 0.9751 | 0.9744 | **0.9896** | +0.0145 |
| E3 transitions oracle / e2e | 0.9170 / 0.9228 | 0.9134 / 0.9164 | unchanged | −0.0036 / −0.0064 |
| E5 ECE oracle / e2e | 0.1632 / 0.3650 | 0.1652 / 0.3851 | unchanged | +0.0020 / +0.0201, **worse** |

E1 (1.0000 / 0.7452 placed) and E4 (1.0000 / 1.0000) did not move in either run. ADR 0019
touched nothing but E3's group rate, as a change to one metric's rule should not.

Charging hand movement also made the **cost weights** visibly wrong: `move` is 1.0 per
fret against `high` at 0.1 per *octave*, a ratio of 120:1, so once movement is charged the
decoder pre-positions the hand high up the neck instead of playing in open position. ADR
0018 has the worked example — the new fingering costs 3.667 against 7.383 for the
open-position one a guitarist would use, so it is a genuine minimiser of a cost model whose
weights are wrong. **Deliberately not fixed by reweighting**: that is tuning with no
validation data (ADR 0012), and on one hand-picked clip it would be selection on an
example. E2 rose, so the corpus does not agree the new output is worse.

### Weights fitted on DadaGP — measured, not the default (ADR 0023)

With DadaGP v1.1 as training data (ADR 0021), the four weights were fitted by maximum
likelihood on 469,363 human chord shapes and the temperature was calibrated on validation.
Every step was pre-registered in a committed script before it ran. On GuitarSet, evaluated
once:

| metric | hand-set (then the default) | fitted + T = 1.721 | |
|---|---|---|---|
| E2 oracle | 0.6599 | **0.6988** | +0.0389 |
| E2 end-to-end | 0.4318 | **0.4603** | +0.0285 |
| E3 chord shapes, oracle / e2e | 0.9970 / 0.9896 | 0.9961 / 0.9883 | below baseline |
| E5 end-to-end | 0.3851 | **0.1411** | −63% |
| E5 oracle | 0.1652 | **0.1286** | −22% |

It is **not the default**, for a reason decided on DadaGP validation before GuitarSet was
run: the fitted weights recover 11 points *less* of the human fingering on clean guitar
parts (0.8097 → 0.6963) and gain only on distorted ones. It also breaches ADR 0016's E3
guardrail by a tenth of a point and is 6.1 points short of the M2 target of 0.760.
GuitarSet's verdict cannot be the reason to adopt it — that would be selecting on the test
set. ADR 0023 lays out the two legitimate routes. `configs/fitted_dadagp.yaml` holds these
weights; the figures above reproduce only at commit e809798, before the hand window.

**Settled since** (ADR 0027): refitted under the hand window on the artist-disjoint split,
they lost the fair test fixed in advance on DadaGP validation — clean recovery 0.6919
against the hand-set weights' 0.8257, decoded chord shapes 0.9987 against 0.9998 — and stay
off. The figures above are the
point-model ones; GuitarSet was not run again for them.

**E1 is two numbers, not one.** *Transcriber (raw)* scores the notes handed to the
fingering stage; *pipeline (placed)* scores the notes that came out. They differ because
placement drops notes the guitar cannot sound, which raises precision without touching
recall. M1 reported only the second (0.7452) and set it beside Phase 0's first (0.7437),
which was comparing two different measurements; raw E1 now reproduces Phase 0's figure
exactly. In oracle mode both are 1.0000, which is a plumbing check, not a result:
reference notes in, the same notes out.

E4 = 1.0000 is the correctness invariant — every (string, fret) sounds the pitch claimed.
Anything below 1.0 there is a bug, not a score.

**These numbers are not comparable to any published GuitarSet figure.** Other work uses
different protocols, channels and splits; spec 3.4 forbids the comparison unless the
method is rerun here, and nothing has been. The M2 target, oracle E2 ≥ 0.760, was set from
this project's own baseline (ADR 0016), not from anyone else's figure.

### What the numbers say, honestly

- Given perfect notes, the default places about **67%** of notes on the string a human
  actually used — 69% for the previous default, 66% before the hand window. ADR 0016's M2
  target is 76%.
- End to end, E2 is bounded by note F1: you cannot finger a note you did not hear. The
  transcriber's own F1 is 0.7437, which is the ceiling to argue about, not 0.7452.
- **Calibration.** Given perfect notes the default's posteriors are now fairly honest
  (E5 0.10); end to end they are less so (0.30), because the decoder is surer of itself and
  the transcriber's mistakes are not its to see. ADR 0016's E5 guardrail (< 0.3851) holds.
- **Both halves of E3 are now measured against human tab.** 99.86% of 16.8 million human
  chord shapes pass the chord rules (ADR 0022), and human tab passes the transition rule on
  99.88% of transitions — 98.6% of the moves where the hand actually shifts — on artists the
  speed limit was not taken from (ADRs 0031, 0036). Our output passes 0.9957 / 0.9889 of its
  chord shapes and 0.9996 / 0.9969 of its transitions.
- **DadaGP validation does not yet predict GuitarSet.** Three times its clean parts were a
  poor guide: the pooled fitted weights (−11 points on clean validation, +3.9 on GuitarSet,
  ADR 0023), the hand window (+0.4 and +2.8: right direction, far off in size) and the clean
  decoder (+3.6 and −1.9). Ege has since decided to hold out one GuitarSet player as
  validation data; the next plan does that, and re-decides the default on it.

### Runtime

Measured, not assumed (spec 4). One 188.7 s (3.14 min) file built from real GuitarSet
playing, **cold** transcriber cache:

| stage | time |
|---|---|
| transcribe (basic-pitch) | 3.59 s |
| group + decode | 0.11 s |
| **total** | **3.70 s** |

Spec 4's target is a 3-minute file in under 60 s on an M4 MacBook Air: **passes with 16x
headroom**. Decode is 3% of the total, so optimising the decoder would be pointless. This
was measured before the hand window, which roughly doubles decode time on the synthetic
benchmark below; the file has not been re-timed since.

E7 depends heavily on what you measure. Per audio minute it is **1.18 s** on one long
file and **2.73 s** across the 360 GuitarSet excerpts, because basic-pitch's model load
is paid once in the first case and 360 times in the second. The corpus figure is the
honest one for evaluation cost; the single-file figure is the honest one for what a user
waits for.

**The E7 column of the `eval-m1` rows in `experiments/results.csv` is too noisy to compare
across runs.** It is decode time with a warm transcriber cache, and on this machine it
measured 0.0370, 0.17, 0.20 and 0.04 s per audio minute for equivalent decode work — a
5× spread with no code change to explain it. Do not read a decode-time change out of it.

For that, benchmark decode directly with `uv run python scripts/bench_decode.py`, run once
per commit against its own source. On a 500-group synthetic line, median of seven warmed
runs, ADR 0018's node lattice took `viterbi` from 5.5 ms to 7.2 ms (+31%) and full `decode`
from 53.8 ms to 57.9 ms (+8%). Against spec 4's 60 s budget, nothing. The hand window
(ADR 0025) then took them to 32.6 ms and 113.7 ms, against 7.2 ms and 58.3 ms measured the
same day, and the stretch (ADR 0030) to 34.3 ms and 120.1 ms.

### Known v1 limitations

- No rhythmic notation: spacing is proportional to time, with no bars or note values
  (ADR 0009).
- A still-ringing note does not reserve its string against the next group (spec 2.2).
- The user picks the decoder: `configs/decoder_clean.yaml`, the default, or
  `configs/decoder_distorted.yaml`. Nothing detects the style from the audio (ADR 0032).
- The distorted decoder still uses the hand-set weights: every distorted fit so far decodes
  slightly more chord shapes that E3 rejects than ADR 0032 allows (ADRs 0032, 0034). The
  clean decoder's fitted weights won on DadaGP validation and lost on GuitarSet (above).
- E3's speed limit (48 frets/s) comes from crowd-sourced tab quantised to a rhythmic grid:
  some very fast moves in it may be notation rather than playing (ADR 0031). It still calls
  about one human hand move in seventy too fast (ADR 0036).
- After a wide chord, a lone low note relaxes the hand's stretch, so alternating the chord's
  low and high notes is charged a fret each time (ADR 0036).
- E3's finger model is a rule, not a measurement: one finger barres the lowest fretted
  fret and every note above it costs a finger, which refuses some partial barres a good
  player manages (ADR 0019).
- Basic Pitch predicts notes the guitar cannot sound (158 of them here, e.g. MIDI 39
  below the open low E). They are dropped and counted, which lowers recall.

## Documentation

| Document | What it covers |
|---|---|
| `docs/spec.md` | Architecture, data contracts, metrics, phases, decisions D1-D16 |
| `docs/plans/` | Executable implementation plans |
| `docs/adr/` | One Architecture Decision Record per decision |
| `docs/devlog/` | One entry per work session |

## Setup

The project deliberately uses **two Python environments**. See ADR 0001.

```bash
# 1. The core package: Python 3.13
uv sync --all-groups

# 2. The transcriber: Python 3.11, isolated (basic-pitch cannot coexist with the core)
bash scripts/setup_transcriber.sh
```

`basic-pitch` 0.4.0 declares `tensorflow-macos<2.15.1` under the marker
`platform_system == "Darwin" and python_version > "3.11"`, and that wheel has no
build past `cp311`. On macOS arm64 it therefore installs only on Python <= 3.11,
where it needs no TensorFlow at all and uses CoreML instead. Rather than drag the
whole project back to 3.11 and numpy 1.x, the transcriber is quarantined behind a
subprocess boundary and an on-disk note-event cache.

Two operational details the setup script handles, both found by running it:

- It pins **`setuptools<81`** in the tool environment. basic-pitch pins
  `resampy<0.4.3`, resampy 0.4.2 imports `pkg_resources`, and setuptools removed
  `pkg_resources` in v81. Without the pin the CLI dies at import.
- `uv tool install` puts the executable in `~/.local/bin`, which is **not on PATH**
  by default. Set `transcriber.exe` in the config to an absolute path if needed.

## Commands

```bash
make install     # uv sync --all-groups
make test        # pytest
make oracle      # brute-force equivalence tests for the decoder
make lint        # ruff check + format --check
make typecheck   # pyright --strict on src/
make check       # lint + typecheck + test
make eval-notes  # Phase 0 gate: Basic Pitch note F1 on GuitarSet
```

## License

MIT. See [LICENSE](LICENSE).

Note that the datasets are not covered by it: GuitarSet has its own terms, and the
symbolic corpora used for training are research-use-by-request. Model weights, if any are
ever published, are licensed per training corpus rather than once.

## Evaluation protocol

**GuitarSet is test-only** (ADR 0003). It is never used for tuning, model selection
or threshold sweeps. Cost weights and hyperparameters are tuned on validation data
drawn from training sources only. Every look at the test set is appended to
`experiments/test_set_access.log`.

Results are always reported in both modes (spec 3.2): **oracle** (reference notes fed
to the fingering stage, which measures fingering quality alone) and **end-to-end**
(transcriber notes, which is what a user actually gets). The gap between them is what
the transcriber costs.

Numbers from other papers are not comparable to ours unless rerun under this
protocol (spec 3.4).
