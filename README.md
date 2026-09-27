# Tab Sampler

![CI](https://github.com/Afillex/TabSampler/actions/workflows/ci.yml/badge.svg)

Turn a guitar recording into guitar tablature.

```text
audio -> note events -> string/fret candidates -> fingering scorer -> decoder -> tab
```

Status: **Milestone M1 reached.**

## Results

All 360 GuitarSet tracks, `audio_mic`, **hand-set cost weights (not tuned)**. Both
modes per spec 3.2: *oracle* feeds reference notes to the fingering stage and so
measures fingering alone; *end-to-end* feeds Basic Pitch's notes and is what a user gets.

| | metric | oracle | end-to-end |
|---|---|---|---|
| E1 | note F1, transcriber (raw) | 1.0000 | 0.7437 |
| E1 | note F1, pipeline (placed) | 1.0000 | 0.7452 |
| **E2** | **exact tab F1 (headline)** | **0.6599** | **0.4318** |
| E3 | playable groups | 0.9970 | 0.9896 |
| E3 | playable transitions | 0.9134 | 0.9164 |
| E4 | pitch validity | 1.0000 | 1.0000 |
| E5 | calibration error (ECE) | 0.1652 | 0.3851 |

62 476 reference notes; 62 476 placed in oracle mode, 62 758 end to end.
**The transcriber costs 0.2281 of E2** (0.6599 → 0.4318).

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
method is rerun here, and nothing has been. No target has been set yet either — ADR 0004
says the target is fixed only once a baseline exists, which is now.

### What the numbers say, honestly

- Given perfect notes, the hand-set model places about **66%** of notes on the string a
  human actually used. Untuned, that is the ground-zero number ADR 0012 promised.
- End to end, E2 is bounded by note F1: you cannot finger a note you did not hear. The
  transcriber's own F1 is 0.7437, which is the ceiling to argue about, not 0.7452.
- **E5 is the weak result.** An ECE of 0.37 end to end means the posteriors are not
  honest yet, and goal 4 of spec 1 — being truthful about uncertainty — is not met.
  Temperature is untuned at 1.0; calibrating it is the obvious next experiment.
- E3 is reported against ADR 0011's span thresholds, which are still **unvalidated**: the
  very first GuitarSet track contains a human-played chord those rules call unplayable.
  ADR 0019 fixed the finger rule, which needed no data; the thresholds still need DadaGP.
  Read E3 as "passes our current rules".

### Runtime

Measured, not assumed (spec 4). One 188.7 s (3.14 min) file built from real GuitarSet
playing, **cold** transcriber cache:

| stage | time |
|---|---|
| transcribe (basic-pitch) | 3.59 s |
| group + decode | 0.11 s |
| **total** | **3.70 s** |

Spec 4's target is a 3-minute file in under 60 s on an M4 MacBook Air: **passes with 16x
headroom**. Decode is 3% of the total, so optimising the decoder would be pointless.

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
from 53.8 ms to 57.9 ms (+8%). Against spec 4's 60 s budget, nothing.

### Known v1 limitations

- No rhythmic notation: spacing is proportional to time, with no bars or note values
  (ADR 0009).
- A still-ringing note does not reserve its string against the next group (spec 2.2).
- The hand-set cost weights are untuned and their `move : high` ratio of 120:1 is almost
  certainly wrong; it makes the decoder prefer parking the hand high to playing in open
  position (ADR 0012, ADR 0018). Tuning is gated on symbolic tab data.
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
