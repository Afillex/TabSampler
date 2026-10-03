# HANDOFF — read this first

You are picking up Tab Sampler with no prior context. This file is the shortest path to
being useful. Written 2026-09-27 after Milestone M1; last updated 2026-10-03, after the
hand-window session.

## In one paragraph

Tab Sampler turns a guitar recording into guitar tablature. The pipeline is
`audio → note events → (string, fret) candidates → fingering scorer → decoder → tab`, and
every stage sits behind a `typing.Protocol` in `src/tabsampler/types.py` so one can be
swapped without touching the others. Phase 0 (an evaluation harness built before any model)
and Phase 1 (a hand-set cost model decoded with Viterbi plus forward-backward posteriors)
are both **done, measured and tagged**, and **Phase 1.5 has closed M1's known defects**
(ADRs 0016-0019). **DadaGP arrived on 2026-10-01** and Phase 2 is under way: the training
protocol is fixed (ADR 0021) and every fit now uses an artist-disjoint split (ADR 0024);
ADR 0011's chord rules are validated on human tab (ADR 0022); the hand is modelled as a
4-fret window (ADR 0025); the default keeps hand-set weights, because fitted ones lost a
fair test fixed in advance (ADR 0027), and has a calibrated temperature (ADR 0026). What
remains is a fingering model good enough for M2, audio conditioning, and an app a
guitarist can use.

## Rules that are not negotiable

1. **GuitarSet and EGDB are test-only** (ADR 0003). No tuning, selection, or threshold
   sweeping against them. `data/splits.py` enforces it; every look is logged to
   `experiments/test_set_access.log` with a written reason.
2. **Never invent a number.** Every figure in the README and `experiments/results.csv` came
   from a run that actually executed. A metric not computed is blank, never 0.0.
3. **No cross-paper comparison** (spec §3.4). `docs/spec.md` quotes TART's 69.2% / 56.0% on
   GuitarSet. Those are under *TART's* protocol. Do not put them beside ours.
4. **Report degradation next to the number.** If a run dropped notes or relaxed a
   constraint, say so where the metric is quoted.

## Start here

```bash
make install                       # uv sync --all-groups, Python 3.13
make check                         # lint + pyright --strict + 403 tests. Must be green.
make oracle                        # the correctness core. Must be green.
```

`make check` piped into `tail` hides its exit code — check the status, not the output.

Then read, in this order: `docs/spec.md` → `docs/plans/2026-10-01-phase-2.md` (the live
plan; C1, C2 and C5 are done, C3 is next) → `docs/devlog/2026-10-02.md` (the most recent
session) → `docs/plans/2026-09-27-rest-of-project.md` → `docs/adr/README.md`.

## Where things stand

| | |
|---|---|
| Repo | `https://github.com/Afillex/TabSampler` — **public**, MIT (ADR 0020) |
| Tags | `v0.0-phase0`, `v0.1-m1` |
| Tests | 403, all offline — no test needs the dataset or the transcriber |
| CI | GitHub Actions, green, ~30 s |
| Current phase | Phase 2, **in progress**: `docs/plans/2026-10-01-phase-2.md`, task C3 next |

**Current results** — 360 GuitarSet tracks, `audio_mic`, the default decoder: hand-set
weights, the hand window, T = 2.9974. Rows before 2026-10-01 have a `commit` column that
references the pre-publication history, which was squashed into the initial commit; the
rows are still the record of which runs produced which numbers. Reproduce the current ones
with `uv run tabsampler eval-m1 --config configs/m2_window_eval.yaml`. The previous default
(hand as a point, T = 1) is in brackets:

| | oracle | end-to-end |
|---|---|---|
| E1 note F1, transcriber (raw) | 1.0000 | 0.7437 |
| E1 note F1, pipeline (placed) | 1.0000 | 0.7452 |
| **E2 exact tab F1 (headline)** | **0.6878** (0.6599) | **0.4396** (0.4318) |
| E3 playable groups | 0.9967 (0.9970) | 0.9904 (0.9896) |
| E3 playable transitions, window rule | 0.9944 | 0.9834 |
| E4 pitch validity | 1.0000 | 1.0000 |
| E5 calibration error | 0.1651 (0.1652) | **0.1272** (0.3851) |

E1 = 1.0 in oracle mode is a plumbing check, not a result. **E4 = 1.0 is an invariant, not a
score: anything below 1.0 is a bug**, because it means a (string, fret) pair does not sound
the pitch we claimed. **The oracle chord-shape rate is 0.0003 below ADR 0016's guardrail**,
and M2's target of 0.760 is 7.2 points away. E3's transition rate is under a new rule and
is not comparable with older rows.

**E1 is two numbers now.** Raw = the transcriber's score, and it reproduces Phase 0's 0.7437
exactly. Placed = the pipeline's, after placement drops notes the guitar cannot sound. M1
reported only the second and compared it with Phase 0's first.

## Training data: DadaGP, and how it may be used

**DadaGP v1.1 arrived on 2026-10-01; ProgGP did not.** The archive lives at
`data/dadagp/DadaGP-v1.1.zip` (gitignored — **never commit it, the repo is public**), and
`data/dadagp/track_meta.json` records which songs are clean standard-tuned guitar. Rebuild the
latter with `scripts/dadagp_track_meta.py` (its docstring has the command).

ADR 0021 is the protocol, and it is not optional: fit on DadaGP **training**, select and
calibrate on DadaGP **validation**, evaluate on GuitarSet **once**, and never choose anything
by looking at GuitarSet. **Use the artist-disjoint split** (`--split artist`, ADR 0024) for
every fit: whole artists sit on one side, frozen by hash. The shipped split, whose artists
overlap, stays loadable only so the 2026-10-01 numbers can be reproduced. Validation parts
are about three-quarters distorted guitar; report clean and distorted parts separately,
because they keep disagreeing.

## Five things that will trip you up

1. **`mir_eval.transcription` takes pitches in Hz, not MIDI.** It derives cents from a
   frequency *ratio* (`transcription.py:435`), so MIDI 40 against MIDI 41 computes as 42.7
   cents — inside the 50-cent tolerance — and a semitone error scores as a **hit**.
   `eval/metrics.py` converts at the boundary and a test pins it. Never bypass it.
2. **There are two Python environments, on purpose (ADR 0001).** The core is 3.13.
   `basic-pitch` cannot be installed alongside it on macOS arm64 — it requires
   `tensorflow-macos` for `python_version > "3.11"` and that wheel stops at cp311 — so it
   lives on 3.11 behind a subprocess and an on-disk cache. Install it with
   `bash scripts/setup_transcriber.sh`, which also pins `setuptools<81` because `resampy`
   imports `pkg_resources`. `uv tool install` puts the binary in `~/.local/bin`, which is
   **not on PATH** by default.
3. **basic-pitch's note-event CSV is ragged.** The header names five columns, but the writer
   does `row.extend(pitch_bend)`, so a row has four fields plus one column per bend value.
   A `csv.DictReader` silently keeps the first and drops the rest. Rows are also **not**
   sorted by onset.
4. **`NoteEvent.bend`'s units are unverified.** basic-pitch's `pitch_bend` is in internal
   contour bins and the bins-per-semitone factor has never been checked. It is carried
   through unused. **Verify it before Phase 7 builds on it.**
5. **`pyright --strict` against the scientific stack is the main source of churn.**
   `mir_eval`, `mirdata`, `scipy.sparse.csgraph` and `scipy.special.logsumexp` ship no
   annotations. The pattern that works: confine each untyped call to one small function with
   an explicit return type and a narrow `# pyright: ignore[...]`, so suppressions sit at the
   boundary and never in the logic. Examples in `eval/metrics.py` and
   `decode/forward_backward.py`.

Also: **ruff 0.16 formats Python code blocks inside Markdown** and will rewrite
`docs/spec.md` if you let it. `docs/**` is excluded for that reason — leave it excluded.

## Known defects

**Closed in Phase 1.5:** hand movement across an all-open chord (ADR 0018) and barre chords
scored as unplayable (ADR 0019). Both moved reported numbers; see
`docs/devlog/2026-09-27-phase-1.5.md`.

**Closed in Phase 2 so far:** E3 not carrying the hand across an all-open shape, and E3
and the cost model counting finger reach as hand movement — both by the hand window (ADR
0025). The fitted-versus-hand-set question of ADR 0023 is settled by ADR 0027's fair test.

Still open:

- **E3's transition rule is still not a playability measure.** With the window, human tab
  passes it 97.95% of the time, short of the 99% bar fixed in advance; what is left is
  largely small, quick shifts that the 12 frets/s speed limit calls impossible. Needs a
  pre-registered experiment on the limit. Until then, quote the rate with ADR 0022's caveat.
- **A still-ringing note does not reserve its string** against the next group (spec §2.2).
  Not yet owned; revisit if metrics show it matters.
- **`eval-m1` labels every row "hand-set weights (ADR 0012)"** and prints "ground-zero
  number" whatever `--decoder-config` says, so the 2026-10-01 fitted-weights rows carry a
  wrong note (their config column is right). Small fix, needs a test.
- **`eval-m1` writes the eval config's hypothesis into `results.csv`.** Give every new
  evaluation its own config with its own hypothesis, as `configs/m2_window_eval.yaml` does;
  the Phase 1.5 rows still carry M1's hypothesis text.

## Things that are settled — don't relitigate them

Read the ADR before proposing a change to any of these. `docs/adr/README.md` is the index.

- GuitarSet is test-only, and `audio_hex` is **forbidden as transcriber input** because one
  channel per string is the E2 ground truth (ADR 0003, 0005).
- Reference MIDI floats are **rounded, not truncated**, for the reference tab; floats are kept
  for E1 (ADR 0006).
- Contract collections are **tuples, not lists** — a list in a frozen dataclass leaves it
  mutable and unhashable, and the metrics need `Position` as a dict key (ADR 0007).
- v1 tab is **time-positioned with no rhythmic notation** (ADR 0009). ADR 0017 settles what
  the exporters do about it — a declared 120 BPM grid plus "rhythm is NOT transcribed"
  inside the file — so **Task B1 is unblocked**.
- Decoder states are **chords, not notes**, so one-note-per-string is structural (ADR 0010).
- **M1 shipped hand-set weights deliberately** (ADR 0012) — Phase 1 had no legal validation
  data. That is a real result, not a placeholder.
- **No C or C++** (ADR 0015). Decode is 3% of runtime; the question is measured and closed.

## Useful facts you would otherwise have to rediscover

- **Runtime:** a 188.7 s file, cold cache, takes 3.70 s total — transcribe 3.59, decode 0.11.
  Spec §4's 60 s target passes with 16× headroom. E7 is 1.18 s per audio minute on one long
  file but 2.73 across the 360 excerpts, because basic-pitch's model load is paid once versus
  360 times. Quote the right one for the question being asked.
- **State space:** mean 4.92 states per group at span 5, 17.67 unpruned. Viterbi is `O(T·S²)`,
  so pruning saves roughly 13× on average. `fingering/states.py::state_count_stats` exists to
  re-measure this; re-run it if Phase 6 makes files much longer.
- **`decode()` is strict and raises on an unfingerable group; `decode_best_effort()` is what
  real audio needs.** It distinguishes three causes — notes outside the instrument (basic-pitch
  predicted MIDI 39, below the open low E, 158 times across the corpus), shapes needing a wider
  stretch, and groups unfingerable at any stretch — and reports every one. Use the strict
  version in tests, the best-effort one on real input.
- **Decoder states are lattice *nodes*, not bare chord shapes** (ADR 0018): a shape plus the
  hand window carried into it. Since ADR 0025 every shape gets one node per reachable
  window: 2.57× the state count on GuitarSet's reference notes (1.07× under the old point
  model), and decode is about twice as slow (`decode`, 500 groups: 58.3 → 113.7 ms).
  Re-measure with `scripts/measure_lattice.py`; benchmark with `scripts/bench_decode.py`.
  Anything reading per-shape quantities out of `forward_backward` must sum over nodes;
  `note_posteriors` does.
- **`fingering.states.shift_window` is the one definition of the hand.** E3, the cost
  model, the decoder's lattice and the CRF fitter all call it; the brute-force oracle
  re-derives it from ADR 0025's text on purpose and must not import it.
- **The window makes moves asymmetric**: from fret 3 (frets 3–7), reaching fret 9 moves the
  window 2; from fret 9, reaching fret 3 moves it 6, because a hand is placed at its lowest
  note. Tests pin both directions.
- **DadaGP fits and calibrations take minutes, not seconds.** A 600-song window fit runs in
  roughly ten minutes; `scripts/calibrate_temperature.py` builds each lattice once and only
  reruns forward-backward per trial temperature. Run long jobs under `caffeinate -i` so the
  machine does not sleep mid-run.
- **E7 in `results.csv` is too noisy to compare across runs.** The `eval-m1` column measured
  0.0370, 0.17, 0.20 and 0.04 s per audio minute for equivalent work on this machine in one
  afternoon. Benchmark decode directly instead.
- **`tests/decode/brute_force.py` is the specification.** It enumerates every path and scores
  it from first principles. It is the independent check that the decoder is correct
  (ADR 0014), so a change to it deserves more scrutiny than a change to the decoder. If they
  disagree, the decoder is wrong.
- **A 4-track smoke test gave oracle E2 = 0.90 where all 360 gave 0.64.** Never quote a number
  from a subset.

## Open items that need Ege, not you

- **The speed limit**: approve a pre-registered experiment on E3's 12 frets/s limit, the
  remaining reason the transition rate is not a playability measure.
- **One temperature or one per style**: oracle E5 did not move on clean GuitarSet with a
  temperature fitted on mostly distorted validation parts. A validation question, best
  taken with task C3's style work.
- **The `eval-m1` row label**: fix it, and decide whether the 2026-10-01 rows get a
  correcting note.
- **D15 — weight release, the half ADR 0020 left open.** The code is MIT and the repo is
  public. Weights are a **per-corpus** decision, because DadaGP and ProgGP are
  research-use-only: check the terms *before* training anything whose weights might be
  published, not after.

## When you finish a session

Write `docs/devlog/YYYY-MM-DD.md`: what was done, what was measured, what's next, open
questions. Update this file if the
state a fresh agent needs has changed.
