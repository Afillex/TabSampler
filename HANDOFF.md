# HANDOFF — read this first

You are picking up Tab Sampler with no prior context. This file is the shortest path to
being useful. Written 2026-09-27 after Milestone M1, updated the same day after Phase 1.5.

## In one paragraph

Tab Sampler turns a guitar recording into guitar tablature. The pipeline is
`audio → note events → (string, fret) candidates → fingering scorer → decoder → tab`, and
every stage sits behind a `typing.Protocol` in `src/tabsampler/types.py` so one can be
swapped without touching the others. Phase 0 (an evaluation harness built before any model)
and Phase 1 (a hand-set cost model decoded with Viterbi plus forward-backward posteriors)
are both **done, measured and tagged**, and **Phase 1.5 has closed M1's known defects**
(branch `phase-1.5`, ADRs 0016-0019). What remains is a learned fingering model, audio
conditioning, and an app a guitarist can use.

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
make check                         # lint + pyright --strict + 325 tests. Must be green.
make oracle                        # the correctness core. Must be green.
```

`make check` piped into `tail` hides its exit code — check the status, not the output.

Then read, in this order: `docs/spec.md` → `docs/plans/2026-09-27-rest-of-project.md` →
`docs/devlog/2026-09-27-phase-1.5.md` (the most recent session) →
`docs/devlog/2026-09-27-m1-retrospective.md` → `docs/adr/README.md`.

## Where things stand

| | |
|---|---|
| Repo | `https://github.com/Afillex/TabSampler` — **public**, MIT (ADR 0020) |
| Tags | `v0.0-phase0`, `v0.1-m1` |
| Tests | 325, all offline — no test needs the dataset or the transcriber |
| CI | GitHub Actions, green, ~30 s |
| Current phase | Phase 2, **blocked**. Phase 1.5 done; the app track (chunk B) is next |

**Current results** — 360 GuitarSet tracks, `audio_mic`, hand-set weights (not tuned).
The `commit` column of `experiments/results.csv` references the pre-publication history,
which was squashed into the initial commit; the rows are still the record of which runs
produced which numbers, and `make eval-m1` reproduces them. M1's figures are in brackets where Phase 1.5 moved them:

| | oracle | end-to-end |
|---|---|---|
| E1 note F1, transcriber (raw) | 1.0000 | 0.7437 |
| E1 note F1, pipeline (placed) | 1.0000 | 0.7452 |
| **E2 exact tab F1 (headline)** | **0.6599** (M1 0.6366) | **0.4318** (M1 0.4231) |
| E3 playable groups | 0.9970 (M1 0.9835) | 0.9896 (M1 0.9751) |
| E3 playable transitions | 0.9134 | 0.9164 |
| E4 pitch validity | 1.0000 | 1.0000 |
| E5 calibration error | 0.1652 | 0.3851 |

E1 = 1.0 in oracle mode is a plumbing check, not a result. **E4 = 1.0 is an invariant, not a
score: anything below 1.0 is a bug**, because it means a (string, fret) pair does not sound
the pitch we claimed. **E5 is the weak result** — at 0.39 the posteriors are not honest yet,
and it got slightly worse in Phase 1.5.

**E1 is two numbers now.** Raw = the transcriber's score, and it reproduces Phase 0's 0.7437
exactly. Placed = the pipeline's, after placement drops notes the guitar cannot sound. M1
reported only the second and compared it with Phase 0's first.

## The one thing blocking most of the remaining work

**Ege must send one email.** DadaGP and ProgGP are both symbolic Guitar Pro corpora,
access-by-request for research use, and Pedro Sarmento is a contact on both. That single
request gates **three** things: all of Phase 2, the cost-weight tuning still outstanding from
M1 (ADR 0012), and ADR 0011's owed validation.

**If it has not been sent, say so at the top of the session** rather than starting Phase 2
work that cannot finish. As of the Phase 1.5 session it had not been. Chunk A is now done;
chunk B needs nothing external either.

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

Still open:

- **The hand-set weights are demonstrably wrong, and Phase 1.5 is what showed it.** `move`
  is 1.0 per fret against `high` at 0.1 per *octave* — 120:1 — so once movement is charged
  honestly the decoder parks the hand high up the neck rather than playing in open position.
  ADR 0018 has the worked example. **Do not reweight without validation data** (ADR 0012);
  this is the single most valuable thing the DadaGP request buys.
- **E3's transition rule contradicts ADR 0011's own text.** The ADR says an all-open group
  "carries the previous [hand position] forward rather than resetting to fret 0", but
  `transition_is_playable` returns "playable" whenever either shape is all open. Reproduced:
  fret 2 → open string → fret 20 at 50 ms spacing (18 frets in 0.1 s) scores 2/2
  transitions, while the same move without the open string in between scores 0/1 with
  "180.0 frets/s exceeds 12.0". So E3
  under-reports fast jumps across an intervening open chord — the same defect ADR 0018 just
  fixed in the cost model, still live in the metric. Unowned; needs its own task because it
  moves E3 and requires choosing between the ADR's text and the metric's behaviour.
- **A still-ringing note does not reserve its string** against the next group (spec §2.2).
  Not yet owned; revisit if metrics show it matters.
- **ADR 0011's span thresholds are unvalidated and probably too strict.** GuitarSet's
  *first* track contains a human-played chord those rules call unplayable, and span pruning
  measurements show span 5 is the smallest bound at which no real chord in a 60-track sample
  is unfingerable — against 4 at the ADR's limit. ADR 0011 owes a validation against real tab.
  **Do not edit ADR 0011**; supersede it when the data arrives, as ADR 0019 did for its
  finger rule.
- **The config loaders have no tests.** `load_phase1_config` and `load_eval_config` are
  untested; the `max_fingers` rename was verified by hand only.
- **`eval-m1` writes the config's hypothesis into `results.csv`**, so the Phase 1.5 rows
  carry M1's hypothesis text. The numbers and commits are right; that column is stale.

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
  hand position carried into it. Only all-open shapes are augmented, which costs 1.07× the
  state count on real data and 3.03× on a line of nothing but open-string pitches. Re-measure
  with `scripts/measure_lattice.py`; benchmark decode with `scripts/bench_decode.py`. Anything reading per-shape quantities out of `forward_backward`
  must sum over nodes; `note_posteriors` does.
- **E7 in `results.csv` is too noisy to compare across runs.** The `eval-m1` column measured
  0.0370, 0.17, 0.20 and 0.04 s per audio minute for equivalent work on this machine in one
  afternoon. Benchmark decode directly instead.
- **`tests/decode/brute_force.py` is the specification.** It enumerates every path and scores
  it from first principles. It is the independent check that the decoder is correct
  (ADR 0014), so a change to it deserves more scrutiny than a change to the decoder. If they
  disagree, the decoder is wrong.
- **A 4-track smoke test gave oracle E2 = 0.90 where all 360 gave 0.64.** Never quote a number
  from a subset.

## Open item that needs Ege, not you

- **D15 — weight release, the half ADR 0020 left open.** The code is MIT and the repo is
  public. Weights are a **per-corpus** decision, because DadaGP and ProgGP are
  research-use-only: check the terms *before* training anything whose weights might be
  published, not after.

## When you finish a session

Write `docs/devlog/YYYY-MM-DD.md`: what was done, what was measured, what's next, open
questions. Update this file if the
state a fresh agent needs has changed.
