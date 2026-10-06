# HANDOFF — read this first

You are picking up Tab Sampler with no prior context. This file is the shortest path to
being useful. Written 2026-09-27 after Milestone M1; last updated 2026-10-06, after
Phase 5 closed.

## In one paragraph

Tab Sampler turns a guitar recording into guitar tablature. The pipeline is
`audio → note events → (string, fret) candidates → fingering scorer → decoder → tab`, and
every stage sits behind a `typing.Protocol` in `src/tabsampler/types.py` so one can be
swapped without touching the others. Phase 0 (an evaluation harness built before any model)
and Phase 1 (a hand-set cost model decoded with Viterbi plus forward-backward posteriors)
are both **done, measured and tagged**, and **Phase 1.5 has closed M1's known defects**
(ADRs 0016-0019). **DadaGP arrived on 2026-10-01** and Phase 2 used it: the training
protocol is fixed (ADR 0021) and every fit now uses an artist-disjoint split (ADR 0024);
both halves of E3 are measured against human tab (ADRs 0022, 0031, 0036); the hand is a 4-fret
window that a wide chord can stretch (ADRs 0025, 0030); and clean and distorted guitar have
a decoder each (ADR 0032) with their own temperatures (ADR 0033) — the clean one, fitted on
clean DadaGP playing, became the default and was then re-decided away (ADR 0038). The
first richer features did not earn a place (ADR 0034). **GuitarSet's player 00 is now
validation data and players 01–05 the test set (ADR 0037)**, because DadaGP's clean parts
mispredicted GuitarSet three times. **Phase 2 closed with M2 missed** (ADR 0045), and **Phase 3
measured audio evidence trained on SynthTab: it costs the decoder on GuitarSet** (ADR 0048).
**Phase 4** made clean electric guitar the target and EGDB a second test set; electric-trained
audio evidence helps on EGDB (ADR 0054). **Phase 5** found Basic Pitch trained on GuitarSet (ADR
0055) and improved end to end with stricter transcriber thresholds (ADR 0057). What remains:
full songs (Phase 6), techniques (Phase 7), and an app a guitarist can use (the B track).

## Where the last session stopped (2026-10-06) — read before anything else

**Phase 5, a better transcriber, is closed (Ege's sign-off, ADR 0057), merged into `main` and
public.** Plan
`docs/plans/2026-10-06-phase-5-transcriber.md`; the day's entry is `docs/devlog/2026-10-06.md`.

- **Basic Pitch trained on 90% of GuitarSet** (its paper; ADR 0055): GuitarSet's end-to-end
  figures are labelled, and transcribers are compared and tuned only on Guitar-TECHS's player 3
  and EGDB.
- **The result:** Basic Pitch at onset 0.7 / frame 0.4 / minimum note length 58 ms, chosen on
  player 3 (`scripts/tune_transcriber.py`): EGDB end-to-end E2 0.4251 → **0.4547**; GuitarSet,
  labelled, 0.4418 → 0.4553. `tabsampler transcribe` uses them (`CHOSEN_PARAMS`); the eval
  configs that recorded earlier figures keep Basic Pitch's defaults.
- **Fine-tuning failed** (ADR 0056): a separate TensorFlow environment
  (`scripts/setup_transcriber_training.sh`, `~/.local/share/tabsampler/bp-train`), Basic Pitch's own
  targets and loss, CoreML export, `--model-path`. **basic-pitch ignores `--model-path` whenever
  `--model-serialization` is given** — the adapter now passes a model alone, and the cache key
  hashes the model's files.
- `scripts/analyse_e2e.py` splits the end-to-end loss into note errors and context;
  `compare_validation.py` uses a ratio bootstrap end to end, where the sides count different notes.

**Next, Ege's:** Phase 6 (full songs), the app track (B1–B4; D14 is long due), or more of Phase 5 —
heavier transcribers (Task 5, unsurveyed) or fine-tuning with an onset weight between the two tried.

Phase 4, real electric audio, closed before it (ADR 0054). Plan `docs/plans/2026-10-06-phase-4-electric.md`,
every task done; the day's entry is `docs/devlog/2026-10-06.md`. Phases 2 and 3 are closed and
public: Phase 2 with M2 missed (ADR 0045), Phase 3 a negative result (ADR 0048). Weights trained
on DadaGP or SynthTab, or started from them, stay unpublished (ADRs 0042, 0046).

- **D2 is clean electric first** (Ege, ADR 0049); acoustic stays measured, on GuitarSet.
- **EGDB is the second test set** (ADR 0050): all 240 clips, direct input, in a committed
  snapshot under the same guard as GuitarSet (`data/splits.py`, `egdb_test_ids`). Its string is
  the **MIDI channel** (0 high e … 5 low E), not the track name (ADR 0053). Google Drive throttles
  `scripts/download_egdb.py` after a few dozen files; Ege downloaded it in a browser.
- **Guitar-TECHS** (`data/guitar-techs/`, CC BY 4.0, 4.1 GB zipped): players 1–2 train, player 3
  validates (ADR 0051). Its pickup MIDI needs cleaning: glitches under 60 ms dropped, overlaps
  trimmed, bends and harmonics left out, and **each take's label delay measured from its own
  audio** (`onset_lag`, ADR 0052) — some of player 2's takes are 50–65 ms early, most 10–25 late.
  `scripts/check_guitartechs.py` checks labels against audio, GuitarSet's player 00 the control.
- **Measured** (ADR 0054): the classifier fine-tuned on Guitar-TECHS from SynthTab
  (`cache/acoustic/gt-ft`, T 0.6957, weight 0.25) is only 0.37 accurate per note on player 3, no
  better than SynthTab's, yet in the decoder: **EGDB (electric, test) 0.6762 → 0.7200**;
  GuitarSet player 00 −0.0230 (a clear loss), test players +0.0057. The decoder alone: EGDB
  0.6762 against GuitarSet's 0.6819. End to end on EGDB: E1 0.7230, E2 0.4251.
- **Scripts:** `train_strings.py --corpus guitartechs [--init]`, `calibrate_acoustic.py --corpus
  guitartechs`, `evaluate_strings.py --examples`, `evaluate_egdb.py` (the one logged look; `--note`
  for the log's reason). The access log has 32 lines; lines 31 and 32 are one look at EGDB whose
  first run stopped reading labels before any metric.

Still open from Phase 4: an electric decoder config with the audio term (it needs a validation
figure that is not the calibration's own).

`make check` (571 tests) and `make oracle` (13) pass on `main`.

Deferred small items, not yet fixed: `fit_weights` silently accepts unknown or empty
`active` names; `paired_bootstrap({}, {})` fails with a raw numpy error; `CostWeights.
string_bias` lacks ADR 0007's tuple guard; the fret-region boundaries (4/5, 11/12) are
untested; ADR 0035's Context was corrected at acceptance without saying so (recorded in
the 2026-10-03 devlog). The decisions waiting for Ege are listed under "Open items" below.

## Rules that are not negotiable

1. **GuitarSet's players 01–05 and EGDB are test-only** (ADRs 0003, 0037); GuitarSet's
   player 00 is validation data. No tuning, selection, or threshold
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
make check                         # lint + pyright --strict + 571 tests. Must be green.
make oracle                        # the correctness core. Must be green.
```

`make check` piped into `tail` hides its exit code — check the status, not the output.

Then read, in this order: `docs/spec.md` → `docs/devlog/2026-10-06.md` (Phases 4 and 5) →
`docs/plans/2026-10-06-phase-5-transcriber.md` and `2026-10-06-phase-4-electric.md` (both done) →
`docs/devlog/2026-10-04.md` (Phase 2's close and Phase 3) → `docs/plans/2026-09-27-rest-of-project.md` → `docs/adr/README.md`.

## Where things stand

| | |
|---|---|
| Repo | `https://github.com/Afillex/TabSampler` — **public**, MIT (ADR 0020) |
| Tags | **None on GitHub.** `v0.0-phase0` and `v0.1-m1`, listed here before, exist nowhere; Phase 0's and M1's commits, `d50f4f0` and `1147495`, survive only locally, under the tag `pre-publication-backup` |
| Tests | 571, all offline — no test needs the dataset or the transcriber |
| CI | GitHub Actions, green, ~30 s |
| Current phase | **Phase 5, a better transcriber — closed 2026-10-06 by Ege's sign-off** (ADR 0057); Phase 4 closed (ADR 0054) |

**Current results** — GuitarSet's test players 01–05, 300 tracks, `audio_mic`, today's
default `configs/decoder_clean.yaml`: the hand-set weights plus ADR 0039's open-string cost,
T = 1.2934, the hand window with the stretch — measured once, as Phase 2's test evaluation
(ADR 0045). **It was chosen on player 00 alone, so this is an independent check** (+0.026 on
the test players after +0.021 on player 00); figures for decoders chosen before ADR 0037 are
baselines, because the 300 tracks were part of every earlier 360-track run. Rows before
2026-10-01 have a `commit` column that references the pre-publication history, which was
squashed into the initial commit; the rows are still the record of which runs produced which
numbers. Likewise, commit ids recorded on 2026-10-02 and 10-03 — in `results.csv`,
`experiments/test_set_access.log`, ADRs 0026–0028 and that devlog — name commits from before a
wording clean-up of the history on 2026-10-03. Each has a counterpart with the same message, in
the same order, with the same code. The run was
`uv run tabsampler eval-m1 --split test --config configs/p5_test_eval.yaml` (Phase 5's look):
the end-to-end column uses Basic Pitch at the thresholds Phase 5 adopted (ADR 0057); its defaults'
figures (ADR 0045, `configs/m2_c6_eval.yaml`) are in brackets. **Basic Pitch trained on most of
these recordings (ADR 0055): the end-to-end column is labelled; EGDB's is the clean figure**
(oracle 0.6762, end to end 0.4547, E1 0.7416).

| | oracle | end-to-end |
|---|---|---|
| E1 note F1, transcriber (raw) | 1.0000 | 0.7812 (0.7493) |
| E1 note F1, pipeline (placed) | 1.0000 | 0.7818 (0.7507) |
| **E2 exact tab F1 (headline)** | **0.6819** | **0.4553** (0.4418) |
| E3 playable groups | 0.9978 | 0.9956 (0.9906) |
| E3 playable transitions, 48 frets/s rule | 0.9996 | 0.9979 (0.9981) |
| E4 pitch validity | 1.0000 | 1.0000 |
| E5 calibration error | **0.0792** | 0.1841 (0.2439) |

E1 = 1.0 in oracle mode is a plumbing check, not a result. **E4 = 1.0 is an invariant, not a
score: anything below 1.0 is a bug**, because it means a (string, fret) pair does not sound
the pitch we claimed. **Every ADR 0016 value holds on the 300** — the oracle transition rate on
its guardrail at four decimals — and **M2's target of 0.760 is missed by 7.8 points**. Player
00, the validation player, is much easier (oracle E2 0.83), so its absolute figures do not carry
over to the test players. The spec's Phase 2 comparison on the test players: the decoder 0.6819,
the learned term alone 0.4514, the learned model 0.6856 (ADR 0045; the learned model was 0.030
*below* the decoder on player 00). **Phase 3's audio evidence lowers the default's oracle E2 on
the test players to 0.6073** (ADR 0048), so the default leaves it off.

**E1 is two numbers now.** Raw = the transcriber's score; on the same 360 tracks it
reproduces Phase 0's 0.7437 exactly (0.7493 on the 300 test tracks). Placed = the
pipeline's, after placement drops notes the guitar cannot sound. M1 reported only the
second and compared it with Phase 0's first.

## Training data: DadaGP, and how it may be used

**DadaGP v1.1 arrived on 2026-10-01; ProgGP did not.** The archive lives at
`data/dadagp/DadaGP-v1.1.zip` (gitignored — **never commit it, the repo is public**), and
`data/dadagp/track_meta.json` records which songs are clean standard-tuned guitar. Rebuild the
latter with `scripts/dadagp_track_meta.py` (its docstring has the command).

ADR 0021 is the protocol, and it is not optional: fit on DadaGP **training**, select and
calibrate on DadaGP **validation**, evaluate on GuitarSet **once**, and never choose anything
by looking at GuitarSet. **Use the artist-disjoint split** (`--split artist`, ADR 0024) for
every fit: whole artists sit on one side, frozen by hash. `load_tracks` has no default
scheme and the scripts refuse to run without `--split`, so the choice is always explicit. The shipped split, whose artists
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

Also: **Typer colours the CLI's error messages in GitHub Actions** (it forces a terminal
when `GITHUB_ACTIONS`, `FORCE_COLOR` or `PY_COLORS` is set), so a test that looks for
text in CLI output must strip colour codes first, as `plain()` in `tests/test_cli.py`
does; locally the same test passes without it. Reproduce with
`GITHUB_ACTIONS=true uv run pytest`.

And **ruff 0.16 formats Python code blocks inside Markdown** and will rewrite
`docs/spec.md` if you let it. `docs/**` is excluded for that reason — leave it excluded.

## Known defects

**Closed in Phase 1.5:** hand movement across an all-open chord (ADR 0018) and barre chords
scored as unplayable (ADR 0019). Both moved reported numbers; see
`docs/devlog/2026-09-27-phase-1.5.md`.

**Closed in Phase 2 so far:** E3 not carrying the hand across an all-open shape, and E3
and the cost model counting finger reach as hand movement (ADR 0025); E3 timing moves
across open strings from the wrong group (ADR 0029); E3's unmeasured speed limit (ADR
0031); wide chords charged phantom movement (ADR 0030); `eval-m1` labelling every row
"hand-set". The fitted-versus-hand-set question of ADR 0023 is settled by ADR 0027.

Still open:

- **Validation is one player.** 60 tracks give wide intervals, and player 00 favoured the
  clean fit where the other players do not (ADR 0038): treat a validation result as
  evidence, not proof, and keep asking a challenger for a clear win.
- **A still-ringing note does not reserve its string** against the next group (spec §2.2).
  Not yet owned; revisit if metrics show it matters.
- **`eval-m1` writes the eval config's hypothesis into `results.csv`.** Give every new
  evaluation its own config with its own hypothesis, as `configs/m2_style_eval.yaml` does;
  the Phase 1.5 rows still carry M1's hypothesis text.

## Things that are settled — don't relitigate them

Read the ADR before proposing a change to any of these. `docs/adr/README.md` is the index.

- GuitarSet's players 01–05 are test-only and player 00 is validation (ADRs 0003, 0037),
  and `audio_hex` is **forbidden as transcriber input** because one channel per string is
  the E2 ground truth (ADR 0005).
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
  hand carried into it — since ADR 0030 a fret range, `Hand = (index, highest)`. Every shape
  gets one node per reachable hand: 2.84× the state count on DadaGP validation, and decode
  is about twice as slow as before the window (`decode`, 500 groups: 58.3 → 120.1 ms).
  Re-measure with `scripts/measure_lattice.py`; benchmark with `scripts/bench_decode.py`.
  Anything reading per-shape quantities out of `forward_backward` must sum over nodes;
  `note_posteriors` does.
- **`fingering.states.shift_window` is the one definition of the hand.** E3, the cost
  model, the decoder's lattice and the CRF fitter all call it; the brute-force oracle
  re-derives it from ADR 0030's text, as a literal search, and must not import it.
- **The per-string biases overlap with `high`.** For single notes the mean fret is close to
  a linear function of the string, so a fit trades one against the other (in the clean
  per-string fit `high` went 0.89 → 1.48). Read fitted string biases only together with
  `high`; regularise before trying more feature groups.
- **E3's speed limit, in words:** human tab passes 48 frets/s on 99.88% of transitions and
  on 98.6% of the moves where the hand actually shifts (ADR 0036) — not "99.86% of moves".
- **Two decoders, one default.** `configs/decoder_clean.yaml` is the CLI's default and
  `configs/decoder_distorted.yaml` the alternative (ADR 0032); `configs/phase1_baseline.yaml`
  is the record of the hand-set baseline, no longer the default.
- **Comparing two decoders on validation**: `scripts/score_validation.py` writes per-song
  counts to `cache/validation/` (never committed: they name DadaGP songs), and
  `scripts/compare_validation.py` gives the paired difference with a song-level bootstrap
  interval. `scripts/fit_cost_weights.py --part {clean,distorted} --features string,region
  --per-song-out ... --weights-out ...` writes the same for its fits. On GuitarSet,
  `eval-m1 --split validation --per-track-out ...` writes the same per track; it is refused
  on the test split, and `compare_validation.py` refuses any file holding a test track.
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

- **The next phase**: Phase 6 (full songs), the app track (B1–B4; D14, the app surface, is long
  due), or more of Phase 5 (heavier transcribers; fine-tuning with an intermediate onset weight).
- **An electric decoder config with the audio term** (Phase 4 loose end).
- **Disk**: Guitar-TECHS's zips (4.1 GB) and `cache/transcriber/data` (1.3 GB) can go when no
  longer needed; about 20 GB is free.
- **More data**: GOAT (access unclear, by request) and more of SynthTab (2 TB) are Ege's calls.
- **The chord-shape condition** on player 00 is *no clear drop* (Ege, 2026-10-04, ADR
  0039). Still open: whether the same test should re-judge the fitted distorted models,
  which the fixed 0.0005 blocked on DadaGP, each 7.7 to 10.3 points better at recovering
  human fingerings.
- **The clean temperature**: for the clean-fitted weights T = 1 had a lower calibration
  error than the likelihood optimum; for today's default the optimum, 1.2934, also has the
  lower error (0.0707 against 0.0795 at T = 1). A temperature calibrated on player 00
  itself (task C5) is still to do.
- **D15 — weight release, the half ADR 0020 left open.** The code is MIT and the repo is
  public. Weights are a **per-corpus** decision, because DadaGP and ProgGP are
  research-use-only: check the terms *before* training anything whose weights might be
  published, not after.

## When you finish a session

Write `docs/devlog/YYYY-MM-DD.md`: what was done, what was measured, what's next, open
questions. Update this file if what a newcomer needs to know has changed.
