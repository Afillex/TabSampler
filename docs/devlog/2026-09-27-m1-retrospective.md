# 2026-09-27 — M1 retrospective

Phase 0 and Phase 1 built in one session, both gates met. Full numbers are in the
README and in `experiments/results.csv`; this is what happened and what it cost.

## The discipline checklist, run honestly

- [ ] I changed two things and reported one result.
- [ ] I looked at test-set numbers while tuning — no tuning happened (ADR 0012), and both
      test-set looks are in `experiments/test_set_access.log`.
- [ ] There's logic in a notebook the app depends on — no notebooks.
- [ ] A file over ~400 lines with no tests — largest is `cli.py` (~430); every `src/`
      module has tests.
- [ ] Building UI or techniques before the gate — no.
- [x] **Citing another paper's number as comparable** — avoided, and the README says so
      explicitly, but TART's figures are in `docs/spec.md` and a reader will be tempted.
- [ ] Fixed a bug without a test that would have caught it — the two real bugs found
      (`max_span` 4, out-of-range notes) both have tests.

## Six findings worth more than the numbers

1. **`mir_eval.transcription` wants Hz, not MIDI** (`transcription.py:435` derives cents
   from a frequency ratio). MIDI 40 vs 41 computes as 42.7 cents — inside the 50-cent
   tolerance — so a semitone error would have scored as a **hit**. Pinned by a test.
2. **basic-pitch cannot be installed alongside the core** on macOS arm64: it needs
   `tensorflow-macos` for Python > 3.11 and that wheel stops at cp311. Hence ADR 0001's
   two environments.
3. **And it does not work out of the box:** it pins `resampy<0.4.3`, resampy imports
   `pkg_resources`, setuptools removed that in v81. `setuptools<81` is required.
4. **Its note-event CSV is ragged** — the header names 5 columns but rows carry one
   column per bend value (17, 72, 18, 69, 21, 65 in the captured fixture). A
   `csv.DictReader` would have kept the first bend value and silently dropped the rest.
5. **`max_span = 4` was wrong, and real data proved it.** GuitarSet's first track has the
   chord `[51, 55, 58, 62, 67, 70]`, whose only string assignment uses frets 6-11 — a
   span of 5. The strict decoder raised and one chord killed the whole file.
6. **A 4-track smoke test gave oracle E2 = 0.90; all 360 gave 0.64.** The first four
   tracks are one player on one progression. This is the single best argument in the
   session for never quoting a number from a subset.

## Questions worth answering before Phase 2

- Why does Viterbi give the single best path while forward-backward gives per-note
  probabilities, and when do they disagree? `decode()` deliberately takes the position
  from one and the confidence from the other — why can it not take both from the
  marginals?
- Why log space? What specifically breaks at 500 groups without it?
- What is the decoder's complexity, and how does span pruning change it?
  (`state_count_stats` exists to answer this with a measurement.)
- Why is E5 so much worse end to end (0.37) than in oracle mode (0.16)?

## What's next

Concrete next experiments, one variable each:

1. **Calibrate the temperature.** Hypothesis: ECE falls below 0.10 in oracle mode at some
   T < 1.0. Variable: `weights.temperature`. Metric: E5. Split: needs validation data —
   see below, this cannot legally be fit on GuitarSet.
2. **Request DadaGP access** (ADR 0012). It unblocks both cost-weight tuning and ADR
   0011's owed validation — the check that real human tab passes our playability rules,
   which finding 5 above already suggests it does not.
3. ~~Measure spec 4's actual target~~ — **done, see the addendum below. PASS, 3.70 s
   against a 60 s target.**

Phase 2 does not start until those are done.

## Open questions

- Oracle E2 is 0.64 with untuned weights. How much of the remaining 36% is the cost
  model being wrong versus genuinely ambiguous fingering that a human chose arbitrarily?
  A human-agreement baseline on a handful of tracks would bound it.
- E1 end to end rose slightly from Phase 0 (0.7437 → 0.7452) because `decode_best_effort`
  drops out-of-range and unfingerable notes, which raises precision. Is dropping them the
  right call for the *reported* E1, or should E1 be measured before the fingering stage
  touches anything?
- `results.csv` stamps UTC while devlog filenames use local date. Pick one.
- Oracle mode still has 4 unfingerable groups at span 4 and 0 at span 5, but E3 judges
  against ADR 0011's 4-below-12 rule. So the decoder is now permitted to produce shapes
  that E3 then calls unplayable. That is not a bug — it is the rules being wrong — but it
  means E3's group rate (0.9835) is partly measuring our thresholds, not our output.
  The DadaGP validation should settle it.

## Addendum: the two owed measurements, now made

Both were flagged as outstanding above. Neither involves tuning or model selection, so
neither touches ADR 0003. The GuitarSet audio access for the runtime benchmark is logged
in `experiments/test_set_access.log` anyway.

### Spec 4's performance target: PASS, with 16x headroom

Eight GuitarSet mic clips concatenated into one 188.7 s (3.14 min) file — real playing
rather than synthetic, because decode cost scales with note density. Cold transcriber
cache.

| stage | time |
|---|---|
| transcribe (basic-pitch, cold) | 3.59 s |
| group + decode | 0.11 s |
| **total** | **3.70 s** |

Target is a 3-minute file under 60 s on the M4. Measured 3.70 s — **16x headroom**.
E4 = 1.0000 on the output. 680 groups, 1099 notes placed.

**And this answers the open question about per-file process startup.** E7 here is
1.18 s per audio minute; the Phase 0 corpus run measured 2.73. The difference is
basic-pitch's model load, paid once here and 360 times there. Per-file process startup
dominates E7 for 30-second clips and is irrelevant for real files. So the Phase 0 E7 is
the right number to quote for *corpus evaluation* and the wrong one to quote for *what a
user experiences*.

Decode is 3% of the total. Optimising the decoder would be pointless: optimise only after
measuring, and the measurement says stop here.

### Decoder complexity: what span pruning actually buys

60 of the 360 tracks (every 6th, so spread across all players and progressions).
Mean states per group, and how many groups end up with no legal state at all:

| span | oracle mean | oracle max | oracle unfingerable | e2e mean | e2e unfingerable |
|---|---|---|---|---|---|
| 2 | 3.30 | 19 | 56 | 2.97 | 294 |
| 3 | 3.72 | 19 | 8 | 3.57 | 169 |
| 4 | 4.06 | 27 | **4** | 3.93 | 107 |
| **5** | **4.92** | **31** | **0** | **4.67** | **84** |
| 6 | 5.49 | 39 | 0 | 5.02 | 80 |
| 8 | 6.72 | 39 | 0 | 6.50 | 70 |
| 12 | 11.33 | 68 | 0 | 10.30 | 67 |
| 22 (unpruned) | 17.67 | 152 | 0 | 15.37 | 65 |

6611 oracle groups, 6232 end-to-end.

Three things fall out of this:

1. **Span 5 is exactly the smallest bound at which every real human-played chord in the
   sample is fingerable.** At 4, four groups have no legal state; at 3, eight; at 2,
   fifty-six. This is independent evidence for the `max_span` fix, and more evidence that
   ADR 0011's "at most 4 frets below fret 12" is too strict for real playing — the rule
   says unplayable about chords a human demonstrably played.
2. **Pruning at span 5 costs nothing and buys a lot.** Mean states fall from 17.67 to
   4.92, which is 27.8% of unpruned, with zero real chords lost. Viterbi is O(T·S²) in
   states per group, so the transition work drops by roughly (17.67/4.92)² ≈ **13x** on
   average and (152/31)² ≈ **24x** at the worst group.
3. **65 end-to-end groups are unfingerable even unpruned.** Span cannot fix those: they
   are transcriber errors, mostly more simultaneous notes than the guitar has strings.
   That is 1.0% of groups, and it is exactly why `decode_best_effort` drops notes rather
   than relaxing further.

Reproduce: the measurement script is throwaway and lives in the scratchpad, not the repo.
It calls `state_count_stats`, which is in `fingering/states.py` for this purpose.
