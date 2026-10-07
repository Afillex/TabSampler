# Phase 6 — Full songs

> Tasks run in order; steps use checkbox (`- [ ]`) syntax. Every rule that picks or adopts
> something is committed before the run it judges; every test-set look is pre-registered.

**Status:** started 2026-10-07; Ege chose Phase 6 after the electric path (ADRs 0062–0064).

**Spec §6:** "`htdemucs_6s` guitar stem, with an experiment feeding stem + mix. Report results
separately from isolated mode." ADR 0002: isolated and full-song results are separate tables,
never merged.

**Done when (proposed; Ege signs off):** on both test sets (EGDB; GuitarSet, labelled), end-to-end
E2 is reported for full-song mixes three ways — the mix straight into Basic Pitch, the separated
guitar stem, and the stem with the mix added back as chosen on validation — beside the isolated
figure; and the app accepts a full song through a "separate the guitar first" option.

## Where this starts

- Isolated guitar, end to end on EGDB: E2 0.4547 (default), 0.4992 (electric option).
- **No public dataset found has full-band mixes with string labels** (searched 2026-10-07; GOAT,
  GuitarSet, EGDB, Guitar-TECHS and the rest are guitar alone). So the mixes are built: a labelled
  guitar take plus backing from a multitrack corpus with its guitars removed, so ours is the only
  guitar and every guitar note has a label.
- **Backing (Ege, 2026-10-07): BabySlakh now, MoisesDB later.** BabySlakh (CC BY 4.0, Zenodo
  4603870): the first 20 songs of Slakh2100, synthesized, **16 kHz** — so separation may look
  easier than on real songs. MoisesDB (real songs, CC BY-NC-SA, by registration at music.ai) is
  the later, more realistic check. Ruled out: Slakh2100 whole (104 GB), MUSDB18 and OnAir (their
  guitars sit inside "other" and cannot be removed).
- `htdemucs_6s` (demucs 4.1.0) resolves on Python 3.13 with the project's PyTorch 2.14.1 and
  separates 20 s in 5 s on this CPU (checked in a throwaway environment, 2026-10-07).

## Global constraints

- Isolated and full-song results never share a table or a row.
- Validation: EGSet12 and IDMT's licks (electric), GuitarSet player 00 (acoustic, labelled);
  BabySlakh songs 1–10 as their backing. Test: EGDB and GuitarSet's players 01–05 (labelled),
  BabySlakh songs 11–20 as their backing. A backing song is never in both.
- The mix's guitar labels are the take's own, delay taken off as before (ADRs 0052, 0062).
- One variable per measurement; demucs enters only with an ADR.

## Task 1: Building full-song mixes (ADR)

- [x] Download BabySlakh; check its stems and which are guitars (its metadata's instrument
      classes). *20 songs, 161–348 s, 16 kHz; 1–5 guitar stems each, 3–14 others; "Ethnic"
      (plucked strings) left out with the guitars.*
- [x] `data/mixes.py`: for a labelled take and a backing song, the backing's non-guitar stems summed,
      resampled, cut or looped to the take's length, and mixed at a stated guitar-to-backing level;
      pairing by a hash of the take's name, so it never depends on order. Pure, tested.
- [x] An ADR (0066): the mix construction, the levels (0 dB and −6 dB guitar-to-backing, both reported),
      the backing split, and what the mixes cannot show (no shared key or tempo between guitar and
      backing; 16 kHz backing).
      *Validation built: 414 mixes (EGSet12 12, IDMT 135, GuitarSet player 00 60, at two levels).*

## Task 2: Separation (ADR for the dependency)

- [x] demucs in its own dependency group; `audio/separate.py` runs `htdemucs_6s` and returns the
      guitar stem, cached by the audio's hash as transcriptions are. *ADR 0065; the project now
      declares its platforms (Apple-silicon Macs, Linux). First real run, an isolated EGSet12 piece:
      8 s in 4.1 s, but the guitar stem kept only 0.381 of the input's RMS — on clean electric
      guitar htdemucs_6s puts much of the guitar elsewhere. Measured in this task, not assumed.*
- [x] **Pre-registered on validation** (28f40bc): `scripts/evaluate_mixes.py --add-mix 0` on the
      414 validation mixes. *Single variable:* Basic Pitch's input — the mix, or its htdemucs_6s
      guitar stem. Beside them: the isolated take (ceiling) and the isolated take through the
      separator (separation's own cost). Default decoder, `CHOSEN_PARAMS`. *Metric:* end-to-end E2
      pooled over takes, per level (0 and −6 dB) and per set (EGSet12, IDMT, GuitarSet player 00),
      stem against mix with a bootstrap interval over takes. *Predictions:* (1) the mix at 0 dB falls
      to at most 0.6 of the isolated E2, lower at −6 dB; (2) the stem beats the mix at both levels,
      pooled, with intervals above zero; (3) the stem stays below the isolated take; (4) separating
      the isolated take costs 0.02 to 0.10 (its first stem kept 0.38 of the RMS). *Rule:* Task 3
      starts from the stem if it beats the mix pooled at both levels; otherwise reported to Ege.
      *Result (pooled E2):* isolated 0.4666; isolated through the separator 0.4541; **0 dB: mix
      0.1902, stem 0.3180 (+0.1278 [+0.1032, +0.1497]); −6 dB: mix 0.1252, stem 0.2572 (+0.1320
      [+0.1039, +0.1562])**. Predictions 1–3 held; 4 failed — pooled, separation alone cost only
      0.0125, but **IDMT's direct-input takes lost 0.1226 to separation alone** (0.3166 → 0.1940),
      and on IDMT the stem does not beat the mix (0 dB −0.0172 [−0.0426, +0.0082]). EGSet12 and
      GuitarSet (microphone) gain +0.08 and +0.17. htdemucs_6s handles clean direct-input guitar
      poorly; EGDB is direct input too. By the rule, Task 3 starts from the stem.

## Task 3: Stem plus mix

- [x] **Pre-registered on validation** (efb8972): `scripts/evaluate_mixes.py --add-mix 0 0.1
      0.25 0.5` — Basic Pitch hears the stem with a share α of the mix added back (0 is the stem
      alone). *Single variable:* α. *Metric:* end-to-end E2 pooled over the three validation sets,
      averaged over the two levels. *Rule:* the α with the highest such E2 is adopted if it beats the
      stem alone (α = 0) by at least +0.01; otherwise the stem alone stays. *Prediction:* the best α
      is 0.1 or 0.25, ahead of the stem by 0 to +0.02, most of it on IDMT, where separation alone
      lost 0.12; α = 0.5 falls back towards the mix. The electric option is reported in Task 4.
      *Result:* every share lowers E2, monotonically — pooled at 0 dB stem 0.3180, +0.1 0.2845,
      +0.25 0.2556, +0.5 0.2338; at −6 dB 0.2572, 0.2188, 0.1886, 0.1724. **The stem alone stays**
      (α = 0); the prediction failed. Only IDMT gains a little from 0.25 (+0.019 at 0 dB). The stem
      figures reproduce Task 2's to four decimals.

## Task 4: The test look

- [ ] **The test look, pre-registered here** (this commit): `scripts/build_mixes.py --split test`,
      then `scripts/evaluate_mixes.py --split test --add-mix 0 --electric` — EGDB's 240 clips and
      GuitarSet's 300 test tracks (labelled, ADR 0055) over BabySlakh songs 11–20, at 0 and −6 dB;
      inputs: the mix and the htdemucs_6s stem (Task 3 kept α = 0), the isolated take and the isolated
      take through the separator beside them; the electric option on EGDB's stems reported, not
      judged. Both scripts log the look; if the 2-hour job limit stops the scoring, it restarts from
      its caches and the extra log line says so. *Predictions,* from validation: **EGDB** (direct
      input, as IDMT): the stem against the mix between −0.04 and +0.04 at each level; separation alone
      costs the isolated take at least 0.05 (from 0.4547). **GuitarSet** (microphone, as player 00): the
      stem beats the mix by +0.10 to +0.20 at each level. Nothing is chosen from the test sets; the
      results go in their own table, with ADR 0066's limits beside them.
      *Run log:* the mixes were built (log line 37) and the scoring started (line 38); the machine
      lost power two minutes in, at EGDB clip 3. The seven stems written by then were checked whole
      (frames equal to their sources). The stem cache now writes a file whole or not at all
      (a test simulates the cut). The scoring restarted from its caches (line 39).
      *Two facts found on the way:* separation here runs at about real time (PyTorch uses the M4's 4
      performance cores, as on validation), so the look takes about nine hours across restarts; and
      demucs's `apply_model` applies one **random** time shift by default (`shifts=1`, unseeded), so
      a stem is not bit-reproducible from scratch — every figure is for the stems as cached.
      *Moving separation to the Mac's GPU (Ege, 2026-10-07), checked first.* Re-measured, the CPU
      would need about 15 more hours: 13 h of test audio to separate (EGDB 5.4 h, GuitarSet 7.6 h)
      at about real time. The GPU ("mps") separated an 18 s test mix in 6.4 s while the CPU run was
      still going. The CPU run was stopped with 44 of 1,620 test stems cached; they are kept.
      **Check before switching, on validation only:** EGSet12's 24 mixes and its 12 isolated takes,
      separated on the GPU into a separate cache — `scripts/evaluate_mixes.py --sets egset12
      --device mps --stems-dir cache/mps --add-mix 0` — and scored as Task 2 scored the CPU's
      stems: stem 0.2935 at 0 dB, 0.2209 at −6 dB, isolated through the separator 0.4046.
      *Rule:* the look continues on the GPU (`--device mps`) if all three are within 0.02 of the
      CPU's figures — demucs's own random shift already moves stems from run to run — and on the
      CPU otherwise. Both the check and the look run only when Ege says so.
      *The check (Ege said run, 2026-10-07; 3 min 14 s):* GPU against CPU — stem at 0 dB 0.2887
      against 0.2935 (−0.0048), stem at −6 dB 0.2413 against 0.2209 (**+0.0204**), isolated through
      the separator 0.3838 against 0.4046 (**−0.0208**). Two of three miss the 0.02 bound, by 0.0004
      and 0.0008, in opposite directions. **By the rule, the look stays on the CPU.** How far a second
      CPU run would move these figures (demucs's random shift) was never measured, so whether 0.02 was
      wider or narrower than a rerun's own noise is not known. Reported to Ege.
      *Ege's call: measure that noise first.* Rule fixed before it runs (this commit): the same
      EGSet12 check on the CPU into a fresh cache (`--device cpu --stems-dir cache/cpu2`). If this
      rerun moves any of the three figures by at least 0.02 from the first CPU run (0.2935, 0.2209,
      0.4046), the GPU's differences are the size of a rerun's own noise and the look runs on the
      GPU; otherwise on the CPU.
      *Result (5 min 25 s):* the CPU rerun gives 0.2921 (−0.0014), 0.2385 (+0.0176), 0.4060
      (+0.0014). Largest 0.0176 < 0.02: **the look runs on the CPU.** The −6 dB difference is
      rerun noise (the rerun moved nearly as far as the GPU); on the isolated takes the GPU moved
      0.0208 against a rerun's 0.0014, so it may separate a little differently. The rerun also ran
      at about three times real time, against about real time in the test run's first minutes —
      when the battery was at 4%; the look's pace is measured, not assumed.

## Task 5: The app

- [x] A "full song — separate the guitar first" option on the page and `--full-song` on the CLI;
      the duration limit reconsidered for songs (ADR 0058's five minutes). *Built (e7425c6): the
      server loads the separator when Demucs is installed and greys the option out with the reason
      otherwise; checked in a browser on a validation mix — page and CLI agree on all 116 notes. The
      five-minute limit is kept for now: separation runs at about real time on this CPU.*
- [ ] Gate to Ege.

## Not in this plan

- MoisesDB (after Ege registers); distorted guitar; separating two guitars from each other.
