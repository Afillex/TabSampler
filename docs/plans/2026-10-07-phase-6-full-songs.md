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

- [ ] Pre-registered on validation: Basic Pitch hears the stem with a share α of the mix added back
      (α in 0, 0.1, 0.25, 0.5; 0 is the stem alone) — a little of the mixture can mask separation
      artefacts. The rule picks α; the electric option is reported beside it.

## Task 4: The test look

- [ ] One pre-registered look at EGDB and one at GuitarSet's test players, full-song mixes, the
      three inputs; a results table separate from the isolated one; an ADR.

## Task 5: The app

- [ ] A "full song — separate the guitar first" option on the page and `--full-song` on the CLI;
      the duration limit reconsidered for songs (ADR 0058's five minutes).
- [ ] Gate to Ege.

## Not in this plan

- MoisesDB (after Ege registers); distorted guitar; separating two guitars from each other.
