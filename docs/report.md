# Tab Sampler — Project Report

This report follows the project from its first line of evaluation code to full-song transcription:
what I built in each phase, what I measured, and what I learnt — including what did not work.
Every figure comes from `experiments/results.csv` or an Architecture Decision Record (ADR) in
`docs/adr/`, and each section names the ADRs behind it.

## 1. Summary

Tab Sampler turns a guitar recording into tablature: the string and fret of every note. The
pipeline transcribes notes with Basic Pitch, generates every playable (string, fret) for them, scores
chord shapes and hand movements with a cost model, and decodes the most likely fingering with
Viterbi, with forward-backward posteriors for uncertainty. A local web app and a CLI expose it, with
MusicXML and Guitar Pro export.

On EGDB — 240 clips of clean electric guitar that no part of the system was trained or tuned on —
the end-to-end exact tab F1 (E2) rose over the project from **0.4251** to **0.4547** (better
transcriber thresholds) to **0.4992** (a string classifier trained on openly licensed data). Given
the true notes, string choice reaches **0.7229**. On GuitarSet's acoustic test players the decoder
reaches 0.6819 with true notes and 0.4553 end to end.

The project's strongest feature is how it was measured: test sets that were never used for a choice,
every look at them logged, every experiment's hypothesis and decision rule committed before it ran,
and failed predictions written down beside the successful ones.

## 2. Goals and method

**The task.** Given audio of a guitar, output tablature: notes with their onset, pitch, string and
fret. Rhythm (bars and note values) is out of scope; notes are placed in time.

**The metrics** (spec §3):

| | What it measures |
|---|---|
| E1 | note F1: did the transcriber hear the right notes (onset within 50 ms, pitch)? |
| **E2** | **exact tab F1**: onset, pitch *and* string must match — the headline |
| E3 | playability: share of chord shapes and hand moves a hand can make |
| E4 | pitch validity: every (string, fret) sounds the pitch claimed — must be 1.0 |
| E5 | calibration error of the note posteriors |
| E7 | runtime |

Every result is reported in two modes. **Oracle** feeds the true notes to the fingering stage and
measures string choice alone; **end to end** feeds the transcriber's notes and is what a user gets.
The gap between them is what transcription costs.

**The rules I worked by.**

- The test set is sacred. GuitarSet (ADR 0003) and later EGDB (ADR 0050) are test-only; the split
  is enforced in code and every look is logged with its reason.
- One variable per experiment, with the hypothesis, the metric, the split and the decision rule
  committed before the run.
- Decisions that are hard to reverse go into ADRs; an accepted ADR is never edited, only superseded.
- No number is written from memory or estimated; a metric that was not computed stays blank.
- No comparison with other papers' figures unless their method is rerun under this protocol.

## 3. Phase 0 — the evaluation harness first

I built the evaluation harness before any model: dataset loading, the split guard, the metrics and
the results log. Basic Pitch was the transcriber from the start. It cannot be installed beside a
Python 3.13 stack on Apple silicon, so it runs in its own Python 3.11 environment behind a
subprocess and a content-addressed note cache (ADR 0001). Phase 0's measure: Basic Pitch's note F1
on GuitarSet, **0.7437** over all 360 tracks.

## 4. Phase 1 and Milestone M1 — a probabilistic decoder

I represented the decoder's states as whole chord shapes, pruned by the hand's span (ADR 0010), and
scored them with a hand-set cost model: hand movement, stretch, high positions and a small reward
for open strings (ADR 0012). Viterbi decodes the best fingering; forward-backward gives each note's
posterior and its ranked alternatives. A brute-force decoder that enumerates every path is the
specification the fast decoder is tested against (ADR 0014). Output is ASCII tab and JSON
(ADR 0013).

**M1, on GuitarSet:** oracle E2 **0.6366**, end to end **0.4231**. The decoder is about 3% of the
runtime, so I ruled out porting it to C++ (ADR 0015). A 3-minute file took 3.70 s.

## 5. Phase 1.5 — fixing what M1 revealed

Two defects, each fixed and measured on its own (ADRs 0018, 0019):

- An all-open chord let the hand jump for free (fret 2 → open → fret 10 cost nothing). Carrying the
  hand position across open shapes raised oracle E2 to **0.6599** and end to end to **0.4318**.
- E3 counted fretted notes, so every barre chord was scored unplayable; counting fingers instead
  raised the playable share of chord shapes from 0.9832 to 0.9970.

Charging movement properly exposed that the hand-set weights were out of balance. I deliberately did
not reweight them by eye: without validation data that would have been tuning on examples.

## 6. Phase 2 — learning fingering from human tab

With DadaGP — a large corpus of Guitar Pro files — as training data (ADR 0021), I set out to learn
fingering from human tablature, with an artist-disjoint validation split (ADR 0024).

- **The playability rules, validated:** 99.86% of 16.8 million human chord shapes pass the chord
  rules; the original speed rule failed (88%), so it was rebuilt from human tab: 48 frets per second,
  passed by 99.88% of unseen artists' transitions (ADRs 0022, 0031, 0036).
- **The hand as a four-fret window** (ADR 0025), which a wide chord can stretch (ADR 0030): oracle E2
  on GuitarSet 0.6599 → 0.6878.
- **Fitted cost weights** won on GuitarSet (+3.9 points) but lost on clean validation parts, so they
  did not become the default (ADRs 0023, 0027). Separate decoders for clean and distorted guitar
  followed, each with its own temperature (ADRs 0032, 0033).
- **DadaGP misled three times.** Its clean parts predicted GuitarSet badly — once in the wrong
  direction. I moved one GuitarSet player (player 00) to validation (ADR 0037) and re-decided the
  default there (ADR 0038).
- **What won on real playing:** a cost on open strings played with the hand up the neck — one fitted
  weight — raised player 00's oracle E2 from 0.8065 to 0.8273 (+0.021, interval +0.007 to +0.036)
  and was adopted (ADR 0039).
- **The learned model** — the decoder plus a learned cost from a bidirectional GRU (ADR 0043) — fit
  DadaGP better but placed player 00's notes worse (0.8273 → 0.7970), so it was not preferred (ADR 0044).

**The test evaluation** (ADR 0045): the default reached oracle E2 **0.6819** on players 01–05, and
the open-string cost held on the test players (+0.026). **Milestone M2 — oracle E2 ≥ 0.760 — was
missed by 7.8 points.** The comparison the spec asked for exists: the decoder 0.6819, the learned
term alone 0.4514, the learned model decoded by Viterbi 0.6856. Weights trained on DadaGP stay
unpublished, as its terms require (ADR 0042).

*What I learnt:* symbolic tab from the internet is a poor guide to how real players finger real
recordings, and one held-out player is a noisy judge in both directions — demanding a clear win
kept the default from moving on noise.

## 7. Phase 3 — listening to the audio (a negative result)

I added audio evidence: a small CNN over a constant-Q window around each note predicts its string,
and the decoder adds a cost per string (ADRs 0046, 0047). Trained on SynthTab's rendered guitar, it
picked the right string for 47% of player 00's ambiguous notes (chance 28%, the decoder 82%). In the
decoder it **lowered** the test players' oracle E2 from 0.6819 to **0.6073** (ADR 0048), and the
weight stayed at zero.

*What I learnt:* rendered guitar does not teach a classifier what real recordings sound like.

## 8. Phase 4 — clean electric guitar and a second test set

I made clean electric guitar the primary target (ADR 0049) and adopted **EGDB** as a second test set
(ADR 0050). Its string labels turned out to be in the MIDI channel, not the track names (ADR 0053).
For training I used **Guitar-TECHS** — real electric guitar with pickup-tracked strings — after
checking its labels against its audio: glitches under 60 ms dropped, and each take's label delay
measured from its own audio, since some takes ran 55–76 ms early (ADRs 0051, 0052).

The classifier, fine-tuned on Guitar-TECHS, raised **EGDB's oracle E2 from 0.6762 to 0.7200** — the
project's first gain from audio — while on acoustic GuitarSet it did not help (ADR 0054). The
decoder alone does about as well on electric as on acoustic guitar (0.6762 against 0.6819).

## 9. Phase 5 — a better transcriber

Reading Basic Pitch's paper showed it was **trained on about 90% of GuitarSet** (ADR 0055), so its
GuitarSet figures are partly in-sample. From then on I chose transcriber settings on Guitar-TECHS's
player 3 and tested them on EGDB.

- **Stricter thresholds** (onset 0.7, frame 0.4, minimum note 58 ms) raised **EGDB's end-to-end E2
  from 0.4251 to 0.4547**; on electric direct input Basic Pitch writes many notes that were not
  played, and the thresholds removed most of them (ADR 0057).
- **Fine-tuning Basic Pitch** on Guitar-TECHS failed twice (ADR 0056): once its onsets fired almost
  everywhere, once almost nowhere.

## 10. The application

I built a local web app and kept one pipeline function shared with the CLI, so the two always agree
(ADR 0058): uncertain notes marked by shape, alternatives on hover, confidences as a ranking, uploads
checked before the transcriber runs, localhost only. Exports to MusicXML and Guitar Pro 5 put notes on
a 1/128-note grid at a declared 120 BPM, with "rhythm is not transcribed" written inside each file
(ADRs 0017, 0059).

Testing it on my own recording exposed a problem the benchmarks never had: my guitar was **0.43 of a
semitone sharp**, and Basic Pitch, which assumes A440, dropped or misplaced most notes. Automatic
correction did more harm than good on validation — it cost 0.013 in tune and corrected the wrong way
near half a semitone (ADR 0060) — so the app now **warns** when a recording is a quarter of a semitone
or more off. On a downloaded recording in tune, the result was a promising draft.

## 11. An optimisation round

Before full songs, I tried three improvements (ADR 0061):

- **Speed:** the tuning check now runs while the transcriber works — a 3-minute file went from 9.05 s
  to 5.17 s.
- **Uncertainty marks:** on player 3, marked notes were only 1.42 times as likely to be wrong as
  unmarked ones, short of the 1.5 I required; the threshold stayed.
- **Removing extra notes:** dropping "octave ghosts" gained +0.025 on player 3 but **−0.001 on EGDB**,
  so it was not adopted.

*What I learnt:* one validation player — 12 takes — had chosen too much; a gain on it did not carry
to the test set.

## 12. Electric guitar with openly licensed data

String choice is the largest loss end to end, and the one measured lever on it was Phase 4's
classifier — whose weights started from SynthTab and could not be published. I searched for real
electric guitar with string labels and checked each candidate's licence and labels (ADR 0062):

- **Training:** Guitar-TECHS's three players and **EGFxSet**'s 690 single notes (both CC BY 4.0).
  EGFxSet numbers its strings from the high e, and IDMT from the low E — I measured the open
  strings' pitch to be sure.
- **Validation:** **EGSet12** (real solo pieces) and **IDMT-SMT-Guitar**'s licks (evaluation only, by
  its licence), whose labels run about 35 ms ahead of the sound and are corrected per take.

The retrained classifier was badly overconfident on unfamiliar audio (calibrated temperature 6.80).
Calibrated on IDMT and judged on EGSet12, it raised oracle E2 0.7250 → 0.7926 (interval +0.013 to
+0.144). **On EGDB: oracle 0.6762 → 0.7229 and end to end 0.4547 → 0.4992**, both inside the
predictions committed beforehand (ADR 0063). It became the app's opt-in electric option, and its
weights are published under CC BY 4.0 (ADR 0064).

## 13. Phase 6 — full songs (in progress)

No public dataset I could find pairs full-band mixes with string labels, so I build them: a labelled
guitar take over **BabySlakh** backing tracks with their guitars removed, at two levels, with
separate backing songs for validation and test (ADR 0066). **Demucs**' `htdemucs_6s` separates the
guitar (ADR 0065).

On validation (pooled over EGSet12, IDMT and GuitarSet's player 00):

| | mix | separated guitar | guitar alone |
|---|---|---|---|
| E2 at 0 dB | 0.1902 | **0.3180** | 0.4666 |
| E2 at −6 dB | 0.1252 | **0.2572** | 0.4666 |

Separation raises the score on mixes by about two-thirds at 0 dB and doubles it at −6 dB. But on clean direct-input electric guitar (IDMT), the
separator alone cost 0.12 even with no backing, and it did not beat the mix. Adding a share of the
mix back to the stem made every case worse. Two practical findings: Demucs applies a random time
shift by default, so a stem is reproducible only from its cache; and on validation the Mac's GPU did
not match the CPU closely enough by my own pre-set rule, so the test evaluation runs on the CPU. The
app has a "full song" option. The test evaluation is running.

## 14. What I learnt

- **Build the evaluation first.** Every later decision rested on a harness I trusted.
- **Validation data must look like the test data.** DadaGP, SynthTab and a single held-out player
  each misled me; real recordings from other players, studios and guitars did not.
- **Pre-registration changes behaviour.** Writing the rule down before the run stopped me from
  adopting several changes that looked good on the data that chose them.
- **Negative results are results.** Phase 3, fine-tuning, the learned model, tuning correction and
  the note filters all failed, and knowing why shaped what worked.
- **Real use finds what benchmarks miss.** A detuned guitar broke the app on day one; no test set
  contained one.
- **Licences are part of the design.** Choosing openly licensed training data is what made a
  publishable model possible.

## 15. What comes next

- Finish Phase 6's test evaluation and report full-song results beside the isolated ones.
- Real full-song recordings (MoisesDB) to check the synthetic mixes.
- A stronger transcriber, the largest remaining loss end to end.
- Rhythm, and playing techniques (bends, slides, hammer-ons).
