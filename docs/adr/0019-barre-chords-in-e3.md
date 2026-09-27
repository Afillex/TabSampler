# ADR 0019: Count fingers, not fretted notes, in E3

Status: accepted (2026-09-27)

**Supersedes ADR 0011 in part**: only its fourth group rule. ADR 0011's span thresholds,
its speed limit, its separate reporting of groups and transitions, and above all **its
owed validation against real human tab remain in force and remain unmet.**

## Context

ADR 0011 says a group passes if "at most **4 fretted notes** need distinct fingers (a barre
is not modelled in v1)". Counting notes makes every full barre chord unplayable: an E-shape
barre at fret 5 sounds six notes, five of them fretted, so it fails a four-note limit — yet
it is one of the first chords any guitarist learns. GuitarSet is full of them, so E3's group
rate was pessimistic by an unknown amount, and "unknown" is the problem: a metric whose
failures are dominated by a modelling gap cannot tell you whether your output is playable.

This is not the threshold question ADR 0011 defers to DadaGP. How many fingers a shape
needs is a fact about hands, not a number to fit. It can be corrected now; the span
thresholds still cannot.

## Decision

Replace the rule with a finger count. `PlayabilityRules.max_fretted_notes` becomes
**`max_fingers: int = 4`**, and **`allow_barre: bool = True`** is added.

```
fingers_needed(shape):
    fretted = frets of the non-open positions, with multiplicity
    if none:              0
    if not allow_barre:   len(fretted)          # ADR 0011's original rule
    else:                 1 + (len(fretted) - how many sit at the lowest fretted fret)
```

One finger covers **every string at the lowest fretted fret**, however many that is,
**unless an open string sits inside the run of strings that finger would cover** — the
barre lies flat, so it would press that string too. Every note above the lowest fret costs
a finger of its own, **even when two of them share a fret.** That last clause is the whole
point of confining the discount: the other fingers are already committed to their own frets
and cannot also lie flat across strings.

A string *inside* the barre's run that is fretted **higher** is not a conflict: the barre
presses it, but it sounds at its own higher fret, which is exactly how an E-shape barre
chord works. Only an open string inside the run rules the barre out.

Worked examples, all at `max_fingers = 4`:

| shape (frets) | fingers | verdict |
|---|---|---|
| E-shape barre at 5: 5, 7, 7, 6, 5, 5 | 1 + 3 = 4 | playable — was refused before |
| 5, 6, 7, 8, 9 | 1 + 4 = 5 | refused: a hand has four fingers |
| 5, 9, 9, 7, 6 | 1 + 4 = 5 | refused: fret 9 cannot be barred under a fret-5 hold |
| 5, 5, 9, 10, 11, 14 | 1 + 4 = 5 | refused, and the 9-fret span refuses it too |
| 5, 5, 5, 5 | 1 | playable |
| low E at 5, A **open**, D at 5, then 6, 7, 8 | 2 + 3 = 5 | refused: the barre cannot cross an open A |

**The rename is deliberately config-visible.** `config.py` names the key, so a config still
saying `max_fretted_notes` now fails with "unknown key(s) ['max_fretted_notes']". That is
the loader working: the rule that key set no longer exists, and silently keeping the default
would make the run unreproducible.

The rule can only ever **relax** the count — `1 + len - barred <= len` whenever anything is
fretted — so E3's group rate cannot fall because of this change. A test pins that.

## Measured effect

All 360 GuitarSet tracks, same commit for everything else, this as the single variable:

| | oracle | end-to-end |
|---|---|---|
| E3 playable groups, counting fretted notes (ADR 0011) | 0.9832 | 0.9744 |
| E3 playable groups, counting fingers | **0.9970** | **0.9896** |
| | +0.0138 | +0.0152 |

The first version of the finger rule, without the open-string condition, scored 0.9971 and
0.9897 — one ten-thousandth higher in each mode, because it accepted a handful of shapes
needing five fingers. Both runs are in `experiments/results.csv`; the row for the corrected
rule is the one to quote.

E1, E2, E4, E5 and the E3 transition rate are byte-identical across every one of these
runs, which is what "one variable per experiment" is supposed to look like. So M1's E3 group
rate was pessimistic by about **1.4 points in oracle mode and 1.5 end to end** — real, and smaller
than the barre problem's prominence suggested, because most GuitarSet groups are one or two
notes rather than six-string chords.

## Alternatives considered

- **Count distinct fretted frets** (what the plan proposed). Rejected: it lets a barre
  happen anywhere. The shape 5, 9, 9, 7, 6 has four distinct frets and a 4-fret span, so it
  would pass — but two notes at fret 9 cannot be barred while fret 5 is held below them.
  The plan's own test required that shape to fail, so the plan was inconsistent with itself
  and this ADR resolves it toward the test.
- **Model fingers individually** (assign finger 1–4 to each note and check reachability).
  Rejected for v1: far more machinery, needs per-finger stretch data we do not have, and
  it would be a second unvalidated model inside a metric.
- **Leave ADR 0011 alone until DadaGP arrives.** Rejected. The thresholds need data; this
  does not. Waiting would keep E3 pessimistic by an amount we could have measured.
- **Drop the finger limit entirely** since span already bounds the shape. Rejected: Review
  Focus 5 is exactly this failure. Span 4 with five distinct frets inside it is five
  fingers, and a hand has four.

## Consequences

**Easier.** E3 now measures what it claims. Full barre chords stop dominating the failure
list, so the remaining failures are informative.

**Harder.** Nothing in the code, but E3's *number* is no longer comparable to M1's
published one without saying which rule produced it. The README carries both.

**What we accept.** The finger model is still a model, and it is now wrong in the strict
direction only. Because just one finger may barre, and only at the lowest fret, it refuses
shapes a good player manages with a second barre. The clearest example is the D6 voicing
`x 5 7 7 7 7`: index at fret 5, ring finger flat across four strings at fret 7. This counts
`1 + 4 = 5` fingers and is refused, though it is a common shape. Modelling a second barre
would mean assigning fingers individually, which is the alternative rejected above. It is
closer than counting notes, it errs toward calling playable output unplayable rather than
the reverse, and it is honest about being a rule rather than a measurement.

> **Correction.** As first written, this ADR granted the barre discount to every note at
> the lowest fret regardless of which strings they were on, which accepted shapes needing
> five fingers — for example low E at 5 with the A string open and the D string at 5, plus
> 6, 7 and 8 above. The open-string condition above closes that. The effect on E3 is in
> the devlog.

**Revisit** with ADR 0011's owed validation, on the same data, at the same time. The
prediction to test: with barres modelled, the share of real human tab that passes should be
close to 100%; if it still is not, the span thresholds are what is wrong.
