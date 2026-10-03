# ADR 0022: ADR 0011's chord rules hold on human tab; its speed rule does not

Status: accepted (2026-10-01); the hand window it proposed is implemented in
[ADR 0025](0025-hand-window.md). Its caveat on E3's transition rate is lifted by
[ADR 0031](0031-e3-speed-limit.md), which measured the speed limit on human tab.

Discharges the validation ADR 0011 owed. Does not change any rule; it changes what E3's
transition rate may be claimed to mean.

## Context

ADR 0011 set E3's playability rules from judgement and owed a check: "the share of real,
human-made tab that passes these rules should be close to 100%. If it is not, the rules are
wrong, not the tab." The expectation recorded in HANDOFF and in the pre-registered
hypothesis was that the **span thresholds** would be the problem — too strict for real
playing.

The check ran on every cleared DadaGP training song (ADR 0021): **19,995 songs, 36,859 guitar
parts, 16,769,383 human chord shapes and 16,732,524 hand movements.** Nothing was fitted;
the rules were evaluated exactly as committed. Crowd-sourced tab contains mistakes, so the
breakdown of failures matters as much as the rate.

## Findings

**The chord rules hold. The hypothesis was wrong.**

| | pass rate |
|---|---|
| chord shapes, rules as committed (barre modelled, ADR 0019) | **0.9986** |
| chord shapes, ADR 0011's original note count, no barre | 0.9794 |

The remaining 0.14% split evenly between a span too wide (12,380) and more than four
fingers (11,633) — the size expected from tab mistakes plus a few genuinely exotic shapes.
Below fret 12, 99.90% of human shapes fit a 4-fret span; at fret 12 and above, 99.97% fit 5.
**The thresholds are right, and the barre rule was necessary**: without it, one human shape
in fifty fails.

**The speed rule does not hold.** Humans "fail" the 12 frets-per-second limit on **11.65%**
of their hand movements as implemented, and on **14.87%** if the hand is carried across an
open shape as ADR 0011's text specifies.

An exploratory follow-up on 2,000 songs says why, and it is not that guitarists move
impossibly fast: **82.6% of the failing moves are three frets or fewer, and 77.9% are between
two single notes.** The rule defines hand position as the lowest fretted fret, so a melody
going from fret 5 to fret 7 in sixteenth notes reads as a 2-fret jump in an eighth of a
second — 16 frets per second — though the hand never moved and a finger simply reached. **The
rule measures finger reach as hand movement.**

## Decision

1. **The chord rules stay as they are.** ADR 0011's span thresholds and ADR 0019's finger
   count are validated on human tab. E3's group rate may be quoted as a playability measure.
2. **E3's transition rate may not be quoted as a playability measure.** It is still computed
   and reported, but always with this caveat: by its own definition, human tab scores 0.88.
   On GuitarSet our decoder's output scores about 0.91 — *above* human tab — which says
   nothing about playability and something about the metric.
3. **The fix is proposed here, not made:** model the hand as a *window* rather than a point —
   a hand position covers roughly four frets, and movement is charged only when a note falls
   outside the current window. That changes E3's transition rule and, because the same
   definition sits in the cost model's movement term, possibly the cost model too. Both are
   decisions for their own ADR, with a hypothesis written first and the human pass rate as
   the acceptance test: a correct rule should pass close to 100% of human moves.

## Alternatives considered

- **Raise the speed limit until human tab passes.** Rejected: it fits a threshold to a
  definition that is wrong in kind. A limit loose enough to pass within-position finger
  reach would pass genuine impossible jumps too.
- **Drop the transition rate from reports.** Rejected: a number with a stated caveat is
  more useful than a gap, and keeping it makes the fix measurable when it lands.
- **Treat the failures as tab errors.** Rejected on the evidence: errors would not
  concentrate on moves of two or three frets between single notes.

## Consequences

**Easier.** E3's group rate can now be stated plainly as playability, which until today it
could not.

**Harder.** ADR 0016's guardrail "E3 transition rate must not fall" now guards a number that
does not mean what its name says. It stays — a change that moves it still needs explaining —
but it is no longer evidence of playability either way.

**The finding reaches past E3.** The cost model charges `move * |Δ lowest fretted fret|`, the
same quantity, so it too treats a reaching finger as a moving hand. Whatever weights are
fitted to that feature (ADR 0023) are fitted to a mis-specified feature, and should be read
with that in mind.

**Revisit** when the hand-window model exists: rerun this validation, and accept it only if
human tab passes the new transition rule at close to 100%.
