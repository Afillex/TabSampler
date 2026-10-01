# ADR 0011: Concrete playability rules for E3

Status: accepted (2026-09-27); the fourth group rule below is superseded by
[ADR 0019](0019-barre-chords-in-e3.md), which counts fingers instead of fretted notes. The
validation owed below was run in [ADR 0022](0022-playability-rules-validated.md): the chord
rules hold on human tab, the speed rule does not.

Decides spec D9.

## Context

E3 asks what share of our output is physically playable, and unlike E2 it needs **no
reference tab** — which makes it the only metric we can run on arbitrary audio,
including my own recordings. But "playable" has to be numbers before it can be measured.

## Decision

A **group** passes if:

- no two notes share a string (structurally guaranteed by `ChordState`, checked anyway);
- the span of fretted notes is at most **4 frets below fret 12**, and at most **5 at
  fret 12 and above** — frets get physically narrower up the neck;
- open strings are free: they need no finger and are excluded from the span;
- at most **4 fretted notes** need distinct fingers (a barre is not modelled in v1).

A **transition** between consecutive groups passes if the hand moves no faster than
**12 frets per second**, measured between hand positions (the lowest fretted fret).
A group with no fretted notes has no hand position, and carries the previous one
forward rather than resetting to fret 0.

Group and transition rates are reported **separately**. A single conflated number hides
which rule is failing.

The rules are data passed into the metric, not constants inside it, so they can be
changed without editing metric code.

## Validation owed

Spec D9 asks for the check that matters: **the share of real, human-made tab that passes
these rules should be close to 100%.** If it is not, the rules are wrong, not the tab.
That check needs a corpus of real tabs — DadaGP — and is therefore **pending**, not done.
Until it runs, these thresholds are informed guesses and E3 should be read as
"passes our current rules", not "playable".

## Alternatives considered

- **A single span limit everywhere.** Rejected: a 5-fret stretch at the first fret is
  much harder than at the twelfth, and a flat limit either forbids reasonable high-neck
  shapes or permits impossible low ones.
- **Counting open strings in the span.** Rejected: it would make an open-E-plus-fret-12
  shape look like a 12-fret stretch when it needs one finger.
- **A fixed maximum fret jump instead of a speed limit.** Rejected: the same jump is
  trivial between whole notes and impossible between sixteenths. Time has to be in it.

## Consequences

**Easier.** E3 runs with no ground truth, so it works on my own playing and on any
audio, and it catches the failure mode E2 cannot see: output that is pitch-correct and
physically impossible.

**Harder.** The thresholds are unvalidated until DadaGP arrives, so E3's absolute value
should not be quoted as "playability" without that caveat. Barres are unmodelled, so
some genuinely playable chords will be marked unplayable.

**Revisit:** as soon as a real tab corpus is available. That check may well move these
numbers, which is the point of writing them down now.
