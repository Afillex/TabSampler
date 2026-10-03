# ADR 0029: E3 times a hand move from the last fretted group

Status: proposed (2026-10-03) — pre-registration; accepted with its result

Changes how ADR 0011's transition rule measures time, as ADR 0025 left it. The window, the
12 frets-per-second limit and the chord rules are unchanged.

## Context

ADR 0011 times a transition "between consecutive groups". Since ADR 0018 and ADR 0025 the
*hand* is carried across an all-open group — such a group needs no hand, so the hand keeps
its place — but the *clock* is not. A move made after an open group is timed from the open
group, not from the last group that used the hand. So `fret 2 → open → fret 10` at 0.25 s
steps fails at 16 frets/s, while `fret 2 → fret 10` in the same 0.5 s passes: an open
string, which frees the hand, made the move harder. The review of the hand-window branch
found it (devlog 2026-10-02), and ADR 0025's acceptance figure of 0.9795 was measured with
it.

## Decision

**A hand move is timed from the onset of the last group with a fretted note** — the last
moment the hand was in place — to the onset of the group that needs the move. All-open
groups in between do not restart the clock. Onsets, not offsets, as everywhere else in E3:
DadaGP gives note durations, not the moment a finger lifts.

`eval.playability.hand_moves` is the one place this lives; `playability_rate` and
`scripts/validate_playability.py` both use it, so the metric and its validation cannot drift.

## Rerun, pre-registered by committing this ADR before it

Same data as ADR 0025's gate: all 19,995 cleared songs of DadaGP's shipped training list,
16,732,524 transitions. Single variable: the timing of moves made after an all-open group.

**Hypothesis: human tab's transition pass rate rises from 0.9795 but stays below 0.99**,
because the fix only touches moves that cross an all-open group, and most of the remaining
failures are between fretted notes. The chord-shape rate (0.9986) cannot change.

## Alternatives considered

- **Time from the previous group, as before.** Rejected: it penalises an open string for
  freeing the hand.
- **Leave moves across open groups out of E3.** Rejected: it hides them instead of judging
  them.

## Consequences

**Easier.** A transition can only gain time, never lose it, so the rate can only rise; the
speed-limit experiment (ADR 0031) starts from a timing that is no longer an artefact.

**Harder.** GuitarSet's transition figures (0.9944 / 0.9834) were measured with the old
timing. They are refreshed at the next GuitarSet run, not by a second look now.
