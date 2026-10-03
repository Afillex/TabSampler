# ADR 0031: E3's speed limit, set from human tab and checked on unseen artists

Status: proposed (2026-10-03) — pre-registration; accepted with its result

Would supersede ADR 0011's **12 frets per second** and ADR 0022's rejection of raising it.
Approved in principle by Ege on 2026-10-03, after the open-string timing was fixed first
(ADR 0029).

## Context

E3's transition rule has two parts: where the hand is (ADR 0025's window, stretched by
ADR 0030) and how fast it may move (ADR 0011: 12 frets per second, a hand-set number). With
the window, the timing fix and the stretch in place, human tab still fails the rule on
1.77% of its moves (0.9823), against ADR 0022's bar of 99%. What remains is mostly small,
quick shifts — moves a guitarist makes routinely and the rule calls impossible.

ADR 0022 rejected raising the limit because the rule was "wrong in kind": it measured a
finger reaching as the hand moving, so any limit loose enough to pass human tab would also
pass impossible jumps. The window has since removed that error. What is left is a number
that was never measured, and human tab is the evidence for it.

## Decision, fixed before the run

**Method.** On every cleared song of the artist-split **training** side (ADR 0024), find the
smallest limit under which **0.9986** of transitions pass, and round it **up** to a whole
number of frets per second. 0.9986 is the rate at which ADR 0011's chord rules hold on human
tab (ADR 0022), so both halves of E3 are then equally strict about human playing. Moves
with no time between groups fail at any limit and are counted against it.

**Check.** On every cleared song of the artist-split **validation** side — artists the limit
has never seen — at least **0.9976** of transitions pass at that limit (0.1 point of
tolerance).

**If the check holds:** `PlayabilityRules.max_frets_per_second` becomes the rounded limit,
and E3's transition rate may be quoted as a measure: *the share of moves no faster than
99.86% of human moves*. That lifts ADR 0022's caveat, with this wording. **If it fails:**
12 frets per second and the caveat stay, and the result is recorded.

This changes what the rule claims — from "physically impossible" to "faster than human
players move" — and the claim is checked on artists the number was not taken from. The
limit is not chosen to make any gate pass: the target rate is the chord rules' measured
one, fixed here before the speeds are seen.

## Alternatives considered

- **Keep 12 frets per second regardless.** Rejected: a rule that fails 1.8% of human moves
  cannot be quoted as playability. It stays only if the check fails.
- **A limit per distance** (a 1-fret shift and a 10-fret jump judged differently). A better
  model, perhaps, but a second variable; it can follow if this one fails.
- **Take the number from studies of guitarists' hand speed.** Not attempted: a published
  figure would measure a different thing from this rule's window shift between onsets,
  and could not be checked against it here.

## Consequences

**Easier.** If it holds, every E3 number can be quoted without a caveat.

**Harder.** The limit depends on DadaGP's tab, which is crowd-sourced and quantised to a
rhythmic grid: very fast moves in it may be notation rather than playing. The check on
unseen artists guards against fitting one community's habits, not against that.
