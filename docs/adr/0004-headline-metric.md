# ADR 0004: The headline metric is end-to-end Exact Tab F1 on GuitarSet

Status: accepted (2026-09-27). The target this ADR deferred is proposed in
[ADR 0016](0016-headline-target.md), which completes it; the metric here is unchanged.

Decides spec D10.

## Context

Spec 1 says "high accuracy has no meaning until it is stated as a number on a named
benchmark". A project with five metrics and no headline has no way to say whether a change
was an improvement, and invites picking whichever metric moved.

## Decision

The headline number is **E2, Exact Tab F1, end-to-end, on GuitarSet, under our protocol**
(ADR 0003). A hit requires onset within 50 ms, pitch, **and** string to match.

It is never reported alone. Every report carries, alongside it:

- **E1** note F1 (did we hear the right notes?),
- **E3** playability rate (is the output physically playable?),
- and the **oracle-mode** E2 beside the end-to-end one, because the gap between them is
  what the transcriber costs (spec 3.2).

E4 (pitch validity) is a correctness invariant, not a score: it must be exactly 1.0, and
anything less is a bug to fix before reporting anything.

**The target is deliberately not set yet.** Setting a target before a baseline exists means
inventing a number. After M1 it is expressed as "baseline + X points" (Task 20, Step 7).

Spec 3.4's reference points — TART reporting 69.2% Exact Tab F1 given reference notes and
56.0% end to end on GuitarSet — are **not** targets and **not** comparable. They were
measured under TART's protocol, not ours. They must not appear in our results table. If
cited at all, it is in separate prose that says plainly they were measured differently.

## Alternatives considered

- **Note F1 plus playability as the headline.** Rejected: both can look good while string
  assignment is wrong, which is the actual problem this project is about.
- **A weighted mix of metrics.** Rejected: the weights would be arbitrary, and a composite
  hides which component moved.

## Consequences

**Easier.** One number to improve, with the others as guardrails against gaming it. A
change that raises E2 while E3 collapses is visibly not an improvement.

**Harder.** E2 is strict, so early numbers will look discouraging, and E2 depends on the
transcriber end-to-end — which is why oracle mode is always reported next to it.

**Revisit:** after M1, to set the target only. Changing the metric itself needs a
superseding ADR.
