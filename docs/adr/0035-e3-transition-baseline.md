# ADR 0035: A baseline for ADR 0016's transition guardrail under the new E3 rule

Status: accepted (2026-10-03) — **the transition baseline is 0.9996 oracle, 0.9969 end to end**

Amends the baseline of one guardrail in [ADR 0016](0016-headline-target.md), as the review of the
hand-window branch recommended. ADR 0016's target and every other guardrail stand.

## Context

ADR 0016 guards M2 with "E3 group and transition rates must not fall below the baseline",
and set that baseline from the figures after Phase 1.5 (0.9134 oracle, 0.9164 end to end).
Those were
measured under a transition rule that no longer exists: since then the hand is a window
(ADR 0025) that can stretch (ADR 0030), a move is timed from the last fretted group
(ADR 0029), and the speed limit is 48 frets per second, measured on human tab (ADR 0031).
A number measured under one rule cannot guard a number measured under another, so the
transition guardrail has had no baseline since ADR 0025 — and was reported, not enforced.
The chord-shape rule has not changed, so the group guardrail (0.9970 / 0.9896) stands.

## Decision

**The transition rates of the GuitarSet run pre-registered in `configs/m2_style_eval.yaml`
become the baseline for ADR 0016's transition guardrail**, in each mode, under the rule as
ADRs 0029–0031 leave it. The run's other guardrails are judged against ADR 0016 as it stands.
A later change to the transition rule needs a new baseline the same way.

## Alternatives considered

- **Keep the M1 figures.** Rejected: they measure a different rule.
- **Take the baseline from human tab** (0.9988 on unseen artists). Rejected: ADR 0016's
  guardrails compare our decoder with itself over time, not with human players.

## Consequences

The guardrail is enforceable again from the next GuitarSet run, and E3's transition rate is
quotable as a measure (ADR 0031) as well as guarded.

## Result (2026-10-03)

The run (commit 0dd856d, `configs/m2_style_eval.yaml` with `configs/decoder_clean.yaml`)
measured E3 transition rates of **0.9996 in oracle mode and 0.9969 end to end**. Those are
ADR 0016's transition baseline from now on. Under the same rule, human tab passes 0.9988
on unseen artists.
