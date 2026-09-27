# ADR 0012: M1 ships with hand-set cost weights; tuning is gated on legal data

Status: accepted (2026-09-27)

Decides spec D7, and records a gap in the spec's Phase 1 plan.

## Context

Spec 6 (Phase 1) says "tune the cost weights on a validation split (grid or random
search)". Spec 3.3 says hyperparameters are tuned on a validation split drawn from the
**training** sources. At Phase 1 there are none:

- GuitarSet is test-only (ADR 0003);
- GOAT, GAPS and Guitar-TECHS arrive at Phase 4;
- DadaGP is access-by-request and may not have arrived.

So Phase 1 as written has **no legal tuning data**. The tempting resolution — carve a
small validation slice out of GuitarSet "just for the weights" — is exactly what ADR
0003 exists to prevent, and it would invalidate every number reported afterwards.

## Decision

1. **M1 is reported with hand-set weights.** This is a legitimate result and a
   legitimate ground-zero number, not a placeholder. It is stated plainly in the README
   and in the results row's `notes`.
2. **Tuning is gated on DadaGP.** Cost-weight tuning needs only *symbolic* tabs — no
   audio — so DadaGP alone unblocks both this and ADR 0011's rule validation. Request it
   at the start of Phase 1.
3. If DadaGP has not arrived, tuning moves to Phase 2 rather than borrowing test data.
4. Every tuning entry point calls `assert_tuning_allowed(split)` first, which refuses
   `Split.TEST`.

Weights live in `configs/phase1_baseline.yaml` with their rationale, so "hand-set" means
"chosen deliberately and written down", not "hard-coded and forgotten".

## Alternatives considered

- **Tune on a GuitarSet validation slice.** Rejected; see ADR 0003. It is the one thing
  that would make the project's numbers meaningless.
- **Tune on my own recordings.** Rejected as a *tuning* signal: they have no reference
  fingering, so there is nothing to fit against. They remain useful for E3, which needs
  no reference, and as a qualitative check.
- **Learn the weights as a CRF.** Deferred: spec D7 suggests it as a Phase 2 bridge
  experiment, and it needs the same labelled symbolic data.

## Consequences

**Easier.** M1 is reachable now, and honestly. No test-set contamination.

**Harder.** The M1 number is weaker than it could be, and we will not know by how much
until tuning runs. That gap must not be quietly attributed to the method when the real
cause is untuned weights.

**Revisit:** when DadaGP access arrives, or at Phase 2.
