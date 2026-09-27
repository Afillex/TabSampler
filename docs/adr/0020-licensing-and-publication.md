# ADR 0020: MIT for the code, public repository, weights decided per corpus

Status: accepted (2026-09-27)

Decides spec D15, **in part**. The code licence and publication are settled here. Weight
release is deliberately left open, because it cannot be answered once for all corpora.

## Context

D15 had no owner until the repository went public, which is one of the three triggers the
plan named (public repo, published weights, published demo — whichever comes first). The
repository is now public and carries a LICENSE file, so the decision is made whether or
not it is written down. Writing it down is the point.

Two questions were tangled together and have different answers:

1. **The code.** Ours, written here, no third-party constraint on it.
2. **Any model weights.** Trained on corpora that are not ours and are not uniformly
   licensed. DadaGP and ProgGP are research-use-by-request; SynthTab, GOAT, GAPS and
   Guitar-TECHS each have their own terms. A model trained on research-only data generally
   cannot be redistributed.

Answering both with one licence would either under-license the code or over-promise on the
weights.

## Decision

- **Code: MIT.** `LICENSE` at the repository root. Chosen over Apache-2.0 because this is a
  small research codebase with no patent surface worth a grant clause, and MIT is the lower
  barrier for anyone who wants to read or reuse a piece of it.
- **The repository is public.** GuitarSet is not redistributed and no dataset is vendored;
  `/data/` and `/cache/` are gitignored and always have been.
- **Weights are decided per training corpus, not once, and nothing is published by
  default.** Before any training run whose weights might be released, check that corpus's
  terms first. If the terms are unclear, the weights stay unpublished — this is not a
  question to resolve optimistically after the fact.
- **The MIT grant does not extend to data or weights**, and the README says so where the
  licence is stated. Someone forking the code gets the code.
- **Results stay reportable regardless.** Publishing measured numbers obtained with a
  research-use corpus is normal scholarly use and is not weight redistribution.

## Alternatives considered

- **Apache-2.0.** Rejected as unnecessary here: the explicit patent grant and NOTICE
  machinery buy nothing for a project with no patentable claim, and add friction for a
  reader who wants to lift one function.
- **Keep it private until Phase 2 finishes.** Rejected: the thing that is useful to show —
  an evaluation harness built before any model, a held-out protocol, and honestly reported
  numbers — already exists, and waiting costs the feedback that publishing might attract.
- **One licence covering code and weights.** Rejected on the facts above. It would promise
  something we are not free to give.
- **No licence at all.** Rejected: without one, default copyright means nobody may legally
  reuse anything, so "open source" would be false.

## Consequences

**Easier.** The repository can be linked in a dataset request, which is where it is now
being used, and anyone may read or reuse the code without asking.

**Harder, and irreversible.** MIT cannot be withdrawn from what is already published. Every
future number in the README is now public too, which raises the cost of quoting one
carelessly — the no-cross-paper-comparison rule (spec §3.4) and "never invent results"
matter more now, not less.

**What stays open.** Weight release. The first real test is Chunk C: check DadaGP's and
ProgGP's terms *before* training anything whose weights might be published, not after.

**Revisit** per corpus, as each dataset arrives.
