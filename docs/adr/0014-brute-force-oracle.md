# ADR 0014: The brute-force oracle is the decoder's specification

Status: accepted (2026-09-27)

Covers a gap in spec 2.2: it defines the cost model and the recurrences, but not how we
establish that the implementation computes them.

## Context

The decoder is the part of this project that fails silently. A wrong span-pruning bound, a
sign error in a log-potential, a backtracking bug that returns the right cost with the
wrong path — none of these crash. They produce tablature that looks entirely plausible and
is wrong, and every metric downstream inherits the error without flagging it.

Ordinary unit tests are weak here. A test that asserts `viterbi` returns a particular cost
is written by reasoning about the same recurrence the implementation encodes, so both can
be wrong in the same way. What is needed is a second implementation that shares no
reasoning with the first.

The dynamic program exists precisely because enumerating every path is intractable. But on
*short* inputs — five groups, a handful of states each — enumeration is trivial, and a
direct sum over all paths is obviously correct in a way the DP never will be.

## Decision

**`tests/decode/brute_force.py` is the specification, and the decoder must agree with it.**

- It enumerates every path through the lattice with `itertools.product` and scores each one
  from first principles: emission costs summed, movement charged against where the hand
  actually is. `brute_force_marginals` computes posteriors as a plain sum over path weights
  in probability space — which underflows on anything long, and is exactly what makes it a
  fair check on the log-space implementation.
- Hypothesis property tests compare the two over randomly generated short sequences. These
  are marked `@pytest.mark.oracle` and run as `make oracle`.
- **It is kept obviously correct and slow.** Nothing in that file is clever on purpose. If
  a function in it cannot be read and seen to be right, it is not doing its job.
- **A change to the oracle deserves more scrutiny than a change to the decoder**, because
  a wrong oracle silently blesses a wrong decoder. When a change to the cost model requires
  changing both — as the hand-position carry in ADR 0018 did — the oracle's version is
  written from the spec text, not copied from the implementation or imported from it.
- **If the two disagree, the decoder is wrong.** That is the default, and overturning it
  requires showing the oracle wrong against the spec, not against the decoder.

## Alternatives considered

- **Hand-written expected values in unit tests.** Rejected: they encode the author's
  reasoning about the recurrence, which is the thing under test. They also rot — every
  reweighting invalidates every magic number, so in practice they get regenerated from the
  implementation, at which point they assert nothing.
- **A second optimised implementation** (numpy-vectorised, or a different DP formulation).
  Rejected: fast implementations share the optimisations that cause the bugs, and a
  disagreement between two clever implementations does not say which is wrong.
- **Testing only end-to-end metrics on real audio.** Rejected: a decoder bug moves E2 by an
  amount indistinguishable from a modelling change, so it would be diagnosed as the cost
  model being bad rather than the code being broken.
- **No oracle; rely on review.** Rejected. Reading a log-space forward-backward pass and
  confirming it correct is exactly the kind of review that feels conclusive and is not.

## Consequences

**Easier.** Any change to the cost model can be made with confidence: if the oracle tests
still pass, the decoder still minimises what it claims to minimise. This is what made
ADR 0018's lattice augmentation safe to attempt — the state space changed shape entirely,
and the oracle confirmed the costs were unchanged.

**Harder.** Every cost-model change now has to be made twice, in the decoder and in the
oracle, and the second one has to be written independently to be worth anything. Copying
the implementation into the oracle would pass every test and check nothing.

**What we accept.** The oracle only covers short inputs — five or six groups. Bugs that
appear only at scale (numerical underflow, pruning interactions over hundreds of groups)
need their own tests, and there are separate ones for exactly that.

**Revisit if** the decoder stops being a dynamic program over an explicit lattice — a
learned model with a different inference procedure would need its own independent check,
not this one.
