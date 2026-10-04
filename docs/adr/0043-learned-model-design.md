# ADR 0043: The learned model — today's decoder with a learned cost in context (D12)

Status: accepted (2026-10-04)

Decides spec D12 for Phase 2 task C4, as the Phase 2 plan sketched it: "a bidirectional encoder
over the note sequence, with the CRF kept as the output layer". Framework and compute are ADR
0041's; the weights stay unpublished (ADR 0042).

## Context

The decoder is already a linear-chain CRF: a lattice of nodes — a chord shape and the hand it is
played with (ADRs 0018, 0030) — scored by a cost linear in twelve weights, decoded by Viterbi and
forward-backward. Its costs see one shape and the hand; they cannot see the music around it.
On player 00, 62% of the default's errors came in runs of four or more (`analyse_errors.py`): whole
passages placed in another position, which is a question of context.

The spec's Phase 2 gate is a comparison in oracle mode: the Viterbi decoder, a learned model on
its own, and the learned model's scores decoded by Viterbi.

## Decision

**The model.** A node's energy is today's cost plus a learned term:

    E_t(node) = w · Φ(node)  +  g(h_t, ψ(node))

- `w · Φ` is the cost model exactly as `fingering/fit.py` expresses it, its weights starting at
  the default's and trained with the rest; movement stays a transition cost on the hand window.
- `h_t` is a bidirectional encoder's state at note group `t`, read from the groups alone: the
  pitches (multi-hot over MIDI 28–100), the number of notes, and the log time since the previous
  group. No string or fret information goes in; that is what is predicted.
- `ψ(node)` describes a candidate: for each string, whether it is played, open, and at which fret
  (scaled), plus the hand's index fret and the shape's span.
- `g` is a small MLP on `[h_t; ψ]`, giving one number per node.
- **The CRF is the output layer**, so one note per string, the legal shapes and the hand stay
  structural (ADR 0010), and every output is pitch-valid by construction — the spec's
  "constrained decoding".

**With `g` switched off the model is today's default**, and a test pins its log-likelihood to the
oracle-verified decoder's.

**The encoder.** Two bidirectional GRU layers of 64 units. A recurrent encoder decodes passages
of any length without a position limit, and trains on a CPU in minutes.

**Training.** The fitter's objective — negative log-likelihood of the human path through the
lattice — on the clean parts of DadaGP's artist-split training songs (ADR 0024), in chunks of
128 groups with the lattice built per chunk. Adam, learning rate 1e-3, batch 32 chunks, seed 0.
Training stops when the negative log-likelihood on DadaGP's clean validation parts has not fallen
for three epochs; the best epoch is kept. Nothing is tuned on GuitarSet.

**The comparison**, in oracle mode, on player 00 now and on the test players once, at task C6:

| | decoding |
|---|---|
| (a) today's decoder | cost model, Viterbi |
| (b) the learned term alone | per group, the node with the lowest `g` — no transitions |
| (c) the learned model | full energy, Viterbi |

On player 00, (c) is judged against (a) by ADR 0039's rule. Being unpublished, it cannot become
the CLI's default (ADR 0042); a win is a research result, reported as one.

## Alternatives considered

- **A transformer encoder**, as the spec words it. It needs a position scheme and a maximum
  length; a recurrent encoder needs neither, and the comparison the spec asks for is between a
  learned model and Viterbi, which this keeps. A transformer can be tried after, as its own
  experiment.
- **A sequence-to-sequence model generating tab tokens.** It would have to learn what the lattice
  already guarantees, and constrain its output to stay pitch-valid.
- **Replacing the cost model instead of adding to it.** Starting from today's decoder means the
  learned term only has to learn what the cost model misses, and the model can be no worse at
  initialisation than the default.

## Consequences

**Easier.** Every guarantee of the decoder carries over, and the baseline is one switch away.

**Harder.** A torch reimplementation of the lattice's scoring that must agree with the numpy one,
and a model whose parameters, unlike the cost weights, cannot be read.
