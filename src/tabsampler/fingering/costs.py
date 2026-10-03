"""Hand-set fingering cost model (spec 2.2, ADR 0012).

Pure: no I/O, no global state.

Total cost of a path, from spec 2.2::

    C = sum(emission) + lambda_move * sum(movement)

with the emission term itself carrying the span, neck-height and open-string parts::

    emission = lambda_span * span
             + lambda_high * (mean fretted fret / 12)
             - open_reward * (number of open strings)
             + lambda_ac * acoustic          # zero until Phase 3

Weights are hand-set for M1 and live in ``configs/phase1_baseline.yaml``. ADR 0012
explains why they are not tuned: Phase 1 has no legal validation data, because GuitarSet
is test-only and the training corpora arrive at Phase 4.

**Movement across an all-open shape (ADR 0018).** ``transition_cost(prev, curr)`` is
stateless, so it cannot charge anything when either shape is all open -- an all-open
shape has no hand position of its own. M1 shipped with that, and a passage going
fret 2 -> open chord -> fret 10 was charged no movement though the hand travelled 8
frets. :meth:`HandSetScorer.transition_cost_from` takes the *hand position* instead of
the previous shape and closes it; the decoder carries that position across all-open
states (see :func:`tabsampler.fingering.states.carry_hand`). ``transition_cost`` is kept
as the stateless view the spec names, defined in terms of the hand-aware one so the two
cannot drift apart.

**Still open from spec 2.2.** A still-ringing note does not reserve its string against
the next group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabsampler.fingering.states import carry_hand, shift_window
from tabsampler.types import (
    ChordState,
    Context,
    CostWeights,
    Hand,
    NoteGroup,
    assert_state_matches_group,
)

#: Frets per octave: used to normalise neck height so ``lambda_high`` is scaled in
#: "octaves up the neck" rather than in raw fret numbers.
FRETS_PER_OCTAVE = 12.0


def mean_fretted_fret(state: ChordState) -> float:
    """Average fret of the notes that need a finger. 0.0 when everything is open."""
    fretted = state.fretted_frets
    return (sum(fretted) / len(fretted)) if fretted else 0.0


def count_open(state: ChordState) -> int:
    return sum(1 for p in state.positions if p.is_open)


@dataclass(frozen=True, slots=True)
class HandSetScorer:
    """A :class:`~tabsampler.types.FingeringScorer` with hand-set weights."""

    weights: CostWeights = field(default_factory=CostWeights)

    def emission_cost(self, group: NoteGroup, state: ChordState, ctx: Context) -> float:
        """How uncomfortable one shape is, on its own.

        Raises:
            ValueError: if ``state`` does not assign exactly one position per note.
        """
        assert_state_matches_group(group, state)
        w = self.weights
        return (
            w.span * state.span
            + w.high * (mean_fretted_fret(state) / FRETS_PER_OCTAVE)
            - w.open_reward * count_open(state)
            # The acoustic term (spec 2.2's lambda_ac) enters at Phase 3, when a
            # per-note -log P(string | audio) exists. Until then it is identically 0,
            # so the weight is inert by construction rather than by omission.
            + w.acoustic * 0.0
        )

    def transition_cost_from(self, previous_hand: Hand | None, curr: ChordState) -> float:
        """Movement cost given where the hand *was*, not which shape it was in.

        This is what the decoder calls. Passing the hand rather than a shape is what lets
        an all-open shape carry the hand forward instead of erasing it (ADR 0018). The hand
        covers a 4-fret window, stretched over a wider chord while one needs it, and moves
        only when ``curr`` has a fretted note outside what it covers (ADR 0025, ADR 0030),
        so a finger reaching within one position costs nothing. ``previous_hand`` is None
        before the first fretted shape, and an all-open ``curr`` needs no move, so both
        cost nothing.
        """
        return self.weights.move * shift_window(previous_hand, curr.fretted_frets)[1]

    def transition_cost(self, prev: ChordState, curr: ChordState) -> float:
        """How far the hand must move between two consecutive shapes.

        The stateless view spec 2.1 names, defined in terms of the hand-aware one. It
        charges nothing when either shape is all open, because on its own it cannot know
        where the hand was; the decoder uses :meth:`transition_cost_from` for that.
        """
        return self.transition_cost_from(carry_hand(None, prev), curr)

    def carry(self, previous_hand: Hand | None, curr: ChordState) -> Hand | None:
        """The hand position after playing ``curr``. An all-open shape inherits.

        Delegates to :func:`tabsampler.fingering.states.carry_hand`, which is where the
        decoder's lattice gets it. The carry is spec 2.2 geometry, not a weight, so it
        is deliberately not something a scorer may redefine -- hence it is not on the
        :class:`~tabsampler.types.FingeringScorer` Protocol.
        """
        return carry_hand(previous_hand, curr)
