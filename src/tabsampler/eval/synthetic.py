"""A synthetic round-trip diagnostic for the fingering model.

Pure: no I/O, no global state. The random generator is passed in, never created here.

Sample a random *playable* fingering, read off the pitches it would sound, feed only
those pitches to the decoder, and ask whether it recovers the fingering. That answers a
question the brute-force oracle tests cannot: they establish that Viterbi minimises the
cost correctly, and this asks whether minimising *this* cost recovers a fingering a player
would use.

⚠️ **This is a diagnostic and a regression test. It is not a tuning signal, and a number
from it must never set a cost weight or the temperature.**

The fingerings it samples are drawn from *our own* notion of a plausible shape -- the same
ADR 0011 playability rules and the same bounded hand walk that the cost model is trying to
express. Fitting weights so that the decoder agrees with them would tune the model to
agree with its own prior and then report the agreement as accuracy. The number would rise
and the tab would not get better.

The real tuning signal is human tab, which is what the DadaGP/ProgGP request is for
(ADR 0012). Until it arrives, treat a round-trip figure as "the decoder can invert its own
generator this often", and quote it with that sentence attached -- in the devlog, in any
report, and in the ``tabsampler diagnose`` output, which prints it next to the number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabsampler.decode.forward_backward import decode
from tabsampler.eval.playability import PlayabilityRules, group_is_playable
from tabsampler.fingering.candidates import candidates
from tabsampler.types import (
    ChordState,
    Context,
    FingeringScorer,
    NoteEvent,
    NoteGroup,
    Position,
    Tuning,
)

#: One sampled group and the fingering that generated it.
SampledStep = tuple[NoteGroup, ChordState]

#: Seconds between consecutive sampled groups. Wide enough that the E3 speed limit never
#: binds, so the sampler tests shape choice rather than tempo.
STEP_SECONDS = 0.5

#: How far the hand may drift between consecutive groups, in frets.
MAX_HAND_STEP = 4

#: Chance that a string in a shape is played open rather than fretted.
OPEN_STRING_PROBABILITY = 0.25

#: Attempts to find a playable shape at one hand position before falling back to a
#: single note there. Keeps sampling total rather than occasionally unbounded.
MAX_SHAPE_ATTEMPTS = 20


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """How often the decoder recovered the fingering that generated the pitches.

    Read :mod:`tabsampler.eval.synthetic`'s docstring before quoting ``accuracy``
    anywhere. It is not a measure of transcription quality.
    """

    n_notes: int  # total notes across all sampled paths
    n_recovered: int  # notes whose recovered Position equals the generating one
    n_paths: int
    n_single_candidate: int  # notes with one legal position -- free wins, reported apart

    @property
    def accuracy(self) -> float:
        """Micro-averaged over notes, never over paths.

        1.0 for no notes: the same empty-input convention the corpus metrics use.
        """
        return self.n_recovered / self.n_notes if self.n_notes else 1.0


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    """Typed boundary around ``Generator.integers``. ``high`` is exclusive."""
    return int(rng.integers(low, high))


def _shuffled(rng: np.random.Generator, count: int) -> list[int]:
    """Typed boundary around ``Generator.permutation``: 0..count-1 in random order."""
    drawn: list[int] = [int(value) for value in rng.permutation(count)]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return drawn


def _draw_hand(rng: np.random.Generator, previous: int, highest: int) -> int:
    """Walk the hand by a bounded random step, staying on the neck."""
    step = _randint(rng, -MAX_HAND_STEP, MAX_HAND_STEP + 1)
    return max(0, min(highest, previous + step))


def _draw_shape(
    rng: np.random.Generator, hand: int, tuning: Tuning, rules: PlayabilityRules
) -> tuple[Position, ...] | None:
    """One candidate shape at ``hand``, or None if this draw is not playable.

    Frets are drawn inside the span the rules allow at this hand position, so most draws
    are playable and the rejection below is a guard rather than the mechanism.
    """
    span = rules.max_span_at(hand)
    n_notes = _randint(rng, 1, min(rules.max_fingers, tuning.n_strings) + 1)
    strings = sorted(_shuffled(rng, tuning.n_strings)[:n_notes])

    positions: list[Position] = []
    for string in strings:
        if float(rng.random()) < OPEN_STRING_PROBABILITY:
            positions.append(Position(string=string, fret=0))
            continue
        fret = hand + _randint(rng, 0, span + 1)
        if fret > tuning.max_fret:
            return None
        positions.append(Position(string=string, fret=fret))

    shape = tuple(positions)
    if group_is_playable(shape, rules) is not None:
        return None
    # A unison makes recovery ambiguous for a reason that says nothing about the cost
    # model, so it is excluded from the generator rather than scored as a miss.
    pitches = [tuning.pitch_at(p.string, p.fret) for p in shape]
    if len(set(pitches)) != len(pitches):
        return None
    return shape


def _step(onset: float, shape: Sequence[Position], tuning: Tuning) -> SampledStep:
    """Pair a shape with the note group it sounds, keeping the two parallel.

    ``NoteGroup`` requires its notes sorted by ``(pitch, onset)`` and ``ChordState``'s
    positions are parallel to them, so both are built from one sort of the same pairs.
    Sorting them independently is how a sampler silently misaligns note and position.
    """
    paired = sorted(
        ((tuning.pitch_at(p.string, p.fret), p) for p in shape), key=lambda item: item[0]
    )
    notes = tuple(
        NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)
        for pitch, _ in paired
    )
    return NoteGroup(notes=notes), ChordState(positions=tuple(p for _, p in paired))


def sample_playable_path(
    rng: np.random.Generator,
    n_groups: int,
    tuning: Tuning,
    rules: PlayabilityRules,
) -> list[SampledStep]:
    """A random sequence of playable shapes, with the notes each one sounds.

    Every shape passes :func:`~tabsampler.eval.playability.group_is_playable`, so the
    generator never asks the decoder to recover something the rules call impossible.

    Args:
        rng: Passed in, never created here, so the caller owns reproducibility.
        n_groups: How many shapes. 0 gives an empty path.
        tuning: The instrument.
        rules: Playability rules the sampled shapes must satisfy.

    Raises:
        ValueError: if ``n_groups`` is negative.
    """
    if n_groups < 0:
        raise ValueError(f"n_groups must be non-negative, got {n_groups}")

    path: list[SampledStep] = []
    hand = _randint(rng, 0, max(1, tuning.max_fret - rules.max_span_high))
    for index in range(n_groups):
        hand = _draw_hand(rng, hand, tuning.max_fret - rules.max_span_high)
        shape: tuple[Position, ...] | None = None
        for _ in range(MAX_SHAPE_ATTEMPTS):
            shape = _draw_shape(rng, hand, tuning, rules)
            if shape is not None:
                break
        if shape is None:
            # A single note at the hand position is always playable, so the sampler
            # terminates rather than looping on an awkward draw.
            shape = (Position(string=0, fret=hand),)
        path.append(_step(index * STEP_SECONDS, shape, tuning))
    return path


def round_trip_accuracy(
    paths: Sequence[Sequence[SampledStep]],
    scorer: FingeringScorer,
    ctx: Context,
) -> RoundTripReport:
    """Decode each sampled path's pitches and count the notes placed back where they came from.

    Per-note, micro-averaged, for the same reason the corpus metrics are: a mean of
    per-path rates lets a one-note path outweigh a twenty-note one.

    **The number this returns must not tune anything.** See the module docstring.

    Raises:
        UnfingerableGroupError: if a sampled group has no legal state under ``ctx``.
            That means ``ctx.max_span`` is tighter than the span ``rules`` allowed the
            sampler, which is a caller error rather than a result.
    """
    n_notes = n_recovered = n_single = 0
    for path in paths:
        if not path:
            continue
        groups = [group for group, _ in path]
        expected = [position for _, state in path for position in state.positions]
        tab = decode(groups, scorer, ctx)
        for note, position in zip(tab, expected, strict=True):
            n_notes += 1
            if note.position == position:
                n_recovered += 1
            if len(candidates(note.note.pitch, ctx.tuning)) == 1:
                n_single += 1
    return RoundTripReport(
        n_notes=n_notes,
        n_recovered=n_recovered,
        n_paths=len(paths),
        n_single_candidate=n_single,
    )
