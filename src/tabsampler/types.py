"""Data contracts and stage interfaces (spec 2.1).

These types are the architecture: every stage talks to its neighbours only through
them, which is what makes a model swappable without touching anything else. Changing
one requires an ADR.

Two deviations from the literal spec text, both recorded in ADR 0007:

- ``NoteEvent.bend`` and ``TabNote.alternatives`` are tuples, not lists. A list inside
  a ``frozen=True`` dataclass is a footgun: the dataclass refuses attribute assignment
  while the list stays mutable, and the instance becomes unhashable.
- ``NoteGroup``, ``ChordState``, ``Context`` and ``CostWeights`` are defined here. The
  spec's Protocols reference the first three but never define them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

MIDI_MIN = 0
MIDI_MAX = 127

# Standard tuning, low E to high e, as MIDI note numbers.
STANDARD_OPEN_PITCHES = (40, 45, 50, 55, 59, 64)


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """One note as heard: when it sounded, what pitch, and how sure the transcriber was."""

    onset: float  # seconds
    offset: float  # seconds
    pitch: int  # MIDI note number
    confidence: float  # 0..1, from the transcriber
    bend: tuple[float, ...] | None = None  # pitch contour; see ADR 0007 and the note below

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise ValueError(f"offset {self.offset} precedes onset {self.onset}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        if not MIDI_MIN <= self.pitch <= MIDI_MAX:
            raise ValueError(f"pitch {self.pitch} outside MIDI range [{MIDI_MIN}, {MIDI_MAX}]")
        if self.bend is not None and not isinstance(self.bend, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard: see ADR 0007
            raise TypeError("bend must be a tuple, not a list -- see ADR 0007")

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass(frozen=True, slots=True)
class Tuning:
    """How the instrument is tuned, and how far up the neck we may go.

    ``capo`` shifts every string, and fret numbers are measured **relative to the
    capo**: with a capo at fret 2, ``fret=0`` sounds two semitones above the open
    string and ``fret=3`` sounds five.
    """

    open_pitches: tuple[int, ...] = STANDARD_OPEN_PITCHES
    n_frets: int = 22
    capo: int = 0

    STANDARD: ClassVar[Tuning]

    def __post_init__(self) -> None:
        if not self.open_pitches:
            raise ValueError("a tuning needs at least one string")
        if self.n_frets < 1:
            raise ValueError(f"n_frets {self.n_frets} must be positive")
        if not 0 <= self.capo <= self.n_frets:
            raise ValueError(f"capo {self.capo} outside [0, {self.n_frets}]")
        if not isinstance(self.open_pitches, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard: see ADR 0007
            raise TypeError("open_pitches must be a tuple -- see ADR 0007")

    @property
    def n_strings(self) -> int:
        return len(self.open_pitches)

    @property
    def max_fret(self) -> int:
        """Highest fret reachable, counted from the capo."""
        return self.n_frets - self.capo

    def pitch_at(self, string: int, fret: int) -> int:
        """The pitch a (string, fret) pair sounds.

        The single source of pitch arithmetic in the project: candidate generation and
        the E4 pitch-validity metric both go through here, so they cannot disagree.
        """
        if not 0 <= string < self.n_strings:
            raise IndexError(f"string {string} outside [0, {self.n_strings - 1}]")
        if not 0 <= fret <= self.max_fret:
            raise ValueError(f"fret {fret} outside [0, {self.max_fret}] (capo {self.capo})")
        return self.open_pitches[string] + self.capo + fret

    def fret_for(self, string: int, pitch: int) -> int | None:
        """The fret on ``string`` that sounds ``pitch``, or None if there isn't one."""
        if not 0 <= string < self.n_strings:
            raise IndexError(f"string {string} outside [0, {self.n_strings - 1}]")
        fret = pitch - self.open_pitches[string] - self.capo
        return fret if 0 <= fret <= self.max_fret else None


Tuning.STANDARD = Tuning()


@dataclass(frozen=True, slots=True)
class Position:
    """A place on the fretboard. ``string`` 0 is the low E; ``fret`` is relative to the capo."""

    string: int
    fret: int

    def __post_init__(self) -> None:
        if self.string < 0:
            raise ValueError(f"string {self.string} must be non-negative")
        if self.fret < 0:
            raise ValueError(f"fret {self.fret} must be non-negative")

    @property
    def is_open(self) -> bool:
        return self.fret == 0


@dataclass(frozen=True, slots=True)
class NoteGroup:
    """Notes whose onsets fall inside one grouping window: a chord, or a single note.

    ``notes`` must already be sorted by ``(pitch, onset)``. Use :meth:`of` to build one
    from unordered input, so that a hand-built group cannot silently differ from one
    produced by the grouping stage.
    """

    notes: tuple[NoteEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.notes, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard: see ADR 0007
            raise TypeError("notes must be a tuple -- see ADR 0007")
        if not self.notes:
            raise ValueError("a note group cannot be empty")
        keys = [(n.pitch, n.onset) for n in self.notes]
        if keys != sorted(keys):
            raise ValueError("notes must be sorted by (pitch, onset); use NoteGroup.of()")

    @classmethod
    def of(cls, notes: list[NoteEvent] | tuple[NoteEvent, ...]) -> NoteGroup:
        return cls(notes=tuple(sorted(notes, key=lambda n: (n.pitch, n.onset))))

    @property
    def onset(self) -> float:
        """The group's time: the earliest onset among its members."""
        return min(n.onset for n in self.notes)

    def __len__(self) -> int:
        return len(self.notes)


@dataclass(frozen=True, slots=True)
class ChordState:
    """One assignment of every note in a group to a distinct string.

    ``positions`` is parallel to the group's ``notes``: same length, same order.
    """

    positions: tuple[Position, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.positions, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard: see ADR 0007
            raise TypeError("positions must be a tuple -- see ADR 0007")
        if not self.positions:
            raise ValueError("a chord state cannot be empty")
        strings = [p.string for p in self.positions]
        if len(set(strings)) != len(strings):
            raise ValueError(f"two notes assigned to one string: {strings}")

    @property
    def fretted_frets(self) -> tuple[int, ...]:
        """Frets that need a finger, in ascending order. Open strings need none."""
        return tuple(sorted(p.fret for p in self.positions if not p.is_open))

    @property
    def span(self) -> int:
        """Fret distance the hand must cover. Zero when every string is open."""
        fretted = self.fretted_frets
        return 0 if not fretted else fretted[-1] - fretted[0]

    @property
    def hand_position(self) -> int | None:
        """Lowest fretted fret (spec 2.2), or None when the shape is all open strings."""
        fretted = self.fretted_frets
        return fretted[0] if fretted else None

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class TabNote:
    """A note placed on the fretboard, with how confident the decoder is about it."""

    note: NoteEvent
    position: Position
    posterior: float  # from forward-backward
    alternatives: tuple[tuple[Position, float], ...] = ()  # other candidates, descending

    def __post_init__(self) -> None:
        if not 0.0 <= self.posterior <= 1.0:
            raise ValueError(f"posterior {self.posterior} outside [0, 1]")
        if not isinstance(self.alternatives, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard: see ADR 0007
            raise TypeError("alternatives must be a tuple -- see ADR 0007")


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Spec 2.2's lambdas, plus the forward-backward temperature.

    Defaults live in ``configs/phase1_baseline.yaml``; these are the fallbacks.
    """

    move: float = 1.0
    span: float = 1.0
    high: float = 0.1
    #: Subtracted per open string: open strings need no finger and are idiomatic.
    open_reward: float = 0.25
    acoustic: float = 0.0  # unused until Phase 3; in the contract so it stays stable
    temperature: float = 1.0
    #: Cost of each note on a string, open or fretted, low E first (ADR 0034).
    string_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    #: Cost of each fretted note at frets 1-4, and at fret 12 or above; frets 5-11 are the
    #: reference (ADR 0034).
    low_region: float = 0.0
    high_region: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature {self.temperature} must be positive")
        if len(self.string_bias) != 6:
            raise ValueError(
                f"string_bias needs six entries, one per string, got {self.string_bias}"
            )


@dataclass(frozen=True, slots=True)
class Context:
    """Everything the scorer needs that is not the notes themselves."""

    tuning: Tuning
    max_span: int = 4
    weights: CostWeights = field(default_factory=CostWeights)

    def __post_init__(self) -> None:
        if self.max_span < 0:
            raise ValueError(f"max_span {self.max_span} must be non-negative")


def assert_state_matches_group(group: NoteGroup, state: ChordState) -> None:
    """Check that a state assigns exactly one position per note in the group."""
    if len(state) != len(group):
        raise ValueError(f"state has {len(state)} positions but the group has {len(group)} notes")


@runtime_checkable
class Transcriber(Protocol):
    """audio -> note events (spec 2, stage 1)."""

    def transcribe(self, audio: NDArray[np.float32], sr: int) -> list[NoteEvent]: ...


#: The frets a fretting hand covers: (index-finger fret, highest fret it reaches). At rest
#: that is four frets above the index (ADR 0025); a chord wider than that stretches it to
#: the chord's own highest fret while a shape needs the stretch (ADR 0030).
Hand = tuple[int, int]


@runtime_checkable
class FingeringScorer(Protocol):
    """How comfortable a shape is, and how hard it is to move between shapes (spec 2.2).

    ``transition_cost_from`` is what the decoder calls, and it takes the *hand* -- the frets
    it covers (ADR 0030) -- rather than a previous shape. The reason is ADR 0018: an all-open
    shape has no hand position of its own, so a scorer that only ever sees the previous
    shape cannot charge the movement across one. ``transition_cost`` is kept as the
    stateless convenience the spec names and must agree with it on any shape that has a
    hand position.
    """

    def emission_cost(self, group: NoteGroup, state: ChordState, ctx: Context) -> float: ...

    def transition_cost(self, prev: ChordState, curr: ChordState) -> float: ...

    def transition_cost_from(self, previous_hand: Hand | None, curr: ChordState) -> float: ...


@runtime_checkable
class Decoder(Protocol):
    """note groups -> placed notes with posteriors (spec 2, stage 4)."""

    def decode(self, groups: list[NoteGroup], scorer: FingeringScorer) -> list[TabNote]: ...
