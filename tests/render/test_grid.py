"""The export grid (ADR 0059): the arithmetic MusicXML and Guitar Pro share."""

from __future__ import annotations

from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from tabsampler.render.grid import UNITS_PER_BAR, UNITS_PER_SECOND, GridEvent, to_grid
from tabsampler.types import NoteEvent, Position, TabNote


def note(onset: float, offset: float, string: int = 0, fret: int = 0) -> TabNote:
    pitch = (40, 45, 50, 55, 59, 64)[string] + fret
    return TabNote(
        note=NoteEvent(onset=onset, offset=offset, pitch=pitch, confidence=0.9),
        position=Position(string, fret),
        posterior=0.9,
    )


def chords(events: list[GridEvent]) -> list[tuple[int, tuple[Position, ...]]]:
    """(start, positions) of each struck chord: note pieces that are not tie continuations."""
    return [(e.start, e.positions) for e in events if e.positions and not e.tie_stop]


def test_an_empty_tab_is_one_bar_of_rest() -> None:
    events = to_grid([])
    assert sum(e.length for e in events) == UNITS_PER_BAR
    assert all(not e.positions for e in events)


def test_a_quarter_note_at_the_start() -> None:
    # 120 BPM: a quarter note is 0.5 s, 32 units.
    events = to_grid([note(0.0, 0.5, string=1, fret=2)])
    assert events[0] == GridEvent(start=0, length=32, positions=(Position(1, 2),))
    assert sum(e.length for e in events) == UNITS_PER_BAR


def test_a_gap_before_the_first_note_is_a_rest() -> None:
    events = to_grid([note(0.25, 0.75)])
    assert events[0] == GridEvent(start=0, length=16, positions=())
    assert chords(events) == [(16, (Position(0, 0),))]


def test_notes_within_the_window_form_one_chord_at_the_first_onset() -> None:
    events = to_grid([note(0.0, 0.5, 0, 3), note(0.02, 0.5, 2, 2), note(0.5, 1.0, 1, 0)])
    assert chords(events) == [(0, (Position(0, 3), Position(2, 2))), (32, (Position(1, 0),))]


def test_a_held_note_is_cut_where_the_next_starts() -> None:
    events = to_grid([note(0.0, 2.0, 0, 0), note(0.5, 1.0, 1, 0)])
    first = [e for e in events if e.positions == (Position(0, 0),)]
    assert sum(e.length for e in first) == 32


def test_a_note_across_a_bar_line_is_split_and_tied() -> None:
    # 1.75 s to 2.25 s straddles the bar line at 2.0 s (128 units).
    events = [e for e in to_grid([note(1.75, 2.25)]) if e.positions]
    assert [(e.start, e.length, e.tie_start, e.tie_stop) for e in events] == [
        (112, 16, True, False),
        (128, 16, False, True),
    ]


def test_an_odd_length_is_tied_plain_values_largest_first() -> None:
    # 7 units: 4 + 2 + 1, no dots.
    events = [e for e in to_grid([note(0.0, 7 / UNITS_PER_SECOND)]) if e.positions]
    assert [e.length for e in events] == [4, 2, 1]
    assert [e.tie_start for e in events] == [True, True, False]
    assert [e.tie_stop for e in events] == [False, True, True]


def test_a_very_short_note_still_gets_one_unit() -> None:
    events = [e for e in to_grid([note(0.0, 0.001)]) if e.positions]
    assert [e.length for e in events] == [1]


tabs = st.lists(
    st.tuples(
        st.floats(0.0, 20.0, allow_nan=False),
        st.floats(0.001, 3.0, allow_nan=False),
        st.integers(0, 5),
        st.integers(0, 12),
    ),
    max_size=40,
).map(lambda rows: [note(on, on + dur, s, f) for on, dur, s, f in rows])


@settings(max_examples=200, deadline=None)
@given(tabs)
def test_pieces_tile_whole_bars_without_gaps_or_overlaps(tab: list[TabNote]) -> None:
    events = to_grid(tab)
    cursor = 0
    for e in events:
        assert e.start == cursor
        assert e.length in (1, 2, 4, 8, 16, 32, 64, 128)
        assert e.start // UNITS_PER_BAR == (e.start + e.length - 1) // UNITS_PER_BAR  # in one bar
        cursor += e.length
    assert cursor % UNITS_PER_BAR == 0 and cursor > 0


@settings(max_examples=200, deadline=None)
@given(tabs)
def test_every_note_is_struck_once_within_half_a_unit_of_its_chord(tab: list[TabNote]) -> None:
    struck = chords(to_grid(tab))
    assert sum(len(p) for _, p in struck) == len(tab)
    starts = [s for s, _ in struck]
    assert starts == sorted(set(starts))
    for t in tab:
        assert any(
            t.position in positions and start <= round(t.note.onset * UNITS_PER_SECOND)
            for start, positions in struck
        )


@settings(max_examples=200, deadline=None)
@given(tabs)
def test_ties_join_pieces_of_the_same_chord(tab: list[TabNote]) -> None:
    events = to_grid(tab)
    for a, b in pairwise(events):
        assert a.tie_start == b.tie_stop
        if a.tie_start:
            assert a.positions == b.positions and a.positions
