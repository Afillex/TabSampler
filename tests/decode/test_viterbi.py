"""Tests for the Viterbi decoder.

The oracle tests are the point: for short inputs, Viterbi's cost must equal the minimum
over every path enumerated by brute force. If those pass the decoder is right; if they
fail, nothing downstream means anything.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings

from tabsampler.decode.viterbi import build_lattice, viterbi
from tabsampler.errors import UnfingerableGroupError
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.fingering.states import HAND_WINDOW, enumerate_states
from tabsampler.types import Context, CostWeights, NoteEvent, NoteGroup, Position, Tuning

from .brute_force import brute_force_min, path_cost
from .strategies import group_sequence

STANDARD = Tuning.STANDARD
CTX = Context(tuning=STANDARD, max_span=4)
SCORER = HandSetScorer()
#: ADR 0039's cost on open strings up the neck, charged by the oracle from the ADR's text.
OPEN_UP_NECK = HandSetScorer(weights=CostWeights(open_up_neck=0.7))

SLOW = settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def n(pitch: int, onset: float = 0.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=onset + 0.4, pitch=pitch, confidence=1.0)


def group(*pitches: int, onset: float = 0.0) -> NoteGroup:
    return NoteGroup.of([n(p, onset) for p in pitches])


# ============================================================ the oracle tests


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=6, max_notes=3))
def test_viterbi_cost_equals_brute_force_minimum(groups: list[NoteGroup]) -> None:
    _, expected = brute_force_min(groups, SCORER, CTX)
    _, got = viterbi(groups, SCORER, CTX)
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=6, max_notes=3))
def test_viterbi_matches_brute_force_with_open_strings_up_the_neck_charged(
    groups: list[NoteGroup],
) -> None:
    _, expected = brute_force_min(groups, OPEN_UP_NECK, CTX)
    path, got = viterbi(groups, OPEN_UP_NECK, CTX)
    assert got == pytest.approx(expected, abs=1e-9)
    assert path_cost(groups, path, OPEN_UP_NECK, CTX) == pytest.approx(got, abs=1e-9)


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=2))
def test_the_returned_path_actually_costs_what_viterbi_says(
    groups: list[NoteGroup],
) -> None:
    # Guards the classic backtracking bug: right cost, wrong path.
    path, cost = viterbi(groups, SCORER, CTX)
    assert path_cost(groups, path, SCORER, CTX) == pytest.approx(cost, abs=1e-9)


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=4, max_notes=3))
def test_the_returned_path_is_a_minimiser_even_when_paths_tie(
    groups: list[NoteGroup],
) -> None:
    # Hand-set integer-ish weights produce ties constantly, so assert the cost is
    # minimal rather than that the path equals brute force's arbitrary choice.
    _, expected = brute_force_min(groups, SCORER, CTX)
    path, _ = viterbi(groups, SCORER, CTX)
    assert path_cost(groups, path, SCORER, CTX) == pytest.approx(expected, abs=1e-9)


# The hand covers 4 frets (ADR 0025). At span 6 some candidate shapes do not fit inside it
# and stretch it from their lowest fret (ADR 0030). The shipped config allows span 5 and best-effort
# decoding relaxes further, so the oracle must see such shapes too.
WIDE = Context(tuning=STANDARD, max_span=6)
WIDE_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)


def has_a_shape_wider_than_the_hand(groups: list[NoteGroup]) -> bool:
    return any(
        state.span > HAND_WINDOW
        for g in groups
        for state in enumerate_states(g, STANDARD, WIDE.max_span)
    )


@pytest.mark.oracle
@WIDE_SETTINGS
@given(groups=group_sequence(max_groups=4, max_notes=2))
def test_viterbi_matches_brute_force_when_shapes_are_wider_than_the_hand(
    groups: list[NoteGroup],
) -> None:
    assume(has_a_shape_wider_than_the_hand(groups))
    _, expected = brute_force_min(groups, SCORER, WIDE)
    path, got = viterbi(groups, SCORER, WIDE)
    assert got == pytest.approx(expected, abs=1e-9)
    assert path_cost(groups, path, SCORER, WIDE) == pytest.approx(got, abs=1e-9)


# ============================================================ structure and edges


def test_one_state_per_group_in_order() -> None:
    groups = [group(45, onset=0.0), group(50, onset=0.5), group(55, onset=1.0)]
    path, _ = viterbi(groups, SCORER, CTX)
    assert len(path) == 3
    for g, state in zip(groups, path, strict=True):
        assert len(state) == len(g)


def test_a_single_group_returns_its_cheapest_state() -> None:
    from tabsampler.fingering.states import enumerate_states

    g = group(64)
    path, cost = viterbi([g], SCORER, CTX)
    best = min(SCORER.emission_cost(g, s, CTX) for s in enumerate_states(g, STANDARD, 4))
    assert cost == pytest.approx(best)
    assert len(path) == 1


def test_an_empty_group_list_gives_an_empty_path_and_zero_cost() -> None:
    path, cost = viterbi([], SCORER, CTX)
    assert path == []
    assert cost == 0.0


def test_a_group_with_no_legal_states_raises_a_typed_error() -> None:
    # Must not return a path of infinite cost that later reads as a valid tab.
    unfingerable = group(39)  # below the open low E
    with pytest.raises(UnfingerableGroupError, match="39"):
        viterbi([unfingerable], SCORER, CTX)


def test_an_unfingerable_group_in_the_middle_is_reported_with_its_index() -> None:
    groups = [group(45), group(39), group(50)]
    with pytest.raises(UnfingerableGroupError, match="index 1"):
        viterbi(groups, SCORER, CTX)


def test_output_pitch_always_equals_input_pitch() -> None:
    # The project's headline invariant, end to end through the decoder.
    groups = [group(45, 52, onset=0.0), group(60, onset=0.5), group(64, 67, onset=1.0)]
    path, _ = viterbi(groups, SCORER, CTX)
    for g, state in zip(groups, path, strict=True):
        for note, pos in zip(g.notes, state.positions, strict=True):
            assert STANDARD.pitch_at(pos.string, pos.fret) == note.pitch


def test_decoding_is_deterministic_across_runs() -> None:
    groups = [group(45, 52), group(60, onset=0.5), group(64, onset=1.0)]
    a = viterbi(groups, SCORER, CTX)
    b = viterbi(groups, SCORER, CTX)
    assert a == b


def test_the_hand_does_not_teleport_across_an_intervening_open_chord() -> None:
    # M1's defect: an all-open shape has no hand position, so fret 22 -> open -> fret 16
    # was charged no movement and the decoder was free to jump. MIDI 80 sits at either
    # (5, 16) or (4, 21); leaving fret 22, the near one is (4, 21) even though (5, 16) is
    # lower on the neck and so has the cheaper emission. Task A1 / ADR 0018.
    groups = [group(86, onset=0.0), group(64, onset=0.5), group(80, onset=1.0)]
    path, _ = viterbi(groups, SCORER, CTX)
    assert path[1].positions[0] == Position(5, 0)  # the open high e, as before
    assert path[2].positions[0] == Position(4, 21)  # not Position(5, 16)


def test_the_first_level_places_each_window_at_its_shapes_lowest_fret() -> None:
    # Nothing precedes group 0, so every fretted shape there puts the index on its own
    # lowest fret, the hand at rest over the four above it, and gets exactly one node
    # (ADR 0025, ADR 0030). Later levels can carry more.
    fretted = group(56)  # no open string sounds MIDI 56
    (level,) = build_lattice([fretted], CTX)
    assert len(level) == len(enumerate_states(fretted, STANDARD, CTX.max_span))
    for node in level:
        frets = node.state.fretted_frets
        assert node.carried_hand == (frets[0], max(frets[0] + 4, frets[-1]))


def test_the_first_level_carries_no_hand_for_an_all_open_state() -> None:
    # Nothing precedes group 0, so an all-open opening shape starts from no hand
    # position -- which keeps the old behaviour for the first group.
    (level,) = build_lattice([group(64)], CTX)
    open_nodes = [node for node in level if node.state.hand_position is None]
    assert len(open_nodes) == 1
    assert open_nodes[0].carried_hand is None


def _state_and_node_counts(groups: list[NoteGroup]) -> tuple[int, int]:
    n_states = sum(len(enumerate_states(g, STANDARD, CTX.max_span)) for g in groups)
    n_nodes = sum(len(level) for level in build_lattice(groups, CTX))
    return n_states, n_nodes


def test_node_count_stays_close_to_state_count() -> None:
    # Under the hand window (ADR 0025) a fretted shape can be played from several windows,
    # so the lattice is larger than under the old point model (1.31x here then). Measured
    # 3.75x on this E-minor pentatonic run; full augmentation by hand position would be 23x.
    pitches = [40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64, 62, 59, 57, 55, 52, 50, 47, 45, 43]
    groups = [group(p, onset=i * 0.25) for i, p in enumerate(pitches)]
    n_states, n_nodes = _state_and_node_counts(groups)
    assert n_nodes <= 6 * n_states, f"{n_nodes} nodes for {n_states} states"


def test_the_pathological_input_is_nothing_but_open_string_pitches() -> None:
    # The carried set grows only through groups that *can* be played all-open, so the
    # worst input is a line of nothing but open-string pitches. Cycling all six measures
    # 4.44x under the hand window (2.92x under the old point model) -- higher than any
    # mixed line, and the number to quote as the worst case. Full augmentation would be
    # max_fret + 1 = 23x. This guards the growth staying structural; it is not a target.
    groups = [group(p, onset=i * 0.5) for i, p in enumerate([40, 45, 50, 55, 59, 64] * 8)]
    n_states, n_nodes = _state_and_node_counts(groups)
    assert n_nodes <= 6 * n_states, f"{n_nodes} nodes for {n_states} states"


def test_the_carried_hand_set_saturates_rather_than_growing_with_the_piece() -> None:
    # Why the ratio above is bounded at all: a level can only carry hand positions that
    # earlier groups actually put the hand in, so the set saturates at the distinct
    # fretted hand positions reachable from the pitches in play. For open-string pitches
    # in standard tuning that is {4,5,9,10,14,15,19} plus None = 8, and it does not grow
    # with the length of the piece. Under full augmentation it would be 23.
    short = [group(p, onset=i * 0.5) for i, p in enumerate([40, 45, 50, 55, 59, 64] * 4)]
    long = [group(p, onset=i * 0.5) for i, p in enumerate([40, 45, 50, 55, 59, 64] * 16)]
    distinct = [
        len({node.carried_hand for level in build_lattice(g, CTX) for node in level})
        for g in (short, long)
    ]
    assert distinct[0] == distinct[1], f"carried set grew with length: {distinct}"
    assert distinct[0] <= 10, f"{distinct[0]} distinct carried hands"


def test_a_shape_with_no_fretted_alternative_never_grows_the_lattice() -> None:
    # An all-open shape whose group has exactly one legal state cannot fan out, because
    # there is no other hand position for the level to carry.
    groups = [group(40, 45, 50, onset=i * 0.5) for i in range(20)]
    n_states, n_nodes = _state_and_node_counts(groups)
    assert n_nodes == n_states


def test_a_scale_stays_in_position_rather_than_jumping_around() -> None:
    # The behavioural point of the whole cost model: an ascending scale should be
    # fingered in one region, not scattered across the neck.
    pitches = [52, 54, 55, 57, 59, 60, 62, 64]
    groups = [group(p, onset=i * 0.25) for i, p in enumerate(pitches)]
    path, _ = viterbi(groups, SCORER, CTX)
    frets = [s.positions[0].fret for s in path if s.positions[0].fret != 0]
    assert max(frets) - min(frets) <= 5


def test_long_sequences_do_not_blow_up() -> None:
    groups = [group(45 + (i % 12), onset=i * 0.2) for i in range(200)]
    path, cost = viterbi(groups, SCORER, CTX)
    assert len(path) == 200
    assert cost == pytest.approx(path_cost(groups, path, SCORER, CTX), abs=1e-6)


def test_a_seven_string_tuning_decodes_with_the_default_weights() -> None:
    # ADR 0008 supports any tuning; ADR 0034's per-string weights, all zero by default,
    # must not break it. G4 (67) can sit on the seventh string, the high e.
    import math

    seven = Tuning(open_pitches=(35, 40, 45, 50, 55, 59, 64))
    ctx = Context(tuning=seven, max_span=4)
    path, cost = viterbi([group(67), group(35, onset=0.5)], SCORER, ctx)
    assert len(path) == 2 and math.isfinite(cost)


@pytest.mark.oracle
@SLOW
@given(groups=group_sequence(max_groups=5, max_notes=3))
def test_viterbi_matches_brute_force_with_the_acoustic_term_switched_on(
    groups: list[NoteGroup],
) -> None:
    # ADR 0047: the term lives in the emission cost, so the oracle charges it unchanged.
    import math

    evidence = {
        note: tuple(math.log((1 + (note.pitch + s) % 4) / 10) for s in range(6))
        for group in groups
        for note in group.notes
    }
    scorer = HandSetScorer(weights=CostWeights(acoustic=0.8), evidence=evidence)
    _, expected = brute_force_min(groups, scorer, CTX)
    path, got = viterbi(groups, scorer, CTX)
    assert got == pytest.approx(expected, abs=1e-9)
    assert path_cost(groups, path, scorer, CTX) == pytest.approx(got, abs=1e-9)
