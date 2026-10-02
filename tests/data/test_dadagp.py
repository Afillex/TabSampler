"""Tests for the DadaGP token parser and split loader (ADR 0021).

Every token string here is written by hand, so the tests need neither the dataset nor
anything derived from it. The format facts they pin were read from the dadaGP encoder's
own source: string 1 is the highest string, a fret is nut-relative, a note's effects follow
it as ``nfx:`` tokens, and 960 ticks make a quarter note.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tabsampler.data.dadagp import (
    ARCHIVE_ROOT,
    ARTIST_VALIDATION_SHA256,
    SPLIT_SHA256,
    artist_of,
    artist_split,
    load_tracks,
    parse_tokens,
)
from tabsampler.data.splits import Split
from tabsampler.errors import TestSetMisuseError
from tabsampler.types import Position

HEAD = "artist:unknown_artist downtune:0 tempo:120 start"


def song(*body: str) -> str:
    return " ".join([HEAD, *body, "end"])


# ------------------------------------------------------------------ strings and pitch


def test_string_six_is_our_low_e_and_string_one_is_our_high_e() -> None:
    # GP numbers strings from the highest; Position numbers them from the low E.
    tracks, _ = parse_tokens(song("clean0:note:s6:f3", "wait:960", "clean0:note:s1:f0", "wait:960"))
    (track,) = tracks
    first, second = track.steps
    assert first[1].positions == (Position(0, 3),)
    assert first[0].notes[0].pitch == 43  # low E + 3 = G2
    assert second[1].positions == (Position(5, 0),)
    assert second[0].notes[0].pitch == 64  # open high e


def test_a_uniform_downtune_does_not_change_the_fingering() -> None:
    # Downtuning shifts every string equally, so the fingering problem is identical.
    # Pitches are reported in standard tuning, which keeps the candidate sets equal too.
    text = "artist:a downtune:-2 tempo:120 start clean0:note:s6:f3 wait:960 end"
    tracks, _ = parse_tokens(text)
    assert tracks[0].steps[0][0].notes[0].pitch == 43


# ------------------------------------------------------------------ grouping and time


def test_notes_at_the_same_tick_on_one_instrument_form_one_chord() -> None:
    tracks, _ = parse_tokens(
        song("clean0:note:s6:f0", "clean0:note:s5:f2", "clean0:note:s4:f2", "wait:960")
    )
    ((group, state),) = tracks[0].steps
    assert len(group) == 3
    assert sorted(state.positions, key=lambda p: p.string) == [
        Position(0, 0),
        Position(1, 2),
        Position(2, 2),
    ]


def test_chord_positions_stay_parallel_to_the_pitch_sorted_notes() -> None:
    # NoteGroup sorts notes by pitch; ChordState must follow the same order, or every
    # note would be paired with another note's position.
    tracks, _ = parse_tokens(song("clean0:note:s1:f0", "clean0:note:s6:f0", "wait:960"))
    group, state = tracks[0].steps[0]
    assert [n.pitch for n in group.notes] == [40, 64]
    assert state.positions == (Position(0, 0), Position(5, 0))


def test_onsets_are_in_seconds_from_the_tempo() -> None:
    # 120 BPM: a quarter note (960 ticks) is 0.5 s.
    tracks, _ = parse_tokens(
        song("clean0:note:s6:f0", "wait:960", "clean0:note:s6:f1", "wait:480", "clean0:note:s6:f2")
    )
    onsets = [group.onset for group, _ in tracks[0].steps]
    assert onsets == pytest.approx([0.0, 0.5, 0.75])


def test_a_tempo_change_applies_from_where_it_occurs() -> None:
    tracks, _ = parse_tokens(
        song(
            "clean0:note:s6:f0",
            "wait:960",
            "bfx:tempo_change:60",  # from here a quarter note lasts 1 s
            "clean0:note:s6:f1",
            "wait:960",
            "clean0:note:s6:f2",
        )
    )
    onsets = [group.onset for group, _ in tracks[0].steps]
    assert onsets == pytest.approx([0.0, 0.5, 1.5])


def test_a_tempo_change_to_zero_is_ignored_and_counted() -> None:
    # 11 of 22,034 cleared songs contain bfx:tempo_change:0. Zero BPM is not a tempo, and
    # it would divide by zero; the current tempo stands and the glitch is counted.
    tracks, stats = parse_tokens(
        song(
            "clean0:note:s6:f0",
            "wait:960",
            "bfx:tempo_change:0",
            "clean0:note:s6:f1",
            "wait:960",
            "clean0:note:s6:f2",
        )
    )
    onsets = [group.onset for group, _ in tracks[0].steps]
    assert onsets == pytest.approx([0.0, 0.5, 1.0])
    assert stats.tempo_changes_ignored == 1


def test_each_guitar_instrument_is_its_own_track() -> None:
    tracks, _ = parse_tokens(song("clean0:note:s6:f0", "distorted0:note:s6:f5", "wait:960"))
    assert sorted(t.instrument for t in tracks) == ["clean0", "distorted0"]


def test_bass_drums_leads_and_pads_are_ignored() -> None:
    tracks, _ = parse_tokens(
        song(
            "bass:note:s4:f0",
            "drums:note:36",
            "leads:note:s1:f5",
            "pads:note:s2:f3",
            "clean0:note:s6:f0",
            "wait:960",
        )
    )
    assert [t.instrument for t in tracks] == ["clean0"]


def test_rests_repeats_and_bend_parameters_are_not_notes() -> None:
    tracks, _ = parse_tokens(
        song(
            "measure:repeat_open",
            "clean0:note:s3:f5",
            "nfx:bend",
            "param:val4:vib0",
            "param:dur6",
            "wait:960",
            "clean0:rest",
            "new_measure",
            "wait:960",
        )
    )
    assert len(tracks[0].steps) == 1


# ------------------------------------------------------------------ notes that are not onsets


@pytest.mark.parametrize("effect", ["nfx:tie", "nfx:dead", "nfx:harmonic:1"])
def test_ties_dead_notes_and_harmonics_are_not_fretted_onsets(effect: str) -> None:
    # A tie continues an earlier note, a dead note has no pitch, and a harmonic does not
    # sound the fretted pitch. None of them is a new note at a (string, fret).
    tracks, stats = parse_tokens(
        song("clean0:note:s6:f3", "wait:960", "clean0:note:s6:f3", effect, "wait:960")
    )
    assert len(tracks[0].steps) == 1
    assert stats.notes_skipped == 1


def test_an_effect_after_another_instruments_note_does_not_attach_to_ours() -> None:
    tracks, _ = parse_tokens(song("clean0:note:s6:f3", "bass:note:s4:f0", "nfx:tie", "wait:960"))
    assert len(tracks[0].steps) == 1  # the tie belonged to the bass note


def test_a_grace_note_effect_keeps_its_main_note() -> None:
    tracks, _ = parse_tokens(song("clean0:note:s3:f7", "nfx:grace:fret5", "wait:960"))
    assert tracks[0].steps[0][1].positions == (Position(3, 7),)


# ---------------------------------------------------------- groups that cannot be represented


def test_a_negative_fret_drops_the_group_and_is_counted() -> None:
    # The encoder writes a drop-tuned open low string as fret -2. Its physical fingering is
    # gone, so the group is dropped rather than guessed at.
    tracks, stats = parse_tokens(
        song("clean0:note:s6:f-2", "clean0:note:s5:f0", "wait:960", "clean0:note:s6:f3", "wait:960")
    )
    assert len(tracks[0].steps) == 1
    assert stats.groups_dropped_negative_fret == 1


def test_a_seventh_string_note_drops_the_group_and_is_counted() -> None:
    tracks, stats = parse_tokens(song("clean0:note:s7:f0", "wait:960", "clean0:note:s6:f0"))
    assert len(tracks[0].steps) == 1
    assert stats.groups_dropped_seventh_string == 1


def test_two_notes_on_one_string_at_once_drop_the_group_and_are_counted() -> None:
    # A malformed tab. ChordState refuses it, so it must not reach ChordState.
    tracks, stats = parse_tokens(
        song("clean0:note:s6:f3", "clean0:note:s6:f5", "wait:960", "clean0:note:s5:f0")
    )
    assert len(tracks[0].steps) == 1
    assert stats.groups_dropped_string_collision == 1


def test_a_track_left_with_no_groups_is_not_returned() -> None:
    tracks, _ = parse_tokens(song("clean0:note:s6:f-2", "wait:960"))
    assert tracks == []


# ------------------------------------------------------------------ the split loader


def make_archive(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny stand-in for the DadaGP archive, built here so no real data is needed."""
    training = [{"tokens.txt": "A/Artist/Song.gp4.tokens.txt", "validation_set": False}]
    validation = [{"tokens.txt": "B/Band/Tune.gp4.tokens.txt", "validation_set": True}]
    archive = tmp_path / "DadaGP.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{ARCHIVE_ROOT}_DadaGP_training.json", json.dumps(training))
        zf.writestr(f"{ARCHIVE_ROOT}_DadaGP_validation.json", json.dumps(validation))
        zf.writestr(f"{ARCHIVE_ROOT}A/Artist/Song.gp4.tokens.txt", song("clean0:note:s6:f0"))
        zf.writestr(f"{ARCHIVE_ROOT}B/Band/Tune.gp4.tokens.txt", song("clean0:note:s5:f0"))
    meta = tmp_path / "track_meta.json"
    meta.write_text(
        json.dumps(
            {
                "A/Artist/Song.gp4.tokens.txt": {"clean": True, "reason": "clean"},
                "B/Band/Tune.gp4.tokens.txt": {"clean": True, "reason": "clean"},
            }
        )
    )
    return archive, meta


def hashes_of(archive: Path) -> dict[Split, str]:
    with zipfile.ZipFile(archive) as zf:
        return {
            split: hashlib.sha256(zf.read(f"{ARCHIVE_ROOT}_DadaGP_{name}.json")).hexdigest()
            for split, name in ((Split.TRAIN, "training"), (Split.VALIDATION, "validation"))
        }


def test_the_loader_reads_only_the_requested_split(tmp_path: Path) -> None:
    archive, meta = make_archive(tmp_path)
    expected = hashes_of(archive)
    train = list(load_tracks(archive, Split.TRAIN, meta, expected_sha256=expected))
    validation = list(load_tracks(archive, Split.VALIDATION, meta, expected_sha256=expected))
    assert [t.song for t in train] == ["A/Artist/Song.gp4.tokens.txt"]
    assert [t.song for t in validation] == ["B/Band/Tune.gp4.tokens.txt"]


def test_the_loader_refuses_the_test_split(tmp_path: Path) -> None:
    # DadaGP has no test split in this project; GuitarSet is the test set (ADR 0003).
    archive, meta = make_archive(tmp_path)
    with pytest.raises(TestSetMisuseError):
        list(load_tracks(archive, Split.TEST, meta, expected_sha256=hashes_of(archive)))


def test_the_loader_refuses_a_split_file_that_has_changed(tmp_path: Path) -> None:
    # The split is frozen by hash. A different file means a different corpus, silently.
    archive, meta = make_archive(tmp_path)
    wrong = {Split.TRAIN: "0" * 64, Split.VALIDATION: "0" * 64}
    with pytest.raises(ValueError, match="sha256"):
        list(load_tracks(archive, Split.TRAIN, meta, expected_sha256=wrong))


def test_the_loader_skips_songs_the_tuning_pass_did_not_clear(tmp_path: Path) -> None:
    archive, meta = make_archive(tmp_path)
    meta.write_text(
        json.dumps(
            {
                "A/Artist/Song.gp4.tokens.txt": {"clean": False, "reason": "capo"},
                "B/Band/Tune.gp4.tokens.txt": {"clean": True, "reason": "clean"},
            }
        )
    )
    assert list(load_tracks(archive, Split.TRAIN, meta, expected_sha256=hashes_of(archive))) == []


def test_the_committed_hashes_are_the_v1_1_release() -> None:
    # Pins the frozen split. Changing these is changing the corpus, which needs an ADR.
    assert SPLIT_SHA256[Split.TRAIN].startswith("471ec175")
    assert SPLIT_SHA256[Split.VALIDATION].startswith("7a7fe387")
    assert ARTIST_VALIDATION_SHA256.startswith("538be675")


# --------------------------------------------------------- artist-disjoint split (ADR 0024)


def test_an_artist_is_the_folder_under_the_letter() -> None:
    assert artist_of("M/Mago de Oz/Mago de Oz - Alma.gp4.tokens.txt") == "mago de oz"


def test_case_variants_of_one_artist_share_a_side() -> None:
    a = artist_split(["M/Mago de Oz/x.tokens.txt", "M/Mago de oz/y.tokens.txt"])
    assert len(set(a.values())) == 1


def test_no_artist_appears_on_both_sides() -> None:
    keys = [f"A/Artist{i % 37}/Song{i}.tokens.txt" for i in range(500)]
    assignment = artist_split(keys)
    sides: dict[str, set[Split]] = {}
    for key, side in assignment.items():
        sides.setdefault(artist_of(key), set()).add(side)
    assert all(len(s) == 1 for s in sides.values())
    assert set(assignment.values()) == {Split.TRAIN, Split.VALIDATION}


def test_the_artist_split_is_deterministic() -> None:
    keys = [f"A/Artist{i}/Song.tokens.txt" for i in range(100)]
    assert artist_split(keys) == artist_split(list(reversed(keys)))


def test_the_loader_serves_the_artist_scheme(tmp_path: Path) -> None:
    archive, meta = make_archive(tmp_path)
    expected = hashes_of(archive)
    train = {
        t.song
        for t in load_tracks(
            archive,
            Split.TRAIN,
            meta,
            expected_sha256=expected,
            scheme="artist",
            artist_sha256=None,
        )
    }
    val = {
        t.song
        for t in load_tracks(
            archive,
            Split.VALIDATION,
            meta,
            expected_sha256=expected,
            scheme="artist",
            artist_sha256=None,
        )
    }
    assert train | val == {"A/Artist/Song.gp4.tokens.txt", "B/Band/Tune.gp4.tokens.txt"}
    assert not train & val


def test_the_artist_scheme_refuses_a_changed_assignment(tmp_path: Path) -> None:
    # The artist split is frozen by the hash of its validation list, like the shipped one.
    archive, meta = make_archive(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        list(
            load_tracks(
                archive,
                Split.TRAIN,
                meta,
                expected_sha256=hashes_of(archive),
                scheme="artist",
                artist_sha256="0" * 64,
            )
        )
