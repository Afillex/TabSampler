"""DadaGP token files to human fingerings (ADR 0021).

TRAINING DATA, NOT TEST DATA. DadaGP is what the cost model is fitted on; GuitarSet stays
the only test set (ADR 0003), and :func:`load_tracks` refuses :attr:`Split.TEST`.

Format facts, read from the dadaGP encoder's own source (``dadagp.py``) rather than assumed:

- A guitar note is ``<instrument>:note:s<string>:f<fret>``, and ``s1`` is the **highest**
  string, the GuitarPro convention. :class:`Position` counts from the low E, so for a
  6-string ``string = 6 - s``.
- The fret is **E-standardised**: GP value + capo - drop shift. Pitch survives, but the
  physical fingering of a capo'd or drop-tuned track does not, and nothing in the tokens
  says which tracks those are. ``scripts/dadagp_track_meta.py`` reads the original GP files
  to decide, and :func:`load_tracks` keeps only the songs it cleared.
- A note's effects follow it as ``nfx:`` tokens. ``wait:<ticks>`` advances time at 960 ticks
  per quarter note, and ``bfx:tempo_change:<bpm>`` changes the tempo from that beat on.

:func:`parse_tokens` is pure. :func:`load_tracks` reads the archive.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabsampler.data.splits import Split, assert_tuning_allowed
from tabsampler.types import MIDI_MAX, ChordState, NoteEvent, NoteGroup, Position, Tuning

#: The directory every path inside the v1.1 archive starts with.
ARCHIVE_ROOT = "DadaGP-v1.1/"

#: The encoder's clock: ``wait`` values are in these units per quarter note.
TICKS_PER_QUARTER = 960

#: Strings on the guitars we keep. Songs with a 7-string guitar are filtered out upstream,
#: and any stray seventh-string note drops its group.
N_STRINGS = 6

#: The encoder's clean (GM 24-28) and distorted (29-31) guitar groups. Bass, drums, leads
#: and pads are other instruments and are never parsed as guitar.
GUITAR_INSTRUMENT = re.compile(r"(clean|distorted)\d+")

#: ``nfx`` kinds that mean a note is not a new fretted onset: a tie continues an earlier
#: note, a dead note has no pitch, and a harmonic does not sound the fretted pitch.
NOT_AN_ONSET = frozenset({"tie", "dead", "harmonic"})

#: The split files, frozen by content. A different file is a different corpus, so a
#: mismatch is an error, never a warning (ADR 0021).
SPLIT_FILES: Mapping[Split, str] = {
    Split.TRAIN: "_DadaGP_training.json",
    Split.VALIDATION: "_DadaGP_validation.json",
}
SPLIT_SHA256: Mapping[Split, str] = {
    Split.TRAIN: "471ec175d1fcd93b76ff40b88313f923d80b08040583956e620a0a2552397462",
    Split.VALIDATION: "7a7fe3871da75afca5045d08d0837d7df57065ffe6292402f6b40d413ec396ce",
}

#: SHA-256 of the artist-disjoint split's validation keys, sorted and newline-joined
#: (ADR 0024): 2,607 of 26,181 songs and 519 of 4,866 artists.
ARTIST_VALIDATION_SHA256 = "538be675d39ffd4f5c43291ab613a6931342380761d16c4c4adc91124405b087"


@dataclass(frozen=True, slots=True)
class HumanTrack:
    """One guitar part as a human fingered it: the notes, and where they were played."""

    song: str  # the ``.tokens.txt`` key inside the archive
    instrument: str  # e.g. ``distorted0``
    steps: tuple[tuple[NoteGroup, ChordState], ...]


@dataclass(frozen=True, slots=True)
class ParseStats:
    """What parsing gave up. Reported next to any number measured on the result."""

    notes_kept: int = 0
    notes_skipped: int = 0  # ties, dead notes and harmonics
    tempo_changes_ignored: int = 0  # a change to zero BPM or below, which is not a tempo
    groups_kept: int = 0
    groups_dropped_negative_fret: int = 0
    groups_dropped_seventh_string: int = 0
    groups_dropped_string_collision: int = 0
    groups_dropped_malformed: int = 0


@dataclass(slots=True)
class _Note:
    instrument: str
    string: int  # GP numbering: 1 is the highest string
    fret: int
    skipped: bool = False


def _to_group(
    onset: float,
    offset: float,
    notes: list[_Note],
    tuning: Tuning,
) -> tuple[NoteGroup, ChordState] | str:
    """A (group, state) pair, or the name of the reason it cannot be one."""
    if any(n.string > N_STRINGS for n in notes):
        return "seventh_string"
    if any(n.fret < 0 for n in notes):
        return "negative_fret"
    if len({n.string for n in notes}) != len(notes):
        return "string_collision"
    if any(n.string < 1 for n in notes):
        return "malformed"
    paired = [
        (tuning.open_pitches[N_STRINGS - n.string] + n.fret, Position(N_STRINGS - n.string, n.fret))
        for n in notes
    ]
    if any(pitch > MIDI_MAX for pitch, _ in paired):
        return "malformed"
    # NoteGroup orders its notes by pitch and ChordState's positions are parallel to them,
    # so both are built from one stable sort of the same pairs.
    paired.sort(key=lambda item: item[0])
    group = NoteGroup(
        notes=tuple(
            NoteEvent(onset=onset, offset=offset, pitch=pitch, confidence=1.0)
            for pitch, _ in paired
        )
    )
    return group, ChordState(positions=tuple(position for _, position in paired))


def parse_tokens(
    text: str, song: str = "", tuning: Tuning = Tuning.STANDARD
) -> tuple[list[HumanTrack], ParseStats]:
    """Every guitar part in one DadaGP token file, as human fingerings.

    Pitches are reported in standard tuning whatever the song's ``downtune``: a uniform
    downtune moves every string together, which changes no fingering decision and leaves
    the candidate sets identical.
    """
    tempo = 120.0
    now = 0.0
    started = False
    beat: list[_Note] = []
    last: _Note | None = None
    beats: dict[str, list[tuple[float, float, list[_Note]]]] = {}
    skipped = 0
    tempo_ignored = 0

    def flush(duration: float) -> None:
        nonlocal skipped
        by_instrument: dict[str, list[_Note]] = {}
        for note in beat:
            if note.skipped:
                skipped += 1
            else:
                by_instrument.setdefault(note.instrument, []).append(note)
        for instrument, notes in by_instrument.items():
            beats.setdefault(instrument, []).append((now, now + duration, notes))

    for token in text.split():
        head, _, rest = token.partition(":")
        if not started:
            if head == "tempo":
                tempo = float(rest)
            started = token == "start"
            continue
        if head == "wait":
            duration = int(rest) / TICKS_PER_QUARTER * 60.0 / tempo
            flush(duration)
            now += duration
            beat, last = [], None
        elif head == "bfx":
            kind, _, value = rest.partition(":")
            if kind == "tempo_change":
                if float(value) > 0:
                    tempo = float(value)
                else:
                    tempo_ignored += 1  # keep the current tempo rather than divide by zero
        elif head == "nfx":
            if last is not None and rest.split(":")[0] in NOT_AN_ONSET:
                last.skipped = True
        elif head == "param":
            continue  # bend and tremolo curve points; they belong to the preceding nfx
        else:
            parts = token.split(":")
            if len(parts) == 4 and parts[1] == "note" and GUITAR_INSTRUMENT.fullmatch(parts[0]):
                last = _Note(parts[0], int(parts[2][1:]), int(parts[3][1:]))
                beat.append(last)
            else:
                last = None  # another instrument's note, a rest, or a bar marker
    flush(0.0)

    counts = {"seventh_string": 0, "negative_fret": 0, "string_collision": 0, "malformed": 0}
    tracks: list[HumanTrack] = []
    kept_groups = kept_notes = 0
    for instrument, instrument_beats in beats.items():
        steps: list[tuple[NoteGroup, ChordState]] = []
        for onset, offset, notes in instrument_beats:
            result = _to_group(onset, offset, notes, tuning)
            if isinstance(result, str):
                counts[result] += 1
            else:
                steps.append(result)
                kept_groups += 1
                kept_notes += len(notes)
        if steps:
            tracks.append(HumanTrack(song=song, instrument=instrument, steps=tuple(steps)))

    return tracks, ParseStats(
        notes_kept=kept_notes,
        notes_skipped=skipped,
        tempo_changes_ignored=tempo_ignored,
        groups_kept=kept_groups,
        groups_dropped_negative_fret=counts["negative_fret"],
        groups_dropped_seventh_string=counts["seventh_string"],
        groups_dropped_string_collision=counts["string_collision"],
        groups_dropped_malformed=counts["malformed"],
    )


def artist_of(key: str) -> str:
    """The artist folder of a token-file key, case-folded: folders that differ only by case
    are one artist on a case-insensitive filesystem (ADR 0021)."""
    return key.split("/")[1].casefold()


def artist_split(
    keys: Sequence[str], buckets: int = 10, validation_buckets: int = 1
) -> dict[str, Split]:
    """Whole artists to one side: about ``validation_buckets / buckets`` of artists go to
    validation, chosen by a hash of the artist so the assignment never depends on order."""
    out: dict[str, Split] = {}
    for key in keys:
        bucket = int(hashlib.sha256(artist_of(key).encode()).hexdigest(), 16) % buckets
        out[key] = Split.VALIDATION if bucket < validation_buckets else Split.TRAIN
    return out


def _split_keys(raw: bytes) -> list[str]:
    """Typed boundary around the split file's JSON."""
    entries: list[dict[str, object]] = json.loads(raw)
    return [str(entry["tokens.txt"]) for entry in entries]


def _cleared(meta_path: Path) -> set[str]:
    """Songs ``scripts/dadagp_track_meta.py`` found to be standard 6-string with no capo."""
    meta: dict[str, dict[str, object]] = json.loads(meta_path.read_text())
    return {key for key, record in meta.items() if record.get("clean") is True}


def load_tracks(
    archive_path: Path | str,
    split: Split,
    meta_path: Path | str,
    expected_sha256: Mapping[Split, str] = SPLIT_SHA256,
    tuning: Tuning = Tuning.STANDARD,
    sample: int | None = None,
    seed: int = 0,
    on_song: Callable[[str, ParseStats], None] | None = None,
    scheme: Literal["shipped", "artist"] = "shipped",
    artist_sha256: str | None = ARTIST_VALIDATION_SHA256,
) -> Iterator[HumanTrack]:
    """Human fingerings for every cleared song in one DadaGP split.

    Args:
        scheme: ``"shipped"`` is DadaGP's own song-level split (ADR 0021); ``"artist"``
            reassigns the same songs so that no artist is on both sides (ADR 0024).
        artist_sha256: The frozen hash of the artist split's validation keys; ``None``
            skips the check, for tests that build their own archives.
        sample: Take this many cleared songs, chosen with ``seed``, instead of all of them.
        on_song: Called with each song's parse statistics, so a caller can report what
            was dropped without this function printing anything.

    Raises:
        TestSetMisuseError: for :attr:`Split.TEST`. DadaGP has no test split here.
        ValueError: if the split file's SHA-256 is not the frozen one.
    """
    assert_tuning_allowed(split)
    if split not in SPLIT_FILES:
        raise ValueError(f"DadaGP has no {split.value} split")

    cleared = _cleared(Path(meta_path))
    with zipfile.ZipFile(archive_path) as archive:

        def verified(side: Split) -> list[str]:
            raw = archive.read(ARCHIVE_ROOT + SPLIT_FILES[side])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected_sha256[side]:
                raise ValueError(
                    f"{SPLIT_FILES[side]} has sha256 {digest}, not the frozen "
                    f"{expected_sha256[side]}: this is a different split (ADR 0021)"
                )
            return _split_keys(raw)

        if scheme == "shipped":
            listed = verified(split)
        else:
            union = verified(Split.TRAIN) + verified(Split.VALIDATION)
            assignment = artist_split(union)
            if artist_sha256 is not None:
                validation = sorted(k for k, side in assignment.items() if side is Split.VALIDATION)
                digest = hashlib.sha256("\n".join(validation).encode()).hexdigest()
                if digest != artist_sha256:
                    raise ValueError(
                        f"artist split validation keys have sha256 {digest}, not the frozen "
                        f"{artist_sha256}: this is a different split (ADR 0024)"
                    )
            listed = [k for k in union if assignment[k] is split]
        names = set(archive.namelist())
        # 32 of the 26,181 listed files are absent from the v1.1 archive: folders whose
        # names differ only in case collapsed into one on a case-insensitive filesystem.
        keys = [k for k in listed if k in cleared and ARCHIVE_ROOT + k in names]
        if sample is not None:
            keys = sorted(random.Random(seed).sample(keys, min(sample, len(keys))))
        for key in keys:
            tracks, stats = parse_tokens(
                archive.read(ARCHIVE_ROOT + key).decode("utf-8"), song=key, tuning=tuning
            )
            if on_song is not None:
                on_song(key, stats)
            yield from tracks
