"""Guitar-TECHS: real electric guitar, each note's string from a multi-track MIDI pickup (ADR 0049).

Reads the dataset as its zips unpack: per take, under ``P<player>_<category>/``,
``audio/directinput/directinput_<name>.wav``, ``audio/micamp/micamp_<name>.wav`` and
``midi/midi_<name>.mid``. The MIDI file has one track per string, named ``e B G D A E`` (``e`` the
high string), in standard tuning; a note's fret is its pitch less its string's open pitch. The
pickup tracks pitch from each string and can mistrack, so ``scripts/check_guitartechs.py``
checks the labels against the audio before anything trains on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tabsampler.types import NoteEvent, Position

#: Standard tuning, low string first.
STANDARD = (40, 45, 50, 55, 59, 64)

#: The MIDI track names, as the pickup's channels are named, to our strings (0 is the low E).
TRACK_STRINGS = {"E": 0, "A": 1, "D": 2, "G": 3, "B": 4, "e": 5}

MAX_FRET = 24


@dataclass(frozen=True, slots=True)
class GuitarTechsTake:
    """One take: its audio and every note's string and fret."""

    player: int
    category: str  # singlenotes, techniques, scales, chords or music
    name: str
    direct_input: Path
    mic_amp: Path | None
    notes: tuple[tuple[NoteEvent, Position], ...]  # by onset, then string


def read_tracks(path: Path) -> list[tuple[str, list[tuple[float, float, int]]]]:
    """Each MIDI track's name and its notes (start s, end s, pitch), under the file's tempo map."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))  # pyright: ignore[reportUnknownMemberType]
    return [
        (
            str(track.name),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            [
                (float(n.start), float(n.end), int(n.pitch))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                for n in track.notes  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            ],
        )
        for track in midi.instruments  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    ]


def parse_midi(path: Path) -> tuple[tuple[tuple[NoteEvent, Position], ...], list[str]]:
    """The notes of one take's MIDI file, placed, and each note dropped with its reason: a pitch
    below its string's open pitch or above its 24th fret is a mistracking, not a fingering.

    Raises:
        ValueError: if a track with notes names no string.
    """
    notes: list[tuple[NoteEvent, Position]] = []
    dropped: list[str] = []
    for name, track in read_tracks(path):
        if not track:
            continue
        if name not in TRACK_STRINGS:
            raise ValueError(f"track {name!r} names no string")
        string = TRACK_STRINGS[name]
        for start, end, pitch in track:
            fret = pitch - STANDARD[string]
            if fret < 0:
                dropped.append(f"{name}: pitch {pitch} below the open string")
            elif fret > MAX_FRET:
                dropped.append(f"{name}: fret {fret} above {MAX_FRET}")
            else:
                note = NoteEvent(onset=start, offset=end, pitch=pitch, confidence=1.0)
                notes.append((note, Position(string=string, fret=fret)))
    notes.sort(key=lambda placed: (placed[0].onset, placed[1].string))
    return tuple(notes), dropped


def load_takes(root: Path) -> tuple[list[GuitarTechsTake], list[tuple[str, str]]]:
    """Every take under ``root`` with a MIDI file and direct-input audio, and each take skipped
    with its reason."""
    takes: list[GuitarTechsTake] = []
    skipped: list[tuple[str, str]] = []
    for midi in sorted(root.glob("P*_*/midi/midi_*.mid")):
        folder = midi.parent.parent
        match = re.fullmatch(r"P(\d+)_(\w+)", folder.name)
        if match is None:
            continue
        name = midi.stem.removeprefix("midi_")
        direct = folder / "audio" / "directinput" / f"directinput_{name}.wav"
        if not direct.exists():
            skipped.append((f"{folder.name}/{name}", "no direct-input audio"))
            continue
        amp = folder / "audio" / "micamp" / f"micamp_{name}.wav"
        notes, _ = parse_midi(midi)
        takes.append(
            GuitarTechsTake(
                player=int(match.group(1)),
                category=match.group(2),
                name=name,
                direct_input=direct,
                mic_amp=amp if amp.exists() else None,
                notes=notes,
            )
        )
    return takes, skipped
