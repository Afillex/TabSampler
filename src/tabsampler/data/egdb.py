"""EGDB: electric guitar played to tab, with a hexaphonic pickup -- the second test set.

Every clip is test data (ADR 0050): :func:`tabsampler.data.splits.egdb_test_ids` lists them, and
nothing may be tuned, selected or calibrated on them. Reads the dataset as its Google Drive folder
downloads: ``audio_label/<n>.midi`` and ``audio_DI/<n>.wav`` for clips 1-240 (44.1 kHz, mono). A
label file's notes sit on MIDI channels 0 (the high e) to 5 (the low E), in standard tuning;
the track names are mostly empty and not used (ADR 0053). The amplifier renderings beside them
are not read: the target is clean electric (ADR 0049).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tabsampler.data.guitartechs import MAX_FRET, STANDARD
from tabsampler.data.splits import egdb_clip_id
from tabsampler.types import NoteEvent, Position

STRINGS = 6


@dataclass(frozen=True, slots=True)
class EgdbClip:
    """One clip: its test id, its direct-input audio and every note's string and fret."""

    clip_id: str
    direct_input: Path
    notes: tuple[tuple[NoteEvent, Position], ...]  # by onset, then string


def read_notes(path: Path) -> list[tuple[float, float, int, int]]:
    """Every note of a MIDI file as (start s, end s, pitch, channel), under its tempo map."""
    import mido

    notes: list[tuple[float, float, int, int]] = []
    sounding: dict[tuple[int, int], list[float]] = {}
    now = 0.0
    for message in mido.MidiFile(str(path)):  # pyright: ignore[reportUnknownVariableType]
        now += float(message.time)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        kind = str(message.type)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        if kind not in ("note_on", "note_off"):
            continue
        key = (int(message.channel), int(message.note))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        if kind == "note_on" and int(message.velocity) > 0:  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            sounding.setdefault(key, []).append(now)
        elif sounding.get(key):
            notes.append((sounding[key].pop(0), now, key[1], key[0]))
    return notes


def parse_labels(path: Path) -> tuple[tuple[tuple[NoteEvent, Position], ...], list[str]]:
    """The notes of one label file, placed by their MIDI channel -- 0 the high e, 5 the low E
    (ADR 0053) -- and each note dropped with its reason."""
    notes: list[tuple[NoteEvent, Position]] = []
    dropped: list[str] = []
    for start, end, pitch, channel in sorted(read_notes(path)):
        if not 0 <= channel < STRINGS:
            dropped.append(f"channel {channel}: no string")
            continue
        string = STRINGS - 1 - channel
        fret = pitch - STANDARD[string]
        if fret < 0:
            dropped.append(f"channel {channel}: pitch {pitch} below the open string")
        elif fret > MAX_FRET:
            dropped.append(f"channel {channel}: fret {fret} above {MAX_FRET}")
        else:
            note = NoteEvent(onset=start, offset=end, pitch=pitch, confidence=1.0)
            notes.append((note, Position(string=string, fret=fret)))
    notes.sort(key=lambda placed: (placed[0].onset, placed[1].string))
    return tuple(notes), dropped


def load_clips(root: Path) -> tuple[list[EgdbClip], list[tuple[str, str]]]:
    """Every clip under ``root`` with labels and direct-input audio, by number, and each clip
    skipped with its reason."""
    clips: list[EgdbClip] = []
    skipped: list[tuple[str, str]] = []
    labels = sorted((root / "audio_label").glob("*.midi"), key=lambda p: int(p.stem))
    for label in labels:
        clip_id = egdb_clip_id(int(label.stem))
        audio = root / "audio_DI" / f"{label.stem}.wav"
        if not audio.exists():
            skipped.append((clip_id, "no direct-input audio"))
            continue
        clips.append(EgdbClip(clip_id, audio, parse_labels(label)[0]))
    return clips, skipped
