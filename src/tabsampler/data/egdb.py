"""EGDB: electric guitar played to tab, with a hexaphonic pickup -- the second test set.

Every clip is test data (ADR 0050): :func:`tabsampler.data.splits.egdb_test_ids` lists them, and
nothing may be tuned, selected or calibrated on them. Reads the dataset as its Google Drive folder
downloads: ``audio_label/<n>.midi`` and ``audio_DI/<n>.wav`` for clips 1-240 (44.1 kHz, mono). A
label file has one MIDI track per string, named ``1`` (the high e) to ``6`` (the low E), in
standard tuning. The amplifier renderings beside them are not read: the target is clean electric
(ADR 0049).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tabsampler.data.guitartechs import MAX_FRET, STANDARD, read_tracks
from tabsampler.data.splits import egdb_clip_id
from tabsampler.types import NoteEvent, Position

STRINGS = 6


@dataclass(frozen=True, slots=True)
class EgdbClip:
    """One clip: its test id, its direct-input audio and every note's string and fret."""

    clip_id: str
    direct_input: Path
    notes: tuple[tuple[NoteEvent, Position], ...]  # by onset, then string


def parse_labels(path: Path) -> tuple[tuple[tuple[NoteEvent, Position], ...], list[str]]:
    """The notes of one label file, placed, and each note dropped with its reason.

    Raises:
        ValueError: if a track with notes names no string 1-6.
    """
    notes: list[tuple[NoteEvent, Position]] = []
    dropped: list[str] = []
    for name, track in read_tracks(path):
        if not track:
            continue
        if name not in {str(n) for n in range(1, STRINGS + 1)}:
            raise ValueError(f"track {name!r} names no string")
        string = STRINGS - int(name)
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
