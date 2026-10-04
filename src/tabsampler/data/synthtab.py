"""SynthTab: audio rendered from DadaGP's tabs, with every note's string and fret (ADR 0046).

Reads the development set as it unpacks: audio at
``SynthTab_Dev/<family>/<tone>/<track>/<track>/*.flac`` (mono, 22,050 Hz) and labels at
``SynthTab_Dev/jams/<track>/*.jams``. A label file holds six ``note_tab`` annotations, one per
string; each string's sandbox gives its Guitar Pro number (1 is the highest string) and its open
pitch, so tunings other than standard come through as written.

Times are Guitar Pro ticks, 960 to the quarter note, converted with the file's tempo annotation.
Checked against the audio on 2026-10-04: on five tracks the rendering runs the last note's end
plus about four seconds of ring-out. ``scripts/check_synthtab.py`` checks that onsets line up.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabsampler.types import NoteEvent, Position

TICKS_PER_QUARTER = 960
STRINGS = 6

#: How much later than its label a note sounds in the rendered audio, per family: measured by
#: ``scripts/check_synthtab.py`` against GuitarSet's player 00 as the control (2026-10-04),
#: at 5.8 ms resolution. Applied where note windows are cut, never to the labels themselves.
RENDER_LATENCY = {
    "acoustic": 0.029,
    "electric_clean": 0.017,
    "electric_distortion_di": 0.017,
    "electric_muted": 0.017,
}


@dataclass(frozen=True, slots=True)
class SynthTabTrack:
    """One rendered track: its audio and every note's string and fret."""

    name: str
    family: str  # acoustic, electric_clean, electric_distortion_di or electric_muted
    tone: str
    audio: Path
    notes: tuple[tuple[NoteEvent, Position], ...]  # by onset, then string
    open_pitches: tuple[int, ...]  # low string first


def ticks_to_seconds(ticks: float, tempo: Sequence[tuple[float, float]]) -> float:
    """``ticks`` in seconds, under ``tempo``: (start tick, beats per minute), sorted, the first
    starting at tick 0, each holding until the next."""
    seconds = 0.0
    for index, (start, bpm) in enumerate(tempo):
        if ticks <= start:
            break
        end = tempo[index + 1][0] if index + 1 < len(tempo) else math.inf
        seconds += (min(ticks, end) - start) * 60.0 / (TICKS_PER_QUARTER * bpm)
    return seconds


def parse_jams(
    data: dict[str, Any],
) -> tuple[tuple[tuple[NoteEvent, Position], ...], tuple[int, ...]]:
    """The notes of one label file, placed, and the open pitches, low string first.

    Raises:
        ValueError: if a string is numbered outside 1-6.
    """
    annotations: list[dict[str, Any]] = data["annotations"]
    tempo = sorted(
        (float(obs["time"]), float(obs["value"]))
        for a in annotations
        if a["namespace"] == "tempo"
        for obs in a["data"]
    )
    notes: list[tuple[NoteEvent, Position]] = []
    open_pitches = [0] * STRINGS
    for annotation in (a for a in annotations if a["namespace"] == "note_tab"):
        number = int(annotation["sandbox"]["string_index"])
        if not 1 <= number <= STRINGS:
            raise ValueError(f"string {number} is outside a six-string guitar")
        string, open_pitch = STRINGS - number, int(annotation["sandbox"]["open_tuning"])
        open_pitches[string] = open_pitch
        for obs in annotation["data"]:
            fret = int(obs["value"]["fret"])
            onset = ticks_to_seconds(float(obs["time"]), tempo)
            offset = ticks_to_seconds(float(obs["time"]) + float(obs["duration"]), tempo)
            note = NoteEvent(onset=onset, offset=offset, pitch=open_pitch + fret, confidence=1.0)
            notes.append((note, Position(string=string, fret=fret)))
    notes.sort(key=lambda placed: (placed[0].onset, placed[1].string))
    return tuple(notes), tuple(open_pitches)


def load_tracks(root: Path) -> tuple[list[SynthTabTrack], list[tuple[str, str]]]:
    """Every usable rendered track under the development set's ``root``, and each track skipped
    with its reason: a few ship without a label file, and a few are for seven-string guitar,
    which the six-string model cannot use. Skipped, not guessed at."""
    tracks: list[SynthTabTrack] = []
    skipped: list[tuple[str, str]] = []
    for audio in sorted(root.glob("*/*/*/*/*.flac")):
        family, tone, name = audio.parts[-5], audio.parts[-4], audio.parts[-3]
        labels = sorted((root / "jams" / name).glob("*.jams"))
        if not labels:
            skipped.append((name, "no label file"))
            continue
        try:
            notes, open_pitches = parse_jams(json.loads(labels[0].read_text()))
        except ValueError as error:
            skipped.append((name, str(error)))
            continue
        tracks.append(SynthTabTrack(name, family, tone, audio, notes, open_pitches))
    return tracks, skipped
