"""Guitar Pro 5 export (ADRs 0017, 0059), through pyguitarpro.

pyguitarpro ships no type stubs, so it is confined to this module's ``_song`` and ``_write``.
The disclaimer goes in the score notice (wrapped at 255 characters, as the format requires)
and, shortened, as text on the first beat. Guitar Pro counts strings from the highest; a capo
is the track's ``offset`` ("Height of the capo", in pyguitarpro's gp5 track reader), and frets
stay relative to it as ours are.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import guitarpro as gp

from tabsampler.render.grid import DISCLAIMER, TEMPO_BPM, UNITS_PER_BAR, GridEvent, to_grid
from tabsampler.types import TabNote, Tuning

#: On the first beat, where a reader of the tab will see it.
BEAT_TEXT = "Rhythm NOT transcribed: see the score notice"


def _song(events: Sequence[GridEvent], tuning: Tuning) -> Any:
    song: Any = gp.Song()  # pyright: ignore[reportUnknownMemberType]
    song.title = "Tab Sampler transcription"
    song.notice = [DISCLAIMER]
    song.tempo = TEMPO_BPM
    n_bars = (events[-1].start + events[-1].length) // UNITS_PER_BAR
    bar_time = 4 * gp.Duration.quarterTime
    song.measureHeaders = []
    for i in range(n_bars):
        header: Any = gp.MeasureHeader(number=i + 1, start=gp.Duration.quarterTime + i * bar_time)  # pyright: ignore[reportUnknownMemberType]
        song.addMeasureHeader(header)
    track: Any = gp.Track(song)  # pyright: ignore[reportUnknownMemberType]
    track.name = "Guitar"
    track.fretCount = tuning.n_frets
    track.offset = tuning.capo
    n = tuning.n_strings
    track.strings = [
        gp.GuitarString(n - s, tuning.open_pitches[s])  # pyright: ignore[reportUnknownMemberType]
        for s in reversed(range(n))
    ]
    song.tracks = [track]

    for k, event in enumerate(events):
        voice: Any = track.measures[event.start // UNITS_PER_BAR].voices[0]
        beat: Any = gp.Beat(voice, duration=gp.Duration(value=UNITS_PER_BAR // event.length))  # pyright: ignore[reportUnknownMemberType]
        beat.status = gp.BeatStatus.normal if event.positions else gp.BeatStatus.rest
        if k == 0:
            beat.text = BEAT_TEXT
        for position in event.positions:
            kind = gp.NoteType.tie if event.tie_stop else gp.NoteType.normal
            beat.notes.append(
                gp.Note(beat, value=position.fret, string=n - position.string, type=kind)  # pyright: ignore[reportUnknownMemberType]
            )
        voice.beats.append(beat)
    return song


def _write(song: Any) -> bytes:
    buffer = io.BytesIO()
    gp.write(song, buffer, version=(5, 1, 0))  # pyright: ignore[reportUnknownMemberType]
    return buffer.getvalue()


def render_guitarpro(tab: Sequence[TabNote], tuning: Tuning, window_s: float = 0.03) -> bytes:
    """The tab as a Guitar Pro 5 file."""
    return _write(_song(to_grid(tab, window_s), tuning))
