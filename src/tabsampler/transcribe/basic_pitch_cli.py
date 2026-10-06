"""Basic Pitch behind a subprocess boundary and a note-event cache (ADR 0001).

``basic-pitch`` 0.4.0 cannot be installed alongside this package: on macOS arm64 it
requires ``tensorflow-macos``, which has no wheel past cp311. It therefore lives in its
own Python 3.11 environment (``scripts/setup_transcriber.sh``) and is driven as a
subprocess, with results cached on disk.

Everything below was verified by running basic-pitch 0.4.0 rather than read from docs:

- The **output directory must already exist** -- ``verify_output_dir`` raises otherwise --
  and ``build_output_path`` raises ``IOError`` if the target file *does* exist. So each
  run goes into a fresh temporary directory and the CSV is moved into the cache after.
- The note-event CSV is **ragged**. The header names five columns, but the writer does
  ``row = [start, end, pitch, velocity]`` then ``row.extend(pitch_bend)``, so a row has
  four fields plus one column per bend value. A ``csv.DictReader`` would keep only the
  first bend value and silently drop the rest.
- Rows are **not sorted by onset**.
- ``velocity`` is ``int(round(127 * amplitude))``, so recovering a 0..1 confidence loses
  precision. Accepted and recorded in ADR 0001; nothing consumes ``confidence`` before
  Phase 3.
- ``pitch_bend`` values are in the model's internal contour bins. The bins-per-semitone
  factor is **unverified**, so they are carried through untouched and unused. Bends are
  Phase 7; verify against the basic-pitch source before building on this field.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from tabsampler.audio.io import DEFAULT_SAMPLE_RATE
from tabsampler.errors import TranscriberFailedError, TranscriberUnavailableError
from tabsampler.types import NoteEvent

#: Exactly the header basic-pitch 0.4.0 writes.
EXPECTED_HEADER = ("start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend")

#: basic-pitch scales amplitude to a MIDI velocity with this factor.
MIDI_VELOCITY_SCALE = 127

DEFAULT_CACHE_DIR = Path("cache/note_events")

#: Where `uv tool install` puts executables. Not on PATH by default, so a bare name
#: is looked up here before giving up.
UV_TOOL_BIN = Path.home() / ".local" / "bin"


def _resolve_exe(exe: str | Path) -> str:
    """Resolve the transcriber executable.

    A path is taken as given. A bare name is looked up on PATH, then in
    ``~/.local/bin``, because ``uv tool install`` installs there and that directory is
    not on PATH by default. If neither finds it, the name is returned unchanged so the
    failure surfaces as TranscriberUnavailableError with a useful message.
    """
    text = str(exe)
    if os.sep in text:
        return text
    found = shutil.which(text)
    if found:
        return found
    candidate = UV_TOOL_BIN / text
    return str(candidate) if candidate.is_file() else text


@dataclass(frozen=True, slots=True)
class BasicPitchParams:
    """Every knob that changes the transcriber's output.

    All of it feeds the cache key and belongs in the ``results.csv`` row, so a number
    can never be compared against one produced under different settings.
    """

    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 127.70
    #: CoreML is basic-pitch's macOS default and the only backend installed by
    #: scripts/setup_transcriber.sh. Set explicitly so the choice is reproducible.
    model_serialization: str = "coreml"
    #: Part of the cache key. Bump it when the transcriber is upgraded, or an upgrade
    #: will silently reuse results produced by the old version.
    version_tag: str = "basic-pitch-0.4.0"


def parse_note_events_csv(path: Path | str) -> list[NoteEvent]:
    """Parse a basic-pitch note-event CSV into sorted :class:`NoteEvent` objects.

    Raises:
        TranscriberFailedError: if the header is not the expected one, or a row has
            fewer than four fields. Both mean the contract changed, which must surface
            rather than silently produce fewer notes.
    """
    csv_path = Path(path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise TranscriberFailedError(f"{csv_path} is completely empty (no header row)")

    header = tuple(field.strip() for field in rows[0])
    if header != EXPECTED_HEADER:
        raise TranscriberFailedError(
            f"unexpected header in {csv_path}: {header!r}; expected {EXPECTED_HEADER!r}. "
            f"The basic-pitch output contract has changed."
        )

    notes: list[NoteEvent] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) < 4:
            raise TranscriberFailedError(
                f"{csv_path}:{line_number} has {len(row)} fields; a note row needs at "
                f"least 4 fields (start, end, pitch, velocity)"
            )
        onset, offset = float(row[0]), float(row[1])
        pitch = round(float(row[2]))  # basic-pitch writes an int; round defensively
        velocity = float(row[3])
        # Columns 5 onward are the pitch-bend list, spread one value per column.
        bend = tuple(float(v) for v in row[4:] if v.strip() != "")
        notes.append(
            NoteEvent(
                onset=onset,
                offset=max(offset, onset),
                pitch=pitch,
                confidence=min(max(velocity / MIDI_VELOCITY_SCALE, 0.0), 1.0),
                bend=bend or None,
            )
        )

    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def _model_digest(model: Path) -> bytes:
    """SHA-256 over a model's files -- one file, or a directory such as an ``.mlpackage`` --
    their paths relative to it and their bytes."""
    digest = hashlib.sha256()
    files = [model] if model.is_file() else sorted(p for p in model.rglob("*") if p.is_file())
    for path in files:
        digest.update(str(path.relative_to(model) if path != model else "").encode())
        digest.update(path.read_bytes())
    return digest.digest()


class BasicPitchCLITranscriber:
    """A :class:`~tabsampler.types.Transcriber` backed by the basic-pitch CLI."""

    def __init__(
        self,
        exe: str | Path = "basic-pitch",
        params: BasicPitchParams | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        timeout_s: float = 900.0,
        model: Path | str | None = None,
    ) -> None:
        """
        Args:
            exe: The basic-pitch executable. ``uv tool install`` puts it in
                ``~/.local/bin``, which is **not** on PATH by default, so this is
                configurable rather than assumed.
            params: Thresholds and backend. Defaults to basic-pitch's own defaults.
            cache_dir: Where parsed note-event CSVs are kept, content-addressed.
            timeout_s: Abort a single file after this long.
            model: A model for ``--model-path`` -- a fine-tuned one (ADR 0056) -- or ``None``
                for the released model. Its files, not its name, enter the cache key, and only
                when it is given, so every key made with the released model stays valid.
        """
        self.exe = _resolve_exe(exe)
        self.params = params or BasicPitchParams()
        self.cache_dir = Path(cache_dir)
        self.timeout_s = timeout_s
        self.model = Path(model) if model is not None else None
        #: How many calls were served from disk. E7 is meaningless without knowing
        #: this: a warm cache turns "transcribe and decode" into "decode".
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------ cache

    def cache_key(self, audio_path: Path) -> str:
        """SHA-256 over the audio bytes and every parameter that affects the output.

        Content-addressed, so the same audio under a different filename is one entry,
        and a changed threshold or transcriber version is a different one.
        """
        digest = hashlib.sha256()
        digest.update(audio_path.read_bytes())
        digest.update(repr(self.params).encode())
        if self.model is not None:
            # "model-path:", not the "model:" of an earlier build that ignored the model (ADR 0056),
            # so the transcriptions it cached under a model's key are never found.
            digest.update(b"model-path:" + _model_digest(self.model))
        return digest.hexdigest()[:32]

    def cache_path(self, audio_path: Path) -> Path:
        return self.cache_dir / f"{self.cache_key(audio_path)}.csv"

    # ------------------------------------------------------------------ invocation

    def build_argv(self, output_dir: Path, audio_path: Path) -> list[str]:
        """The exact command line. Positional order is ``output_dir`` then audio.

        basic-pitch ignores ``--model-path`` whenever ``--model-serialization`` is given (its
        ``predict.py``), so a model (ADR 0056) is passed alone and its type inferred from it.
        """
        model = (
            ["--model-path", str(self.model)]
            if self.model is not None
            else ["--model-serialization", self.params.model_serialization]
        )
        return [
            self.exe,
            str(output_dir),
            str(audio_path),
            "--save-note-events",
            *model,
            "--onset-threshold",
            str(self.params.onset_threshold),
            "--frame-threshold",
            str(self.params.frame_threshold),
            "--minimum-note-length",
            str(self.params.minimum_note_length_ms),
        ]

    def transcribe_file(self, audio_path: Path | str) -> list[NoteEvent]:
        """Transcribe a file, using the cache when possible.

        Raises:
            FileNotFoundError: if the audio file does not exist.
            TranscriberUnavailableError: if the executable cannot be run at all.
            TranscriberFailedError: if it exits non-zero, produces no CSV, or writes
                output that does not match the expected contract.
        """
        audio = Path(audio_path)
        if not audio.is_file():
            raise FileNotFoundError(f"no audio file at {audio}")

        cached = self.cache_path(audio)
        if cached.is_file():
            self.cache_hits += 1
            return parse_note_events_csv(cached)

        self.cache_misses += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # A fresh directory per run: basic-pitch requires the output directory to exist
        # and refuses to overwrite an existing output file.
        with tempfile.TemporaryDirectory(prefix="tabsampler-bp-") as tmp:
            out_dir = Path(tmp)
            argv = self.build_argv(out_dir, audio)
            try:
                result = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise TranscriberUnavailableError(
                    f"could not run {self.exe!r}: {exc}. Install it with "
                    f"`bash scripts/setup_transcriber.sh` and pass its path "
                    f"(uv tool install puts it in ~/.local/bin, which is not on PATH "
                    f"by default)."
                ) from exc
            except PermissionError as exc:
                raise TranscriberUnavailableError(f"{self.exe!r} is not executable: {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise TranscriberFailedError(
                    f"{self.exe} timed out after {self.timeout_s}s on {audio.name}"
                ) from exc

            if result.returncode != 0:
                raise TranscriberFailedError(
                    f"{self.exe} exit {result.returncode} on {audio.name}\n"
                    f"stderr:\n{result.stderr.strip()}"
                )

            produced = out_dir / f"{audio.stem}_basic_pitch.csv"
            if not produced.is_file():
                found = sorted(p.name for p in out_dir.iterdir())
                raise TranscriberFailedError(
                    f"{self.exe} exited 0 but wrote no note-event CSV for {audio.name}. "
                    f"Expected {produced.name}; found {found}. Returning an empty note "
                    f"list here would score as note F1 = 0.0 and read as a result."
                )

            # Move into the cache only once the run has succeeded, so a crash never
            # leaves a truncated cache entry behind.
            shutil.move(str(produced), str(cached))

        return parse_note_events_csv(cached)

    def transcribe(
        self, audio: NDArray[np.float32], sr: int = DEFAULT_SAMPLE_RATE
    ) -> list[NoteEvent]:
        """Transcribe samples, satisfying the :class:`Transcriber` protocol.

        The subprocess needs a file, so the array is written to a temporary float WAV.
        Prefer :meth:`transcribe_file` when the audio is already on disk: it caches by
        the file's own bytes and avoids a re-encode.
        """
        with tempfile.TemporaryDirectory(prefix="tabsampler-in-") as tmp:
            path = Path(tmp) / "audio.wav"
            sf.write(path, audio, sr, subtype="FLOAT")
            return self.transcribe_file(path)
