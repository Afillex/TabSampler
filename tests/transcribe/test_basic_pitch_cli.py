"""Tests for the isolated basic-pitch adapter (ADR 0001).

The subprocess is faked with a stub script, so CI needs neither basic-pitch nor
Python 3.11. The CSV fixture is real output captured from basic-pitch 0.4.0.

The error-handling tests are the important ones. An empty note list scores as note
F1 = 0.0, which is indistinguishable from a genuine result -- so a missing executable,
a crash, or an absent output file must all raise.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tabsampler.errors import TranscriberFailedError, TranscriberUnavailableError
from tabsampler.transcribe.basic_pitch_cli import (
    BasicPitchCLITranscriber,
    BasicPitchParams,
    parse_note_events_csv,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "example_basic_pitch.csv"

HEADER = "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend"


def make_stub(tmp_path: Path, csv_body: str, exit_code: int = 0, stderr: str = "") -> Path:
    """A fake `basic-pitch`: writes <stem>_basic_pitch.csv into the output dir."""
    script = tmp_path / "stub-basic-pitch"
    script.write_text(
        "#!/bin/sh\n"
        'out="$1"; shift\n'
        'audio="$1"\n'
        'stem=$(basename "$audio"); stem="${stem%.*}"\n'
        f'if [ -n "{stderr}" ]; then echo "{stderr}" >&2; fi\n'
        f"if [ {exit_code} -ne 0 ]; then exit {exit_code}; fi\n"
        "cat > \"$out/${stem}_basic_pitch.csv\" <<'CSV'\n"
        f"{csv_body}\n"
        "CSV\n"
        'echo invoked >> "$out/../invocations.txt" 2>/dev/null || true\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def make_silent_stub(tmp_path: Path) -> Path:
    """A fake that exits 0 but writes no CSV at all."""
    script = tmp_path / "stub-no-output"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def wav(tmp_path: Path, name: str = "clip.wav") -> Path:
    p = tmp_path / name
    sf.write(p, np.zeros(2205, dtype=np.float32), 22050, subtype="FLOAT")
    return p


# ============================================================ parsing the real CSV


def test_parses_the_real_fixture_into_note_events() -> None:
    notes = parse_note_events_csv(FIXTURE)
    assert len(notes) == 6
    # basic-pitch found the three fundamentals E2/A2/D3 plus three harmonics.
    assert {40, 45, 50}.issubset({n.pitch for n in notes})


def test_ragged_rows_keep_every_bend_value() -> None:
    # The contract detail that a csv.DictReader would silently destroy: the header
    # names 5 columns, but save_note_events does `row.extend(pitch_bend)`, so a row
    # has 4 fields plus ONE COLUMN PER BEND VALUE. The fixture's rows are 17, 72,
    # 18, 69, 21 and 65 columns wide.
    notes = parse_note_events_csv(FIXTURE)
    bend_lengths = sorted(len(n.bend or ()) for n in notes)
    assert bend_lengths == [13, 14, 17, 61, 65, 68]


def test_velocity_is_normalised_to_zero_one() -> None:
    # basic-pitch writes int(round(127 * amplitude)); we invert it. The quantisation
    # loss is accepted and recorded in ADR 0001.
    notes = parse_note_events_csv(FIXTURE)
    by_pitch = {n.pitch: n for n in notes}
    assert by_pitch[40].confidence == pytest.approx(92 / 127)
    assert all(0.0 <= n.confidence <= 1.0 for n in notes)


def test_rows_are_sorted_by_onset() -> None:
    # basic-pitch emits rows in an unsorted order -- the fixture starts at 1.59 s and
    # ends at 0.01 s -- so the parser must sort, or grouping downstream breaks.
    onsets = [n.onset for n in parse_note_events_csv(FIXTURE)]
    assert onsets == sorted(onsets)
    assert onsets[0] < 0.1


def test_a_row_with_no_bend_values_has_bend_none(tmp_path: Path) -> None:
    p = tmp_path / "nb.csv"
    p.write_text(f"{HEADER}\n0.1,0.5,40,64\n")
    (note,) = parse_note_events_csv(p)
    assert note.bend is None
    assert note.pitch == 40


def test_a_genuinely_empty_csv_yields_no_notes(tmp_path: Path) -> None:
    # A silent clip legitimately produces no notes. Must be distinguishable from
    # the transcriber failing.
    p = tmp_path / "empty.csv"
    p.write_text(f"{HEADER}\n")
    assert parse_note_events_csv(p) == []


def test_a_malformed_row_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(f"{HEADER}\n0.1,0.5\n")
    with pytest.raises(TranscriberFailedError, match="4 fields"):
        parse_note_events_csv(p)


def test_a_csv_without_the_expected_header_raises(tmp_path: Path) -> None:
    p = tmp_path / "wrong.csv"
    p.write_text("onset,offset,midi\n0.1,0.5,40\n")
    with pytest.raises(TranscriberFailedError, match="header"):
        parse_note_events_csv(p)


# ============================================================ failure must not be silent


def test_missing_executable_raises_transcriber_unavailable(tmp_path: Path) -> None:
    t = BasicPitchCLITranscriber(
        exe="definitely-not-installed-anywhere", cache_dir=tmp_path / "cache"
    )
    with pytest.raises(TranscriberUnavailableError, match="definitely-not-installed"):
        t.transcribe_file(wav(tmp_path))


def test_nonzero_exit_raises_and_includes_stderr(tmp_path: Path) -> None:
    # The failure that must never be silent: returning [] here would score as
    # note F1 = 0.0 and read as a result rather than a bug.
    stub = make_stub(tmp_path, "", exit_code=3, stderr="model file corrupt")
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "cache")
    with pytest.raises(TranscriberFailedError) as excinfo:
        t.transcribe_file(wav(tmp_path))
    assert "model file corrupt" in str(excinfo.value)
    assert "exit 3" in str(excinfo.value)


def test_missing_output_csv_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    stub = make_silent_stub(tmp_path)
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "cache")
    with pytest.raises(TranscriberFailedError, match="no note-event CSV"):
        t.transcribe_file(wav(tmp_path))


def test_a_stub_producing_an_empty_csv_returns_empty_without_raising(tmp_path: Path) -> None:
    stub = make_stub(tmp_path, HEADER)
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "cache")
    assert t.transcribe_file(wav(tmp_path)) == []


# ============================================================ the cache


def test_a_second_call_is_served_from_cache(tmp_path: Path) -> None:
    stub = make_stub(tmp_path, f"{HEADER}\n0.1,0.5,40,64\n")
    cache = tmp_path / "cache"
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=cache)
    audio = wav(tmp_path)

    first = t.transcribe_file(audio)
    # Break the stub: a cache hit must not need it.
    stub.unlink()
    second = t.transcribe_file(audio)

    assert first == second
    assert len(first) == 1


def test_changing_a_threshold_changes_the_cache_key(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    audio = wav(tmp_path)
    a = BasicPitchCLITranscriber(
        exe="x", params=BasicPitchParams(onset_threshold=0.5), cache_dir=cache
    )
    b = BasicPitchCLITranscriber(
        exe="x", params=BasicPitchParams(onset_threshold=0.7), cache_dir=cache
    )
    assert a.cache_path(audio) != b.cache_path(audio)


def test_changing_the_version_tag_changes_the_cache_key(tmp_path: Path) -> None:
    # Otherwise a transcriber upgrade silently reuses stale results.
    cache = tmp_path / "cache"
    audio = wav(tmp_path)
    a = BasicPitchCLITranscriber(
        exe="x", params=BasicPitchParams(version_tag="0.4.0"), cache_dir=cache
    )
    b = BasicPitchCLITranscriber(
        exe="x", params=BasicPitchParams(version_tag="0.5.0"), cache_dir=cache
    )
    assert a.cache_path(audio) != b.cache_path(audio)


def test_different_audio_gets_a_different_cache_key(tmp_path: Path) -> None:
    t = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache")
    quiet = tmp_path / "a.wav"
    loud = tmp_path / "b.wav"
    sf.write(quiet, np.zeros(2205, dtype=np.float32), 22050, subtype="FLOAT")
    sf.write(loud, np.ones(2205, dtype=np.float32) * 0.5, 22050, subtype="FLOAT")
    assert t.cache_path(quiet) != t.cache_path(loud)


def test_cache_key_ignores_the_filename(tmp_path: Path) -> None:
    # Content-addressed: the same audio under two names is one cache entry.
    t = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache")
    a = wav(tmp_path, "one.wav")
    b = tmp_path / "two.wav"
    b.write_bytes(a.read_bytes())
    assert t.cache_path(a) == t.cache_path(b)


# ============================================================ awkward inputs


def test_a_filename_with_spaces_and_unicode_round_trips(tmp_path: Path) -> None:
    stub = make_stub(tmp_path, f"{HEADER}\n0.2,0.6,45,64\n")
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "cache")
    audio = wav(tmp_path, "my recording é ñ.wav")
    (note,) = t.transcribe_file(audio)
    assert note.pitch == 45


def test_a_missing_audio_file_raises_file_not_found(tmp_path: Path) -> None:
    t = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        t.transcribe_file(tmp_path / "absent.wav")


def test_transcribe_from_an_array_satisfies_the_protocol(tmp_path: Path) -> None:
    from tabsampler.types import Transcriber

    stub = make_stub(tmp_path, f"{HEADER}\n0.0,0.4,52,70\n")
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "cache")
    assert isinstance(t, Transcriber)
    (note,) = t.transcribe(np.zeros(2205, dtype=np.float32), 22050)
    assert note.pitch == 52


def test_the_output_directory_is_created_for_the_subprocess(tmp_path: Path) -> None:
    # basic-pitch's verify_output_dir raises if the directory does not already exist,
    # and build_output_path raises IOError if the target file does. Running into a
    # fresh directory avoids both.
    stub = make_stub(tmp_path, f"{HEADER}\n0.0,0.4,40,64\n")
    t = BasicPitchCLITranscriber(exe=str(stub), cache_dir=tmp_path / "deep" / "cache")
    assert len(t.transcribe_file(wav(tmp_path))) == 1
    assert (tmp_path / "deep" / "cache").is_dir()


def test_params_appear_in_the_command_line(tmp_path: Path) -> None:
    # Guards against a threshold being accepted in config but never passed through.
    t = BasicPitchCLITranscriber(
        exe="basic-pitch",
        params=BasicPitchParams(onset_threshold=0.42, frame_threshold=0.21),
        cache_dir=tmp_path,
    )
    argv = t.build_argv(Path("/tmp/out"), Path("/tmp/a.wav"))
    assert "--onset-threshold" in argv
    assert "0.42" in argv
    assert "--frame-threshold" in argv
    assert "0.21" in argv
    assert "--save-note-events" in argv
    assert argv.index("/tmp/out") < argv.index("/tmp/a.wav")


def test_environment_path_is_configurable_because_local_bin_is_not_on_path() -> None:
    # `uv tool install` puts the executable in ~/.local/bin, which is not on PATH by
    # default. The adapter must accept an explicit path rather than assume.
    assert "exe" in BasicPitchCLITranscriber.__init__.__code__.co_varnames
    _ = os  # keep the import used


# ============================================================ a fine-tuned model (ADR 0056)


def test_without_a_model_the_cache_key_is_what_it_always_was(tmp_path: Path) -> None:
    # Every transcription cached before models existed must still be found.
    import hashlib

    audio = wav(tmp_path)
    t = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache")
    digest = hashlib.sha256(audio.read_bytes())
    digest.update(repr(t.params).encode())
    assert t.cache_key(audio) == digest.hexdigest()[:32]


def test_a_model_path_appears_in_the_command_line(tmp_path: Path) -> None:
    model = tmp_path / "model.mlpackage"
    model.mkdir()
    t = BasicPitchCLITranscriber(exe="basic-pitch", cache_dir=tmp_path, model=model)
    argv = t.build_argv(Path("/tmp/out"), Path("/tmp/a.wav"))
    assert argv[argv.index("--model-path") + 1] == str(model)
    released = BasicPitchCLITranscriber(exe="basic-pitch", cache_dir=tmp_path)
    assert "--model-path" not in released.build_argv(Path("/tmp/out"), Path("/tmp/a.wav"))


def test_the_models_files_are_part_of_the_cache_key(tmp_path: Path) -> None:
    # Retraining to the same path must never reuse the old model's transcriptions.
    audio = wav(tmp_path)
    model = tmp_path / "model.mlpackage"
    (model / "Data").mkdir(parents=True)
    weights = model / "Data" / "weights.bin"
    weights.write_bytes(b"one")
    t = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache", model=model)
    first = t.cache_key(audio)
    assert first != BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache").cache_key(audio)
    weights.write_bytes(b"two")
    assert t.cache_key(audio) != first
    copy = tmp_path / "elsewhere.mlpackage"
    (copy / "Data").mkdir(parents=True)
    (copy / "Data" / "weights.bin").write_bytes(b"two")
    moved = BasicPitchCLITranscriber(exe="x", cache_dir=tmp_path / "cache", model=copy)
    assert moved.cache_key(audio) == t.cache_key(audio)  # content, not path
