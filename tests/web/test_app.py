"""The local HTTP API (plan Task 2, ADR 0058)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tabsampler.config import load_phase1_config
from tabsampler.errors import TranscriberFailedError, TranscriberUnavailableError
from tabsampler.pipeline import transcribe_path
from tabsampler.types import NoteEvent
from tabsampler.web.app import DEFAULT_HOST, MAX_DURATION_S, MAX_UPLOAD_BYTES, create_app

CFG = load_phase1_config(Path("configs/decoder_clean.yaml"))

NOTES = [
    NoteEvent(onset=0.0, offset=0.4, pitch=40, confidence=0.9),
    NoteEvent(onset=0.5, offset=0.9, pitch=55, confidence=0.9),
]


class Fake:
    def __init__(
        self,
        notes: list[NoteEvent] | None = None,
        error: Exception | None = None,
        needs_file: bool = True,
    ) -> None:
        self.notes = NOTES if notes is None else notes
        self.error = error
        self.needs_file = needs_file
        self.calls: list[Path] = []

    def transcribe_file(self, path: Path, /) -> list[NoteEvent]:
        self.calls.append(path)
        assert path.is_file() or not self.needs_file, (
            "the upload must exist while the transcriber reads it"
        )
        if self.error is not None:
            raise self.error
        return list(self.notes)

    def is_available(self) -> bool:
        return self.error is None


def wav_bytes(seconds: float = 0.5, sr: int = 22050) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buffer.getvalue()


def post(client: TestClient, data: bytes, name: str = "clip.wav"):
    return client.post("/api/transcribe", files={"audio": (name, data, "audio/wav")})


@pytest.fixture
def fake() -> Fake:
    return Fake()


@pytest.fixture
def client(fake: Fake) -> TestClient:
    return TestClient(create_app(CFG, fake))


def test_health_reports_whether_the_transcriber_is_available(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "transcriber_available": True}
    down = TestClient(create_app(CFG, Fake(error=TranscriberUnavailableError("x"))))
    assert down.get("/api/health").json()["transcriber_available"] is False


def test_transcribe_returns_the_json_tab_document(client: TestClient, fake: Fake) -> None:
    r = post(client, wav_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == 1
    assert body["rhythm"] == "time_positioned_only"
    expected = transcribe_path(Path("x.wav"), CFG, Fake(needs_file=False))
    assert [(n["string"], n["fret"]) for n in body["notes"]] == [
        (t.position.string, t.position.fret) for t in expected.tab
    ]
    assert body["metrics"]["n_notes_detected"] == 2
    assert body["metrics"]["pitch_validity"] == 1.0
    assert body["uncertainty_threshold"] == CFG.uncertainty_threshold
    assert body["degradation"]["n_notes_out_of_range"] == 0


def test_the_upload_keeps_its_extension_and_is_deleted_after(
    client: TestClient, fake: Fake
) -> None:
    post(client, wav_bytes(), name="take.flac")
    (path,) = fake.calls
    assert path.suffix == ".flac"  # Basic Pitch reads by extension
    assert not path.exists()


def test_degradation_is_surfaced_to_the_client() -> None:
    low = NoteEvent(onset=1.0, offset=1.2, pitch=30, confidence=0.9)  # below the low E
    client = TestClient(create_app(CFG, Fake([*NOTES, low])))
    body = post(client, wav_bytes()).json()
    assert body["degradation"]["n_notes_out_of_range"] == 1
    assert body["degradation"]["is_clean"] is False
    assert body["degradation"]["details"]


def test_a_file_that_is_not_audio_gets_400(client: TestClient, fake: Fake) -> None:
    r = post(client, b"not audio")
    assert r.status_code == 400
    assert "could not be read as audio" in r.json()["detail"]
    assert "tabsampler-upload" not in r.json()["detail"]  # no server paths in the reply
    assert fake.calls == []


def test_a_missing_file_field_gets_422_not_500(client: TestClient) -> None:
    assert client.post("/api/transcribe").status_code == 422


def test_a_file_over_the_size_limit_gets_413(fake: Fake) -> None:
    client = TestClient(create_app(CFG, fake, max_upload_bytes=1000))
    r = post(client, wav_bytes(seconds=1.0))
    assert r.status_code == 413
    assert "larger than" in r.json()["detail"]
    assert fake.calls == []


def test_audio_longer_than_the_limit_gets_413(fake: Fake) -> None:
    client = TestClient(create_app(CFG, fake, max_duration_s=0.25))
    r = post(client, wav_bytes(seconds=0.5))
    assert r.status_code == 413
    assert "longer than" in r.json()["detail"]
    assert fake.calls == []


def test_a_missing_transcriber_gets_503_naming_the_setup_script() -> None:
    client = TestClient(create_app(CFG, Fake(error=TranscriberUnavailableError("gone"))))
    r = post(client, wav_bytes())
    assert r.status_code == 503
    assert "setup_transcriber.sh" in r.json()["detail"]


def test_a_failed_transcriber_gets_502() -> None:
    client = TestClient(create_app(CFG, Fake(error=TranscriberFailedError("exit 1"))))
    r = post(client, wav_bytes())
    assert r.status_code == 502
    assert "exit 1" in r.json()["detail"]


def test_the_limits_are_the_ones_adr_0058_states() -> None:
    assert MAX_UPLOAD_BYTES == 100 * 1024 * 1024
    assert MAX_DURATION_S == 300.0


def test_the_server_binds_to_localhost_by_default() -> None:
    # No authentication, and it runs a subprocess on uploads: never 0.0.0.0 by default.
    assert DEFAULT_HOST == "127.0.0.1"


def test_serve_without_the_web_group_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from typer.testing import CliRunner

    from tabsampler import cli

    monkeypatch.setitem(sys.modules, "uvicorn", None)  # import now raises ImportError
    result = CliRunner().invoke(cli.app, ["serve"])
    assert result.exit_code == 1
    assert "uv sync --all-groups" in " ".join(result.output.split())  # rich wraps lines


def test_serve_defaults_to_localhost() -> None:
    import inspect

    from tabsampler import cli

    assert inspect.signature(cli.serve).parameters["host"].default == DEFAULT_HOST


@pytest.mark.parametrize(
    ("fmt", "media", "suffix"),
    [
        ("musicxml", "application/vnd.recordare.musicxml+xml", ".musicxml"),
        ("gp5", "application/octet-stream", ".gp5"),
    ],
)
def test_export_turns_the_document_back_into_a_file(
    client: TestClient, fmt: str, media: str, suffix: str
) -> None:
    doc = post(client, wav_bytes()).json()
    r = client.post(f"/api/export/{fmt}", json=doc)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(media)
    assert f'filename="tab{suffix}"' in r.headers["content-disposition"]
    assert b"Rhythm is NOT transcribed" in r.content


def test_export_refuses_a_malformed_document_with_400(client: TestClient) -> None:
    r = client.post("/api/export/musicxml", json={"schema_version": 1})
    assert r.status_code == 400
    assert "not a tab document" in r.json()["detail"]


def test_export_refuses_an_unknown_format_with_404(client: TestClient) -> None:
    doc = post(client, wav_bytes()).json()
    assert client.post("/api/export/pdf", json=doc).status_code == 404


def test_an_oversized_upload_is_refused_from_its_header_before_it_is_read(fake: Fake) -> None:
    # Starlette reads a multipart body in full before the handler runs; a 2 GB WAV must be
    # refused on its Content-Length, not after it has been received.
    client = TestClient(create_app(CFG, fake, max_upload_bytes=1000))
    r = client.post(
        "/api/transcribe",
        content=b"x" * 10,
        headers={"content-length": str(10**10), "content-type": "multipart/form-data; boundary=b"},
    )
    assert r.status_code == 413
    assert fake.calls == []


def test_the_response_reports_the_tuning_offset_and_its_warning_threshold(
    client: TestClient,
) -> None:
    sharp = Path("tests/fixtures/sharp_take.wav").read_bytes()
    body = post(client, sharp).json()
    assert body["metrics"]["tuning_offset"] == pytest.approx(0.4, abs=0.05)
    assert body["tuning_warning_threshold"] == 0.25
    assert post(client, wav_bytes()).json()["metrics"]["tuning_offset"] == 0.0  # silence
