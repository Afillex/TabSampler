"""The local HTTP API over the pipeline (ADR 0058).

``POST /api/transcribe`` takes an audio file and returns ADR 0013's JSON document plus what
the user must be told about it: what was dropped (``degradation``), E3 and E4 (``metrics``)
and the posterior below which a note is marked uncertain. Bad uploads are refused before the
transcriber sees them, and every failure has its own status code -- never a bare 500.

Local only: there is no authentication and every upload runs a subprocess, so the server
binds ``127.0.0.1`` unless told otherwise.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from tabsampler.audio.tuning import WARNING_THRESHOLD
from tabsampler.config import Phase1Config
from tabsampler.errors import TranscriberFailedError, TranscriberUnavailableError
from tabsampler.pipeline import Hearing, PipelineResult, transcribe_path
from tabsampler.render.guitarpro import render_guitarpro
from tabsampler.render.json_out import tab_from_dict, tab_to_dict
from tabsampler.render.musicxml import render_musicxml
from tabsampler.types import NoteEvent, TabNote, Tuning

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Product limits, not tuned values (ADR 0058). Isolated guitar (ADR 0002): a riff or a take,
#: not an album side.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DURATION_S = 300.0

#: How many degradation detail lines the response carries.
MAX_DETAILS = 20

_CHUNK = 1024 * 1024

#: The page: plain HTML, CSS and JavaScript shipped inside the package (ADR 0058).
STATIC_DIR = Path(str(files("tabsampler.web").joinpath("static")))


#: Export formats (ADRs 0017, 0059): name -> (writer, media type, file suffix).
EXPORTS: dict[str, tuple[Callable[[Sequence[TabNote], Tuning], bytes], str, str]] = {
    "musicxml": (render_musicxml, "application/vnd.recordare.musicxml+xml", ".musicxml"),
    "gp5": (render_guitarpro, "application/octet-stream", ".gp5"),
}


class ServerTranscriber(Protocol):
    def transcribe_file(self, path: Path, /) -> list[NoteEvent]: ...

    def is_available(self) -> bool: ...


def audio_duration_s(path: Path) -> float:
    """The file's length from its header, without decoding it.

    Raises:
        ValueError: if the file cannot be read as audio.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # libsndfile raises its own error types; all mean "not audio"
        raise ValueError(str(exc)) from exc
    return float(info.frames) / float(info.samplerate)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _save_upload(source: BinaryIO, target: Path, limit: int) -> None:
    """Copy the upload in chunks, refusing it at the first chunk past ``limit``."""
    written = 0
    with target.open("wb") as out:
        while chunk := source.read(_CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"the file is larger than {limit / (1024 * 1024):g} MB")
            out.write(chunk)


def response_document(result: PipelineResult, cfg: Phase1Config) -> dict[str, Any]:
    """ADR 0013's document plus degradation, metrics and the uncertainty threshold."""
    d = result.degradation
    doc = tab_to_dict(result.tab, cfg.tuning)
    doc["degradation"] = {
        "is_clean": d.is_clean,
        "n_groups": d.n_groups,
        "n_groups_relaxed": d.n_groups_relaxed,
        "n_groups_dropped": d.n_groups_dropped,
        "n_notes_out_of_range": d.n_notes_out_of_range,
        "n_notes_dropped": d.n_notes_dropped,
        "max_span_used": d.max_span_used,
        "details": list(d.details[:MAX_DETAILS]),
    }
    doc["metrics"] = {
        "group_rate": result.group_rate,
        "transition_rate": result.transition_rate,
        "pitch_validity": result.pitch_validity,
        "n_notes_detected": result.n_notes_detected,
        "tuning_offset": result.tuning_offset,
    }
    doc["tuning_warning_threshold"] = WARNING_THRESHOLD
    doc["uncertainty_threshold"] = cfg.uncertainty_threshold
    return doc


def create_app(
    cfg: Phase1Config,
    transcriber: ServerTranscriber,
    *,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_duration_s: float = MAX_DURATION_S,
    electric: tuple[Phase1Config, Hearing] | None = None,
) -> FastAPI:
    """``electric``: the electric-guitar decoder and its string classifier, already loaded
    (ADR 0063); None when PyTorch or the trained weights are missing."""
    app = FastAPI(title="Tab Sampler", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Starlette reads a multipart body in full before the handler runs, so the declared size is
    # checked first; _save_upload still counts the bytes, as a header can lie. The slack covers
    # the multipart framing around the file.
    @app.middleware("http")
    async def refuse_oversized(request: Request, call_next: Any) -> Any:  # pyright: ignore[reportUnusedFunction]
        declared = request.headers.get("content-length", "")
        too_big = declared.isdigit() and int(declared) > max_upload_bytes + _CHUNK
        if request.url.path == "/api/transcribe" and too_big:
            limit_mb = max_upload_bytes / (1024 * 1024)
            return JSONResponse({"detail": f"the file is larger than {limit_mb:g} MB"}, 413)
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            # Revalidate the page and its files on every load, so an upgrade is seen at once.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return {
            "status": "ok",
            "transcriber_available": transcriber.is_available(),
            "electric_available": electric is not None,
        }

    # A plain def: FastAPI runs it in a worker thread, so the transcriber's subprocess does
    # not block the event loop.
    @app.post("/api/transcribe")
    def transcribe(  # pyright: ignore[reportUnusedFunction]
        audio: UploadFile = File(...),  # noqa: B008
        guitar: str = Form("standard"),
    ) -> dict[str, Any]:
        if guitar not in ("standard", "electric"):
            raise HTTPException(400, f"guitar must be standard or electric, not {guitar!r}")
        if guitar == "electric" and electric is None:
            raise HTTPException(
                503,
                "the electric option is not available here: it needs the `model` dependency "
                "group (PyTorch) and the trained classifier (configs/decoder_electric.yaml)",
            )
        suffix = Path(audio.filename or "").suffix.lower() or ".wav"
        tmp = Path(tempfile.mkdtemp(prefix="tabsampler-upload-"))
        try:
            path = tmp / f"upload{suffix}"
            _save_upload(audio.file, path, max_upload_bytes)
            try:
                seconds = audio_duration_s(path)
            except ValueError as exc:
                reason = str(exc).replace(str(path), audio.filename or "the upload")
                raise HTTPException(400, f"the file could not be read as audio ({reason})") from exc
            if seconds > max_duration_s:
                raise HTTPException(
                    413,
                    f"the audio is {seconds:.0f} s, longer than the {max_duration_s:.0f} s limit; "
                    f"Tab Sampler transcribes isolated guitar takes, not whole recordings",
                )
            try:
                if guitar == "electric" and electric is not None:
                    result = transcribe_path(path, electric[0], transcriber, hear=electric[1])
                else:
                    result = transcribe_path(path, cfg, transcriber)
            except TranscriberUnavailableError as exc:
                raise HTTPException(
                    503,
                    f"the transcriber is not installed or cannot run: {exc}. "
                    f"Install it with `bash scripts/setup_transcriber.sh`.",
                ) from exc
            except TranscriberFailedError as exc:
                raise HTTPException(502, f"the transcriber failed: {exc}") from exc
            doc = response_document(result, cfg)
            doc["guitar"] = guitar
            return doc
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/api/export/{fmt}")
    def export(fmt: str, doc: dict[str, Any] = Body(...)) -> Response:  # pyright: ignore[reportUnusedFunction]  # noqa: B008
        if fmt not in EXPORTS:
            raise HTTPException(404, f"no export format {fmt!r}; try {', '.join(EXPORTS)}")
        writer, media, suffix = EXPORTS[fmt]
        try:
            tab, tuning = tab_from_dict(doc)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(
            writer(tab, tuning),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="tab{suffix}"'},
        )

    return app
