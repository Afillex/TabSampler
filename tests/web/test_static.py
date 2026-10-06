"""The tab view (plan Task 3, ADR 0058): what the page must say and must not claim."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tabsampler.config import load_phase1_config
from tabsampler.types import NoteEvent
from tabsampler.web.app import create_app

CFG = load_phase1_config(Path("configs/decoder_clean.yaml"))


class Fake:
    def transcribe_file(self, path: Path, /) -> list[NoteEvent]:
        return []

    def is_available(self) -> bool:
        return True


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(CFG, Fake()))


def static(name: str) -> str:
    return files("tabsampler.web").joinpath("static", name).read_text()


def test_the_index_page_is_served_at_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<script src="/static/app.js"' in r.text


def test_the_page_states_that_confidences_are_uncalibrated(client: TestClient) -> None:
    # An end-to-end calibration error of 0.1841 (ADR 0057): a ranking, not a probability.
    text = client.get("/").text.lower()
    assert "uncalibrated" in text
    assert "0.1841" in text


def test_the_page_states_that_rhythm_is_not_transcribed(client: TestClient) -> None:
    assert "rhythm is not transcribed" in client.get("/").text.lower()  # ADR 0009


def test_the_page_states_it_is_for_isolated_guitar(client: TestClient) -> None:
    assert "isolated guitar" in client.get("/").text.lower()  # ADR 0002


def test_the_page_shows_no_percentages() -> None:
    # A posterior printed as "84%" would claim a calibration nobody measured.
    script = static("app.js")
    assert "%" not in script
    assert "* 100" not in script and "*100" not in script
    assert "toFixed" not in script  # no posterior printed as a number either


def test_uncertain_notes_are_marked_by_shape_not_colour_alone() -> None:
    css, script = static("style.css"), static("app.js")
    rule = css[css.index(".note.uncertain rect") :].split("}", 1)[0]
    assert "stroke-dasharray" in rule
    assert "`(${" in script  # the fret is wrapped in parentheses


def test_the_page_never_loads_anything_from_the_network(client: TestClient) -> None:
    for text in (client.get("/").text, static("app.js"), static("style.css")):
        text = text.replace("http://www.w3.org/2000/svg", "")  # a namespace name, not a fetch
        assert "http://" not in text and "https://" not in text


def test_static_assets_are_served(client: TestClient) -> None:
    for path, kind in (("/static/app.js", "javascript"), ("/static/style.css", "css")):
        r = client.get(path)
        assert r.status_code == 200, path
        assert kind in r.headers["content-type"]


def test_the_static_files_ship_in_the_package() -> None:
    for name in ("index.html", "app.js", "style.css"):
        assert files("tabsampler.web").joinpath("static", name).is_file()


def test_the_page_offers_both_exports(client: TestClient) -> None:
    page = client.get("/").text
    assert 'data-export="musicxml"' in page and 'data-export="gp5"' in page
    assert "/api/export/" in static("app.js")


def test_the_page_and_its_files_are_revalidated_on_every_load(client: TestClient) -> None:
    # A browser that cached last version's app.js would draw this version's data wrongly.
    for path in ("/", "/static/app.js", "/static/style.css"):
        assert client.get(path).headers["cache-control"] == "no-cache", path
