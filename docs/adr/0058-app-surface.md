# ADR 0058: The app surface — a local web page served by the package, plus file exports

Status: accepted (2026-10-06) — Ege chose the app track and its local web UI (D14)

Decides **D14** (spec §7), due since M1. Plan: `docs/plans/2026-10-06-app-track.md`.

## Context

Tab Sampler has a CLI (`tabsampler transcribe`) that prints ASCII tab or writes the JSON document
of ADR 0013. The JSON already carries what ASCII cannot: exact times, each note's posterior and its
ranked alternatives. Nobody but the CLI's author will read it. The spec's cue for D14 is "CLI →
local web UI"; a DAW plugin only after a C++ port, which ADR 0015 closed as will-not-do.

Three things shape the surface and would go wrong if left implicit:

- **The pipeline shells out** to a transcriber in its own environment (ADR 0001) and takes an
  uploaded file. Anything reachable from the network would expose file upload and subprocess
  execution with no authentication.
- **The posteriors are not probabilities yet.** End to end on GuitarSet's test players (labelled,
  ADR 0055) the calibration error is 0.1841 (ADR 0057); EGDB's is not computed. A page that prints
  "84%" beside a note claims a calibration nobody measured.
- **Two front ends must agree.** If the page and the CLI each wire the pipeline, they drift, and
  the page would show a tab no number in `results.csv` describes.

## Decision

1. **A local web page, served by the package.** `tabsampler serve` starts a FastAPI app under
   uvicorn; `POST /api/transcribe` takes an audio file and returns ADR 0013's JSON document plus a
   `degradation` and a `metrics` object; `/` serves one static page. FastAPI, uvicorn and
   python-multipart sit in a **`web` dependency group**, so the CLI and the evaluation harness stay
   installable without them.
2. **Localhost only.** The default host is `127.0.0.1`. Another host needs an explicit `--host`
   and prints a warning naming what it exposes.
3. **One pipeline function.** `tabsampler.pipeline.transcribe_path(path, cfg, transcriber)` is
   what both the CLI and the server call. The page and `tabsampler transcribe` therefore agree
   note for note by construction, and a test pins it. `pipeline.py` and `web/` join `cli.py`,
   `config.py`, `results.py` and `scripts/` as the places I/O may live.
4. **Refuse bad uploads before the transcriber sees them.** Over 100 MB → 413; longer than
   5 minutes → 413 (isolated guitar, ADR 0002; a podcast would block the request for minutes);
   not readable as audio → 400; transcriber missing → 503 naming `scripts/setup_transcriber.sh`;
   transcriber failed → 502. Never a bare 500. The limits are product limits, not tuned values.
5. **Confidence is shown as a ranking, never a percentage.** Notes under the config's
   uncertainty threshold (0.6, ADR 0013) are marked uncertain by shape and by parentheses — not by
   colour alone — and the page says the confidences are uncalibrated. Hovering or focusing a note
   lists its alternatives in rank order. Degradation counts (notes out of range, dropped) appear
   in a banner, so a lost note is never invisible.
6. **No build step and nothing from the network.** Plain HTML, CSS and JavaScript drawing SVG,
   shipped inside the package. A local tool should work offline and a reader should be able to
   read the page's source.
7. **Exports per ADR 0017**, from both the CLI (`-o tab.musicxml`, `-o tab.gp5`) and the page.
   MusicXML is written with the standard library's `xml.etree` — it is one file format, and
   music21 would be a large dependency for it. Guitar Pro 5 uses **pyguitarpro**, a core
   dependency because the CLI writes it too; it is pure Python and small. ADR 0017's disclaimer
   goes in the bytes of every file.

## Alternatives considered

- **CLI only.** Leaves the posteriors and alternatives, which Phases 1–5 worked to make
  meaningful, unread by anyone who does not parse JSON.
- **A desktop app** (Tauri, Electron, PyInstaller). Packaging, signing and a second runtime for
  no capability a local page lacks.
- **A front-end framework with a build step** (React, Svelte). One page with one upload and one
  drawing does not need it, and a build step is one more thing a newcomer must install.
- **A hosted demo.** Needs the transcriber on a server, upload storage and abuse handling, and
  publishing weights trained on Guitar-TECHS is D15, still open. Revisit after D15.
- **Percentages beside notes.** Rejected for the calibration reason above.

## Consequences

**Easier.** A guitarist can drop a file on a page and read the tab with its doubts visible; the
exports let them fix rhythm by hand in MuseScore or TuxGuitar.

**Harder.** Two new dependency sets (`web`, and pyguitarpro in core) to keep resolving on 3.13;
the page has no automated browser test in CI (CI stays offline and fast), so the end-to-end check
of the page is done by hand at each change and recorded in the devlog.

**Revisit** when the end-to-end posteriors are calibrated on a test set (the wording in point 5
changes from "a ranking" to a probability), when D15 allows a hosted demo, or when full-song mode
(Phase 6) changes the duration limit.
