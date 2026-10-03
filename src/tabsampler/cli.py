"""Command line interface.

All I/O lives here: reading config, loading the dataset, logging test-set access,
writing ``experiments/results.csv`` and printing. ``eval/``, ``decode/`` and
``fingering/`` stay pure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tabsampler.config import EvalConfig, load_eval_config, load_phase1_config
from tabsampler.data.guitarset import (
    load_dataset,
    reference_note_arrays,
    reference_notes,
    reference_tab,
)
from tabsampler.data.splits import (
    Split,
    assert_tuning_allowed,
    guitarset_test_ids,
    guitarset_validation_ids,
    record_test_set_access,
)
from tabsampler.decode.robust import Degradation, decode_best_effort
from tabsampler.errors import TestSetMisuseError
from tabsampler.eval.harness import (
    EvalReport,
    FullReport,
    FullTrackResult,
    TrackResult,
    evaluate_full,
    evaluate_notes,
)
from tabsampler.eval.metrics import pitch_validity_rate, tab_notes_to_placed
from tabsampler.eval.playability import playability_rate
from tabsampler.eval.synthetic import round_trip_accuracy, sample_playable_path
from tabsampler.fingering.candidates import group_notes
from tabsampler.fingering.costs import HandSetScorer
from tabsampler.render.ascii import render_ascii_with_legend
from tabsampler.render.json_out import render_json
from tabsampler.results import append_row, describe_weights, m1_row_notes, results_row
from tabsampler.transcribe.basic_pitch_cli import BasicPitchCLITranscriber
from tabsampler.types import NoteEvent, TabNote

app = typer.Typer(
    add_completion=False,
    help="Turn a guitar recording into guitar tablature.",
    no_args_is_help=True,
)
console = Console()

RESULTS_PATH = Path("experiments/results.csv")


@app.command("eval-notes")
def eval_notes(
    config: Annotated[Path, typer.Option("--config", "-c", help="Experiment config YAML.")] = Path(
        "configs/phase0_eval_notes.yaml"
    ),
    results: Annotated[
        Path, typer.Option("--results", help="results.csv to append to.")
    ] = RESULTS_PATH,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Do not append a results row.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N tracks (smoke test)."),
    ] = None,
) -> None:
    """E1 note F1 for the transcriber on GuitarSet. The Phase 0 gate.

    GuitarSet is test-only (ADR 0003), so this is a deliberate look at the test set
    and is recorded in experiments/test_set_access.log.
    """
    cfg = load_eval_config(config)
    console.print(f"[bold]config[/bold] {config}")
    console.print(f"[bold]hypothesis[/bold] {cfg.hypothesis.strip()}")

    effective_limit = limit if limit is not None else cfg.limit
    track_ids = list(guitarset_test_ids())
    total = len(track_ids)
    if effective_limit is not None:
        track_ids = track_ids[:effective_limit]
        console.print(
            f"[yellow]limit={effective_limit}: evaluating {len(track_ids)} of {total} "
            f"tracks. This is a smoke test, not a result.[/yellow]"
        )

    record_test_set_access(
        f"eval-notes on {len(track_ids)} GuitarSet tracks via {config}",
    )

    dataset = load_dataset(cfg.dataset.data_home)
    # mirdata's load_tracks is untyped; this is the boundary where it becomes typed.
    tracks: dict[str, Any] = dataset.load_tracks()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    transcriber = BasicPitchCLITranscriber(
        exe=cfg.transcriber.exe,
        params=cfg.transcriber.params,
        cache_dir=cfg.transcriber.cache_dir,
    )

    def audio_path(track_id: str) -> Path:
        return Path(str(getattr(tracks[track_id], cfg.dataset.path_attribute)))

    def estimate(track_id: str) -> list[NoteEvent]:
        return transcriber.transcribe_file(audio_path(track_id))

    def duration(track_id: str) -> float:
        info = sf.info(audio_path(track_id))
        return float(info.frames) / float(info.samplerate)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("transcribing and scoring", total=len(track_ids))

        def advance(result: TrackResult, done: int, total_tracks: int) -> None:
            progress.update(
                task,
                completed=done,
                description=f"{result.track_id} (F1 {result.e1_onset.f1:.3f})",
            )

        report = evaluate_notes(
            track_ids=track_ids,
            reference=lambda t: reference_note_arrays(tracks[t]),
            estimate=estimate,
            audio_duration=duration,
            onset_tolerance=cfg.onset_tolerance,
            mode="e2e",
            on_track_done=advance,
        )

    _print_report(report, cfg)

    if dry_run:
        console.print("[yellow]--dry-run: no results row written.[/yellow]")
        return

    note = (
        f"{cfg.dataset.channel}; onset {cfg.transcriber.params.onset_threshold}; "
        f"frame {cfg.transcriber.params.frame_threshold}; "
        f"{cfg.transcriber.params.version_tag}; {len(track_ids)} tracks"
    )
    append_row(
        results,
        results_row(
            commit=_git_commit(),
            config=str(config),
            hypothesis=cfg.hypothesis,
            dataset=cfg.dataset.name,
            split=Split.TEST.value,
            mode="e2e",
            seed=cfg.seed,
            e1=report.e1_onset.f1,
            e7=report.runtime_seconds_per_audio_minute,
            notes=note,
        ),
    )
    console.print(f"[green]appended a results row to {results}[/green]")


@app.command("eval-m1")
def eval_m1(
    config: Annotated[Path, typer.Option("--config", "-c", help="Evaluation config YAML.")] = Path(
        "configs/m1_full_eval.yaml"
    ),
    decoder: Annotated[Path, typer.Option("--decoder-config", help="Decoder config YAML.")] = Path(
        "configs/decoder_clean.yaml"
    ),
    results: Annotated[
        Path, typer.Option("--results", help="results.csv to append to.")
    ] = RESULTS_PATH,
    limit: Annotated[
        int | None, typer.Option("--limit", help="First N tracks only (smoke test).")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Do not append results rows.")] = False,
) -> None:
    """E1-E5 and E7 on GuitarSet, in oracle and end-to-end mode. The M1 gate."""
    cfg = load_eval_config(config)
    dec = load_phase1_config(decoder)
    console.print(f"[bold]config[/bold] {config}  [bold]decoder[/bold] {decoder}")
    console.print(f"[bold]hypothesis[/bold] {cfg.hypothesis.strip()}")
    console.print(f"[bold]decoder weights[/bold] {describe_weights(dec.weights)}")

    track_ids = list(guitarset_test_ids())
    total = len(track_ids)
    if limit is not None:
        track_ids = track_ids[:limit]
        console.print(
            f"[yellow]limit={limit}: {len(track_ids)} of {total} tracks. "
            f"A smoke test, not a result.[/yellow]"
        )

    record_test_set_access(
        f"eval-m1 (oracle + e2e) on {len(track_ids)} GuitarSet tracks via {config}"
    )

    dataset = load_dataset(cfg.dataset.data_home)
    tracks: dict[str, Any] = dataset.load_tracks()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    transcriber = BasicPitchCLITranscriber(
        exe=cfg.transcriber.exe,
        params=cfg.transcriber.params,
        cache_dir=cfg.transcriber.cache_dir,
    )
    scorer = HandSetScorer(weights=dec.weights)

    def audio_path(track_id: str) -> Path:
        return Path(str(getattr(tracks[track_id], cfg.dataset.path_attribute)))

    def duration(track_id: str) -> float:
        info = sf.info(audio_path(track_id))
        return float(info.frames) / float(info.samplerate)

    def place(notes: list[NoteEvent]) -> tuple[list[TabNote], Degradation]:
        groups = group_notes(notes, window_s=dec.group_window_s)
        if not groups:
            return [], Degradation()
        return decode_best_effort(groups, scorer, dec.context)

    def oracle_notes(track_id: str) -> list[NoteEvent]:
        """Reference notes: measures fingering quality on its own (spec 3.2)."""
        return reference_notes(tracks[track_id])

    def e2e_notes(track_id: str) -> list[NoteEvent]:
        """Transcriber notes: what a user actually gets."""
        return transcriber.transcribe_file(audio_path(track_id))

    modes: tuple[tuple[str, Callable[[str], list[NoteEvent]]], ...] = (
        ("oracle", oracle_notes),
        ("e2e", e2e_notes),
    )

    reports: dict[str, FullReport] = {}
    for mode, notes_in in modes:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"{mode}", total=len(track_ids))

            def advance(result: FullTrackResult, done: int, _total: int, _task: Any = task) -> None:
                progress.update(
                    _task, completed=done, description=f"{result.track_id} (E2 {result.e2.f1:.3f})"
                )

            reports[mode] = evaluate_full(
                track_ids=track_ids,
                reference=lambda t: reference_note_arrays(tracks[t]),
                reference_tab=lambda t: reference_tab(tracks[t], dec.tuning),
                notes_in=notes_in,
                place=place,
                audio_duration=duration,
                tuning=dec.tuning,
                rules=dec.rules,
                onset_tolerance=cfg.onset_tolerance,
                mode=mode,
                group_window_s=dec.group_window_s,
                on_track_done=advance,
            )

    _print_m1_table(reports, decoder)
    console.print(
        f"[dim]transcriber cache: {transcriber.cache_hits} hits, "
        f"{transcriber.cache_misses} misses. E7 above is decode time when the cache is "
        f"warm, not the cost of transcribing from scratch -- the cold transcription cost "
        f"is the Phase 0 E7 in results.csv.[/dim]"
    )

    if dry_run:
        console.print("[yellow]--dry-run: no results rows written.[/yellow]")
        return

    for mode, report in reports.items():
        append_row(
            results,
            results_row(
                commit=_git_commit(),
                config=f"{config} + {decoder}",
                hypothesis=cfg.hypothesis,
                dataset=cfg.dataset.name,
                split=Split.TEST.value,
                mode=mode,
                seed=cfg.seed,
                e1=report.e1_onset.f1,
                e2=report.e2.f1,
                e3=report.e3_group_rate,
                e5=report.e5_ece,
                e7=report.runtime_seconds_per_audio_minute,
                notes=m1_row_notes(
                    channel=cfg.dataset.channel,
                    decoder=decoder,
                    weights=dec.weights,
                    e1_raw=report.e1_incoming.f1,
                    e3_transitions=report.e3_transition_rate,
                    e4=report.e4,
                    n_tracks=len(track_ids),
                    cache_hits=transcriber.cache_hits,
                    cache_misses=transcriber.cache_misses,
                ),
            ),
        )
    console.print(f"[green]appended {len(reports)} results rows to {results}[/green]")


def _print_m1_table(reports: dict[str, FullReport], decoder: Path) -> None:
    table = Table(title=f"M1 results (GuitarSet, decoder {decoder}, micro-averaged)")
    table.add_column("metric")
    for mode in reports:
        table.add_column(mode, justify="right")

    rows: list[tuple[str, Callable[[FullReport], str]]] = [
        ("E1 transcriber (raw)", lambda r: f"{r.e1_incoming.f1:.4f}"),
        ("E1 pipeline (placed)", lambda r: f"{r.e1_onset.f1:.4f}"),
        ("E2 exact tab F1", lambda r: f"{r.e2.f1:.4f}"),
        ("E3 playable groups", lambda r: f"{r.e3_group_rate:.4f}"),
        ("E3 playable transitions", lambda r: f"{r.e3_transition_rate:.4f}"),
        ("E4 pitch validity", lambda r: f"{r.e4:.4f}"),
        ("E5 calibration error", lambda r: f"{r.e5_ece:.4f}"),
        ("E7 s per audio minute", lambda r: f"{r.runtime_seconds_per_audio_minute:.2f}"),
        ("notes placed", lambda r: str(r.e2.n_est)),
        ("reference notes", lambda r: str(r.e2.n_ref)),
        ("groups span-relaxed", lambda r: str(r.n_groups_relaxed)),
        ("notes out of range", lambda r: str(r.n_notes_out_of_range)),
        ("notes dropped", lambda r: str(r.n_notes_dropped)),
    ]
    for label, render in rows:
        table.add_row(label, *(render(reports[m]) for m in reports))
    console.print(table)

    console.print(
        "[dim]Two E1 rows, deliberately. 'transcriber (raw)' scores the notes handed to "
        "the fingering stage and is the number comparable with Phase 0; 'pipeline "
        "(placed)' scores the notes that survived placement, which drops notes the guitar "
        "cannot sound and so raises precision. In oracle mode they are equal by "
        "construction.[/dim]"
    )

    oracle, e2e = reports.get("oracle"), reports.get("e2e")
    if oracle is not None and e2e is not None:
        console.print(
            f"[bold]The transcriber costs {oracle.e2.f1 - e2e.e2.f1:.4f} of E2[/bold] "
            f"(oracle {oracle.e2.f1:.4f} -> end-to-end {e2e.e2.f1:.4f}). Spec 3.2."
        )
    for mode, report in reports.items():
        if report.e4 < 1.0:
            console.print(
                f"[bold red]{mode}: E4 = {report.e4:.4f}. Below 1.0 is a bug, not a "
                f"result -- a (string, fret) pair does not sound its claimed pitch."
                f"[/bold red]"
            )
    console.print(
        "[dim]Not comparable to published GuitarSet figures, which were measured under "
        "different protocols (spec 3.4).[/dim]"
    )


@app.command("transcribe")
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Guitar audio file (isolated guitar).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write here. .json gives JSON, else ASCII."),
    ] = None,
    config: Annotated[Path, typer.Option("--config", "-c", help="Decoder config YAML.")] = Path(
        "configs/decoder_clean.yaml"
    ),
    exe: Annotated[
        str, typer.Option("--transcriber", help="basic-pitch executable.")
    ] = "basic-pitch",
) -> None:
    """audio -> tab. The end-to-end pipeline (spec 2)."""
    cfg = load_phase1_config(config)
    transcriber = BasicPitchCLITranscriber(exe=exe)

    notes = transcriber.transcribe_file(audio)
    groups = group_notes(notes, window_s=cfg.group_window_s)
    if not groups:
        console.print("[yellow]no notes detected; nothing to render.[/yellow]")
        return

    tab, degradation = decode_best_effort(groups, HandSetScorer(weights=cfg.weights), cfg.context)

    play = playability_rate(tab, cfg.rules, window_s=cfg.group_window_s)
    validity = pitch_validity_rate(tab_notes_to_placed(tab), cfg.tuning)

    if output is not None and output.suffix == ".json":
        text = render_json(tab, cfg.tuning)
    else:
        text = render_ascii_with_legend(
            tab, cfg.tuning, uncertainty_threshold=cfg.uncertainty_threshold
        )

    if output is None:
        console.print(text)
    else:
        output.write_text(text)
        console.print(f"[green]wrote {output}[/green]")

    console.print(
        f"{len(tab)} notes | E3 groups {play.group_rate:.3f}, "
        f"transitions {play.transition_rate:.3f} | E4 {validity:.3f}"
    )
    if not degradation.is_clean:
        console.print(
            f"[yellow]degraded: {degradation.n_groups_relaxed} of "
            f"{degradation.n_groups} groups needed a wider span (max "
            f"{degradation.max_span_used}); "
            f"{degradation.n_notes_out_of_range} notes outside the instrument; "
            f"{degradation.n_notes_dropped} dropped as unfingerable; "
            f"{degradation.n_groups_dropped} groups dropped entirely. "
            f"Lost notes lower recall.[/yellow]"
        )
        for line in degradation.details[:5]:
            console.print(f"  [dim]{line}[/dim]")
    if validity < 1.0:
        console.print(
            "[bold red]E4 below 1.0 is a bug, not a result: a (string, fret) pair "
            "does not sound the pitch we claimed.[/bold red]"
        )


@app.command("diagnose")
def diagnose(
    decoder: Annotated[Path, typer.Option("--decoder-config", help="Decoder config YAML.")] = Path(
        "configs/decoder_clean.yaml"
    ),
    paths: Annotated[int, typer.Option("--paths", help="How many synthetic paths.")] = 200,
    groups: Annotated[int, typer.Option("--groups", help="Shapes per path.")] = 8,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
) -> None:
    """Round-trip diagnostic: can the decoder recover a fingering it was given the pitches of?

    Needs no dataset. NOT a tuning signal -- see the warning it prints.
    """
    dec = load_phase1_config(decoder)
    rng = np.random.default_rng(seed)
    sampled = [
        sample_playable_path(rng, groups, dec.tuning, dec.rules) for _ in range(max(paths, 0))
    ]
    report = round_trip_accuracy(sampled, HandSetScorer(weights=dec.weights), dec.context)

    table = Table(title=f"Synthetic round trip ({paths} paths, {groups} groups each, seed {seed})")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_row("notes sampled", str(report.n_notes))
    table.add_row("notes recovered", str(report.n_recovered))
    table.add_row("round-trip accuracy", f"{report.accuracy:.4f}")
    table.add_row("of which single-candidate", str(report.n_single_candidate))
    console.print(table)

    free = report.n_single_candidate
    if report.n_notes:
        console.print(
            f"[dim]{free} of {report.n_notes} notes ({free / report.n_notes:.1%}) had only one "
            f"legal position, so any decoder places them correctly. That share is a floor on "
            f"the accuracy above, not a result.[/dim]"
        )
    console.print(
        "[bold yellow]This number must not tune a cost weight or the temperature.[/bold yellow] "
        "The fingerings it scores were sampled from our own notion of a plausible shape, so "
        "fitting weights to agree with them would tune the model to agree with its own prior "
        "and report the agreement as accuracy. It is a diagnostic and a regression test. The "
        "real tuning signal is human tab (ADR 0012), which is what the DadaGP/ProgGP request "
        "is for. Quote this figure with that sentence attached."
    )


@app.command("check-split")
def check_split() -> None:
    """Verify the split snapshot loads and that tuning on it is refused."""
    ids = guitarset_test_ids()
    held = guitarset_validation_ids()
    console.print(f"[green]{len(ids)} GuitarSet test tracks[/green] ({ids[0]} ... {ids[-1]})")
    console.print(
        f"[green]{len(held)} GuitarSet validation tracks[/green], player {held[0][:2]} (ADR 0037)"
    )
    try:
        assert_tuning_allowed(Split.TEST)
    except TestSetMisuseError as exc:
        console.print(f"[green]guard works:[/green] {exc}")
    else:  # pragma: no cover - would mean the guard is broken
        console.print("[red]the test-set misuse guard did NOT fire. This is a bug.[/red]")
        raise typer.Exit(1)


def _print_report(report: EvalReport, cfg: EvalConfig) -> None:
    table = Table(title="E1 note F1 (micro-averaged over notes)")
    table.add_column("variant")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ref", justify="right")
    table.add_column("est", justify="right")
    table.add_column("hit", justify="right")
    for name, prf in (
        ("onset only", report.e1_onset),
        ("onset+offset", report.e1_onset_offset),
    ):
        table.add_row(
            name,
            f"{prf.precision:.4f}",
            f"{prf.recall:.4f}",
            f"{prf.f1:.4f}",
            str(prf.n_ref),
            str(prf.n_est),
            str(prf.n_match),
        )
    console.print(table)
    console.print(
        f"tracks {len(report.tracks)} | audio {report.total_audio_seconds / 60:.1f} min | "
        f"E7 {report.runtime_seconds_per_audio_minute:.1f} s per audio minute | "
        f"channel {cfg.dataset.channel}"
    )
    f1 = report.e1_onset.f1
    if f1 <= 0.01 or f1 >= 0.99:
        console.print(
            f"[bold red]F1 = {f1:.4f} is implausible. Treat this as a bug, not a result: "
            f"near zero usually means a silent transcriber failure or a MIDI/Hz mix-up; "
            f"near one usually means the reference leaked into the estimate.[/bold red]"
        )


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


if __name__ == "__main__":  # pragma: no cover
    app()
