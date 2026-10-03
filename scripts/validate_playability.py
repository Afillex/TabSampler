"""Validate the E3 playability rules against real human tablature (ADR 0011's owed check).

ADR 0011 says the share of real, human-made tab that passes the rules "should be close to
100%; if it is not, the rules are wrong, not the tab". This measures that share on the
DadaGP training split. It fits nothing: the rules are evaluated exactly as committed, and
what to change in response is a separate decision.

Hypothesis, written before the run: the span thresholds (4 frets below fret 12, 5 at or
above) are too strict, so well under 100% of human chord shapes pass, and span is the
dominant reason.

Rerun for ADR 0025 (2026-10-02): the transition rule is now the hand window. Acceptance,
fixed in ADR 0022 before this run: human tab passes the transition rule at >= 0.99.
If it does not, the result is recorded and reported; neither the window nor the speed
limit is tuned to pass.

Rerun for ADR 0029 (2026-10-03): a move is timed from the last fretted group, not from an
all-open group in between. Prediction, fixed in ADR 0029 before the run: the transition
pass rate rises from 0.9795 but stays below 0.99.

Crowd-sourced tab contains mistakes, so not every failure indicts a rule. The failure
breakdown is what makes the number interpretable.

    uv run python scripts/validate_playability.py data/dadagp/DadaGP-v1.1.zip \\
        data/dadagp/track_meta.json
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from tabsampler.data.dadagp import ParseStats, load_tracks
from tabsampler.data.splits import Split
from tabsampler.eval.playability import (
    PlayabilityRules,
    fingers_needed,
    group_is_playable,
    judge_transitions,
)


def main() -> None:
    archive, meta = Path(sys.argv[1]), Path(sys.argv[2])
    rules = PlayabilityRules()
    no_barre = PlayabilityRules(allow_barre=False)

    groups = passed = passed_without_barre = 0
    reasons: Counter[str] = Counter()
    spans_low: Counter[int] = Counter()  # span of human shapes whose lowest fret is below 12
    spans_high: Counter[int] = Counter()
    fingers: Counter[int] = Counter()
    transitions = transitions_passed = 0
    parse = Counter[str]()
    songs = tracks = 0

    def on_song(_: str, stats: ParseStats) -> None:
        nonlocal songs
        songs += 1
        for name in ParseStats.__dataclass_fields__:
            parse[name] += getattr(stats, name)

    # DadaGP's shipped training list: ADR 0022's and ADR 0025's figures were measured on
    # it. Nothing is fitted here, so the artist overlap (ADR 0024) does not matter.
    for track in load_tracks(archive, Split.TRAIN, meta, scheme="shipped", on_song=on_song):
        tracks += 1
        for _, state in track.steps:
            positions = state.positions
            groups += 1
            reason = group_is_playable(positions, rules)
            if reason is None:
                passed += 1
            else:
                reasons[reason.split(" ")[1] if reason[0].isdigit() else reason.split(" ")[0]] += 1
            passed_without_barre += group_is_playable(positions, no_barre) is None
            fretted = sorted(p.fret for p in positions if p.fret > 0)
            if fretted:
                span = fretted[-1] - fretted[0]
                (spans_high if fretted[0] >= rules.high_neck_fret else spans_low)[span] += 1
            fingers[fingers_needed(positions, rules)] += 1
        # The hand and the clock are carried across all-open shapes (ADR 0025, ADR 0029).
        for verdict in judge_transitions([(g.onset, s.positions) for g, s in track.steps], rules):
            transitions += 1
            transitions_passed += verdict is None

    def share(part: int, whole: int) -> str:
        return f"{part / whole:.4f}" if whole else "n/a"

    print(f"songs {songs}, tracks {tracks}, human chord shapes {groups}, transitions {transitions}")
    print(f"parse: {dict(parse)}")
    print()
    print(f"E3 groups, rules as committed (barre modelled): {share(passed, groups)}")
    print(f"E3 groups, without the barre rule (ADR 0011):   {share(passed_without_barre, groups)}")
    print(f"failure reasons: {dict(reasons.most_common())}")
    print()
    for label, spans, limit in (
        ("below fret 12", spans_low, rules.max_span_low),
        ("at fret 12 and above", spans_high, rules.max_span_high),
    ):
        total = sum(spans.values())
        over = sum(v for s, v in spans.items() if s > limit)
        cumulative = 0
        coverage: list[str] = []
        for s in sorted(spans):
            cumulative += spans[s]
            coverage.append(f"<={s}: {cumulative / total:.4f}")
        print(
            f"span {label}: {total} fretted shapes, {over} over the limit of {limit} "
            f"({share(over, total)})"
        )
        print("   cumulative share by span:", ", ".join(coverage[:10]))
    print(f"fingers needed: {dict(sorted(fingers.items()))}")
    print()
    print(
        "E3 transitions, hand window timed from the last fretted group (ADR 0029): "
        f"{share(transitions_passed, transitions)}"
    )
    print("acceptance (ADR 0022, fixed in advance): >= 0.99")


if __name__ == "__main__":
    main()
