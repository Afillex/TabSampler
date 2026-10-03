"""The speed limit under which a given share of human hand moves pass (ADR 0031).

Pure: no I/O, no global state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def speed_limit_for(
    pass_rate: float, speeds: Sequence[float], free: int, transitions: int
) -> float:
    """The smallest limit, in frets per second, under which ``pass_rate`` of ``transitions`` pass.

    ``speeds`` are the moves that take time; ``free`` counts transitions with no move, which
    pass at any limit. A move with no time between groups is in neither and fails at any
    limit, so it counts against the rate.

    Raises:
        ValueError: if no finite limit reaches ``pass_rate``.
    """
    needed = math.ceil(pass_rate * transitions) - free
    if needed <= 0:
        return 0.0
    if needed > len(speeds):
        raise ValueError(f"no finite limit passes {pass_rate} of {transitions} transitions")
    return sorted(speeds)[needed - 1]
