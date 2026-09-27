# ADR 0015: The C++ decoder port will not be done

Status: accepted (2026-09-27)

Closes spec D16 as **will not do**.

## Context

Spec D16 asked whether any part should be written in C or C++, and answered honestly that
performance was not the reason. It was carried as "an optional stretch goal after M1: port
the decoder to C++ with pybind11 and benchmark it against Python and Numba" — an exercise
rather than a response to a measurement.

Both of its justifications are now gone.

1. **Performance was already not a reason, and is now measured.** On a 188.7 s file with a
   cold cache, transcription takes 3.59 s and grouping plus decoding takes **0.11 s** —
   3% of the total, against a spec §4 budget of 60 s that the whole pipeline clears with
   16× headroom. Speeding up 3% of a task that finishes 16× inside its budget is not an
   optimisation, and spec §2.2 says to optimise only after measuring. Measured.
2. **There was never a second reason.** A port kept as an exercise has no justification
   once the measurement above exists.

## Decision

**No C or C++ in this project.** No pybind11, no Numba port of the decoder, no
Python-versus-native benchmark table.

If decode ever becomes a real bottleneck — a plausible future is Phase 6 full-song mode
with many more groups per file — the first responses are vectorising the inner loop with
numpy and tightening span pruning, both of which stay in one language. A native port is
reconsidered only if those are measured and found insufficient.

## Alternatives considered

- **Keep it as an optional stretch goal.** Rejected: an open decision that will never be
  acted on is clutter, and leaving it open implies the performance question is unsettled
  when it is measured and settled.
- **Numba instead of C++.** Rejected for the same reason — it optimises 3% of the runtime.
  Numba is already an indirect dependency via librosa; adding a hot path that depends on
  it would buy nothing measurable.

## Consequences

**Easier.** One build system, one language, one set of wheels. No cross-compilation, no
ABI, no second toolchain in CI.

**Harder.** If Phase 6 does make decoding dominant, the vectorisation work is still ahead
and will be done under time pressure rather than at leisure. The measurement to watch is
`state_count_stats` mean states per group and the decode share of total runtime; if decode
exceeds roughly a third of the total, revisit.
