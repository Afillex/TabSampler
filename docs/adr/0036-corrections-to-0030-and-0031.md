# ADR 0036: What ADRs 0030 and 0031 actually deliver

Status: accepted (2026-10-03)

Corrects two claims in accepted ADRs, found in the review of the branch that carried them.
An accepted ADR is not edited after review (the 2026-09-27 precedent), so the corrections
live here and the two status lines point to it. **No decision changes**: E3's limit stays
48 frets per second, and the hand still stretches over a wide chord.

## 1. ADR 0031: 99.86% is a share of transitions, not of hand moves

ADR 0031 set the limit so that 0.9986 of human **transitions** pass, the rate at which the
chord rules hold, and it said E3's transition rate could be quoted as "the share of moves no
faster than 99.86% of human moves". That wording is wrong. About 92% of transitions involve
no hand move at all and pass at any limit; counted over the moves where the hand actually
shifts, the limit lets through far less than 99.86%:

| human tab, artist-disjoint split | transitions passing | hand moves passing |
|---|---|---|
| training, at 12 frets/s | 16,238,538 / 16,530,064 (0.9824) | 1,069,841 / 1,361,367 (78.59%) |
| **training, at 48** | 16,508,744 / 16,530,064 (0.9987) | **1,340,047 / 1,361,367 (98.43%)** |
| validation, at 12 | 1,951,211 / 1,988,584 (0.9812) | 123,461 / 160,834 (76.76%) |
| **validation, at 48** | 1,986,282 / 1,988,584 (0.9988) | **158,532 / 160,834 (98.57%)** |

(`scripts/estimate_speed_limit.py` at commit 0b50af3; the limit and both rates are the
same as ADR 0031's run.)

**The correct quotable wording:** *the share of transitions that pass a speed limit which
human tab passes on 99.88% of transitions — 98.6% of the moves where the hand actually
shifts — on artists the limit was not taken from.* ADR 0031's premise that both halves of
E3 are then "equally strict" holds per transition and per chord shape, not per hand move:
E3 still calls about one human hand move in seventy too fast.

## 2. ADR 0030: the stretch covers a chord's notes only while it lasts

ADR 0030 said that after a wide chord "no movement is charged for a chord's own notes".
That holds only while consecutive shapes keep needing the stretch. A lone note low in the
chord relaxes the hand, by the rule ADR 0030 itself states, so `(12, 17) → 12 → 17 → 12 →
17` is charged a fret at every step after the first lone 12: three frets in all (checked
against `shift_window`). The rule stays; this is a known limitation of letting the stretch
relax.

## Consequences

The README, HANDOFF and the 2026-10-03 devlog use the corrected wording. The review also
noted that ADR 0035's Context was corrected while it was being accepted ("M1's numbers" →
"the figures after Phase 1.5") without the ADR saying so; that is recorded in the devlog
rather than by editing ADR 0035 again.
