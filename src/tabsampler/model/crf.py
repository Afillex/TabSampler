"""The decoder's CRF in torch, for the learned model (ADR 0043).

Pure functions of tensors over padded batches: energies ``(B, T, N)``, transitions
``(B, T - 1, N, N)``, lengths ``(B,)``. Costs, not scores -- lower is better, as in the decoder,
and a path's probability is ``exp(-cost) / Z`` at temperature 1, as in the fitter.

Forbidden and padded entries carry ``PAD``, a cost so large that ``exp(-PAD)`` is exactly zero
in float32 and float64 alike, while every logsumexp and its gradient stay finite: an infinite
cost would turn a padded row's gradient into NaN.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tabsampler.fingering.fit import WEIGHT_NAMES
from tabsampler.model.batch import Batch

PAD = 1.0e4
MOVE = WEIGHT_NAMES.index("move")


def cost_energies(batch: Batch, weights: Tensor) -> tuple[Tensor, Tensor]:
    """The cost model's node energies and transition costs: ``w . Phi`` for each node, and
    ``move`` times the hand window's movement for each allowed transition."""
    energies = torch.where(batch.nodes, batch.features @ weights, PAD)
    transitions = torch.where(batch.allowed, weights[MOVE] * batch.movement, PAD)
    return energies, transitions


def log_partition(energies: Tensor, transitions: Tensor, lengths: Tensor) -> Tensor:
    """``log Z`` of each sequence, by the forward algorithm in log space."""
    alpha = -energies[:, 0]
    for t in range(1, energies.shape[1]):
        step = -energies[:, t] + torch.logsumexp(alpha.unsqueeze(2) - transitions[:, t - 1], dim=1)
        alpha = torch.where((t < lengths).unsqueeze(1), step, alpha)
    return torch.logsumexp(alpha, dim=1)


def path_energy(energies: Tensor, transitions: Tensor, lengths: Tensor, path: Tensor) -> Tensor:
    """The cost of one path per sequence, ``path`` holding a node per level ``(B, T)``."""
    rows = torch.arange(energies.shape[0])
    total = energies[rows, 0, path[:, 0]]
    for t in range(1, energies.shape[1]):
        step = energies[rows, t, path[:, t]] + transitions[rows, t - 1, path[:, t - 1], path[:, t]]
        total = total + torch.where(t < lengths, step, torch.zeros_like(step))
    return total


def viterbi(energies: Tensor, transitions: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    """The lowest-cost path ``(B, T)`` and its cost ``(B,)``; levels past a sequence's length
    repeat its last node."""
    size, levels, width = energies.shape
    delta = energies[:, 0]
    stay = torch.arange(width).expand(size, width)
    pointers: list[Tensor] = []
    for t in range(1, levels):
        best, arg = (delta.unsqueeze(2) + transitions[:, t - 1]).min(dim=1)
        live = (t < lengths).unsqueeze(1)
        delta = torch.where(live, energies[:, t] + best, delta)
        pointers.append(torch.where(live, arg, stay))
    cost, last = delta.min(dim=1)
    path = [last]
    for arg in reversed(pointers):
        path.append(arg.gather(1, path[-1].unsqueeze(1)).squeeze(1))
    path.reverse()
    return torch.stack(path, dim=1), cost
