"""The learned model's network (ADR 0043): today's cost model plus a learned cost in context.

    E_t(node) = w . Phi(node) + g(h_t, psi(node))

``w`` starts at a decoder's cost weights and trains with the rest. ``h_t`` is a bidirectional
GRU's state at group ``t``, read from the groups alone
(:func:`tabsampler.model.lattice.group_inputs`).
``g`` is a small MLP on ``[h_t; psi]``; its last layer starts at zero, so a new model is exactly
the decoder it starts from, and with ``learned=False`` it stays so.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tabsampler.fingering.fit import weights_to_vector
from tabsampler.model.batch import Batch
from tabsampler.model.crf import PAD, cost_energies
from tabsampler.model.lattice import DESCRIPTOR_SIZE, GROUP_INPUT_SIZE
from tabsampler.types import CostWeights

#: ADR 0043's sizes: a 64-unit projection of each group's input, two bidirectional GRU layers
#: of 64 units, and a 64-unit hidden layer in the scorer.
WIDTH = 64
LAYERS = 2


class LearnedModel(nn.Module):
    """Node energies and transition costs for a padded batch of lattices."""

    def __init__(
        self, start: CostWeights, learned: bool = True, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.learned = learned
        # Built in ``dtype`` from the start: converting afterwards would round the cost weights.
        self.weights = nn.Parameter(torch.tensor(weights_to_vector(start), dtype=dtype))
        self.project = nn.Sequential(nn.Linear(GROUP_INPUT_SIZE, WIDTH, dtype=dtype), nn.ReLU())
        self.encoder = nn.GRU(
            WIDTH, WIDTH, num_layers=LAYERS, batch_first=True, bidirectional=True, dtype=dtype
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * WIDTH + DESCRIPTOR_SIZE, WIDTH, dtype=dtype),
            nn.ReLU(),
            nn.Linear(WIDTH, 1, dtype=dtype),
        )
        last = self.scorer[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def energies(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """``(B, T, N)`` node energies and ``(B, T - 1, N, N)`` transition costs; padded nodes
        and forbidden transitions cost ``PAD``."""
        energies, transitions = cost_energies(batch, self.weights)
        if not self.learned:
            return energies, transitions
        return torch.where(batch.nodes, energies + self.learned_term(batch), PAD), transitions

    def learned_term(self, batch: Batch) -> Tensor:
        """``g`` alone, ``(B, T, N)``, ``PAD`` on padded nodes: ADR 0043's decoding (b)."""
        context = self.context(batch)  # (B, T, 2 * WIDTH)
        width = batch.descriptors.shape[2]
        paired = torch.cat(
            [context.unsqueeze(2).expand(-1, -1, width, -1), batch.descriptors], dim=-1
        )
        learned: Tensor = self.scorer(paired).squeeze(-1)
        return torch.where(batch.nodes, learned, PAD)

    def context(self, batch: Batch) -> Tensor:
        """The encoder's state at every group, each sequence read at its own length."""
        projected: Tensor = self.project(batch.group_inputs)
        packed = pack_padded_sequence(
            projected, batch.lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        context, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=batch.group_inputs.shape[1]
        )
        return context
