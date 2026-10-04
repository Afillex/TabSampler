"""Padded batches of lattices, as torch tensors, for the learned model (ADR 0043).

Sequences differ in length and levels in their number of nodes, so a batch pads both. ``nodes``
marks the real nodes, ``allowed`` the transitions the lattice permits between real nodes, and
``lengths`` each sequence's real levels; :mod:`tabsampler.model.crf` turns the rest into a cost
nothing can choose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from tabsampler.fingering.fit import WEIGHT_NAMES
from tabsampler.model.lattice import DESCRIPTOR_SIZE, GROUP_INPUT_SIZE, LatticeArrays


@dataclass(frozen=True, slots=True)
class Batch:
    """B sequences, padded to T levels of N nodes."""

    features: Tensor  # (B, T, N, len(WEIGHT_NAMES)): the cost model's node features
    descriptors: Tensor  # (B, T, N, DESCRIPTOR_SIZE): what the learned term sees of a node
    group_inputs: Tensor  # (B, T, GROUP_INPUT_SIZE): what the encoder reads
    movement: Tensor  # (B, T - 1, N, N): the hand window's movement
    allowed: Tensor  # (B, T - 1, N, N), bool: a transition between real nodes the lattice allows
    nodes: Tensor  # (B, T, N), bool: a real node
    lengths: Tensor  # (B,), long: real levels
    human: Tensor  # (B, T), long: the human path's node, 0 where unknown or padded


def make_batch(items: Sequence[LatticeArrays], dtype: torch.dtype = torch.float32) -> Batch:
    """Pad ``items`` into one batch.

    Raises:
        ValueError: if ``items`` is empty.
    """
    if not items:
        raise ValueError("a batch needs at least one sequence")
    size = len(items)
    levels = max(len(item.states) for item in items)
    width = max(len(level) for item in items for level in item.states)
    steps = max(levels - 1, 0)
    features = np.zeros((size, levels, width, len(WEIGHT_NAMES)))
    descriptors = np.zeros((size, levels, width, DESCRIPTOR_SIZE))
    inputs = np.zeros((size, levels, GROUP_INPUT_SIZE))
    movement = np.zeros((size, steps, width, width))
    allowed = np.zeros((size, steps, width, width), dtype=bool)
    nodes = np.zeros((size, levels, width), dtype=bool)
    lengths = np.zeros(size, dtype=np.int64)
    human = np.zeros((size, levels), dtype=np.int64)
    for i, item in enumerate(items):
        count = len(item.states)
        lengths[i] = count
        inputs[i, :count] = item.group_inputs
        for t, level in enumerate(item.states):
            features[i, t, : len(level)] = item.features[t]
            descriptors[i, t, : len(level)] = item.descriptors[t]
            nodes[i, t, : len(level)] = True
        for t, (move, ok) in enumerate(zip(item.movement, item.allowed, strict=True)):
            before, after = move.shape
            movement[i, t, :before, :after] = move
            allowed[i, t, :before, :after] = ok
        if item.human is not None:
            human[i, :count] = item.human
    return Batch(
        features=torch.tensor(features, dtype=dtype),
        descriptors=torch.tensor(descriptors, dtype=dtype),
        group_inputs=torch.tensor(inputs, dtype=dtype),
        movement=torch.tensor(movement, dtype=dtype),
        allowed=torch.tensor(allowed),
        nodes=torch.tensor(nodes),
        lengths=torch.tensor(lengths),
        human=torch.tensor(human),
    )
