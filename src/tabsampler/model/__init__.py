"""The learned model of Phase 2 task C4: today's decoder plus a learned cost (ADR 0043).

Only the network and the CRF import torch, from the ``model`` dependency group (ADR 0041);
:mod:`tabsampler.model.lattice` is numpy. Weights trained here stay unpublished (ADR 0042).
"""
