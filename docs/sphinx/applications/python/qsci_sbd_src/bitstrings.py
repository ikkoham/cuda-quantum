# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Bitstring utilities shared by the QSCI and SBD-TPB paths."""


def count_alpha_beta_electrons(bitstring: str) -> tuple[int, int]:
    """Count alpha and beta electrons in CUDA-Q/OpenFermion spin-orbital order."""

    alpha = sum(char == "1" for char in bitstring[0::2])
    beta = sum(char == "1" for char in bitstring[1::2])
    return alpha, beta


def target_alpha_beta_electrons(
    electron_count: int, multiplicity: int
) -> tuple[int, int]:
    """Return alpha and beta electron counts from electron count and multiplicity."""

    ms2 = multiplicity - 1
    if (electron_count + ms2) % 2 != 0:
        raise ValueError(
            f"inconsistent electron count {electron_count} and multiplicity {multiplicity}"
        )
    n_alpha = (electron_count + ms2) // 2
    n_beta = electron_count - n_alpha
    return n_alpha, n_beta


def filter_by_alpha_beta(
    bitstrings: list[str], n_alpha: int, n_beta: int
) -> list[str]:
    """Keep determinants with the requested alpha/beta electron counts."""

    filtered = []
    for bitstring in bitstrings:
        alpha, beta = count_alpha_beta_electrons(bitstring)
        if alpha == n_alpha and beta == n_beta:
            filtered.append(bitstring)
    return filtered
