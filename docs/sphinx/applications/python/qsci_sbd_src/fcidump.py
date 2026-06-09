# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""FCIDUMP file writer for the SBD-TPB executable wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyscf.tools import fcidump as pyscf_fcidump

if TYPE_CHECKING:
    from openfermion import MolecularData


def write_fcidump_from_molecule(
    path: str | Path,
    molecule: MolecularData,
    tolerance: float = 1.0e-12,
) -> None:
    """Write an FCIDUMP file from an OpenFermion-PySCF molecule."""

    mf = molecule._pyscf_data["scf"]
    pyscf_fcidump.from_scf(mf, str(path), tol=tolerance)
