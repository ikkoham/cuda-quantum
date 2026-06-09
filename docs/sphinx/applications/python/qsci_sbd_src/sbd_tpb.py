# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Python-facing SBD-TPB diagonalization helper."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

from .bitstrings import filter_by_alpha_beta, target_alpha_beta_electrons
from .fcidump import write_fcidump_from_molecule

if TYPE_CHECKING:
    from openfermion import MolecularData

SBDDeterminants = list[str] | tuple[list[str], list[str]]


@dataclass(frozen=True)
class SBDTPBResult:
    energy: float
    num_alpha_dets: int
    num_beta_dets: int
    product_dimension: int


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def _parse_sbd_executable_energy(output: str) -> float:
    energy_matches = re.findall(
        r"(?:Sample-based diagonalization:\s*)?Energy\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        output,
    )
    if not energy_matches:
        raise RuntimeError("could not parse SBD executable energy from stdout")
    return float(energy_matches[-1])


def _auto_basis_comm_sizes(mpi_processes: int) -> tuple[int, int, int]:
    task_comm_size = 1
    adet_comm_size = 1
    for factor in range(1, int(mpi_processes**0.5) + 1):
        if mpi_processes % factor == 0:
            adet_comm_size = factor
    bdet_comm_size = mpi_processes // adet_comm_size
    return task_comm_size, adet_comm_size, bdet_comm_size


def _alpha_beta_counts_from_molecule(molecule: MolecularData) -> tuple[int, int]:
    return target_alpha_beta_electrons(molecule.n_electrons, molecule.multiplicity)


def _cudaq_bitstring_to_sbd_half_dets(bitstring: str) -> tuple[str, str]:
    alpha = "".join(reversed(bitstring[0::2]))
    beta = "".join(reversed(bitstring[1::2]))
    return alpha, beta


def _split_to_sbd_half_dets(
    bitstrings: list[str],
    n_alpha: int,
    n_beta: int,
) -> tuple[list[str], list[str], list[str]]:
    filtered = filter_by_alpha_beta(bitstrings, n_alpha, n_beta)
    alpha_dets = set()
    beta_dets = set()
    for bitstring in filtered:
        alpha, beta = _cudaq_bitstring_to_sbd_half_dets(bitstring)
        alpha_dets.add(alpha)
        beta_dets.add(beta)
    return filtered, sorted(alpha_dets), sorted(beta_dets)


def _split_sbd_determinants(
    molecule: MolecularData, determinants: SBDDeterminants
) -> tuple[list[str], list[str]]:
    if isinstance(determinants, tuple):
        return determinants

    n_alpha, n_beta = _alpha_beta_counts_from_molecule(molecule)
    filtered, alpha_dets, beta_dets = _split_to_sbd_half_dets(
        determinants, n_alpha, n_beta
    )
    if not filtered:
        raise ValueError(
            "no determinants remain after alpha/beta electron-number filtering"
        )
    return alpha_dets, beta_dets


def _determinant_bit_length(alpha_dets: list[str], beta_dets: list[str]) -> int:
    return max(20, len(alpha_dets[0]), len(beta_dets[0]))


def diag_sbd_tpb(
    molecule: MolecularData,
    determinants: SBDDeterminants,
    *,
    executable: str | Path,
    mpi_processes: int = 1,
    mpirun: str = "mpirun",
    iteration: int = 8,
    block: int = 20,
    tolerance: float = 1.0e-6,
    verbose: bool = False,
) -> SBDTPBResult:
    """Run SBD-TPB from full bitstrings or an ``(alpha_dets, beta_dets)`` pair."""

    alpha_dets, beta_dets = _split_sbd_determinants(molecule, determinants)
    return _run_sbd_tpb_executable(
        molecule,
        alpha_dets,
        beta_dets,
        executable=executable,
        mpi_processes=mpi_processes,
        mpirun=mpirun,
        iteration=iteration,
        block=block,
        tolerance=tolerance,
        verbose=verbose,
    )


def _run_sbd_tpb_executable(
    molecule: MolecularData,
    alpha_dets: list[str],
    beta_dets: list[str],
    *,
    executable: str | Path,
    mpi_processes: int,
    mpirun: str,
    iteration: int,
    block: int,
    tolerance: float,
    verbose: bool,
) -> SBDTPBResult:
    if not alpha_dets or not beta_dets:
        raise ValueError("alpha and beta determinant lists must be non-empty")

    task_comm_size, adet_comm_size, bdet_comm_size = _auto_basis_comm_sizes(
        mpi_processes
    )
    h_comm_size = mpi_processes // (
        task_comm_size * adet_comm_size * bdet_comm_size
    )

    executable = Path(executable).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="qsci-sbd-tpb-") as temp_dir:
        run_dir = Path(temp_dir)
        fcidump_path = run_dir / "fcidump.txt"
        alpha_path = run_dir / "alpha_dets.txt"
        beta_path = run_dir / "beta_dets.txt"

        write_fcidump_from_molecule(fcidump_path, molecule, tolerance=1.0e-12)
        _write_text_lines(alpha_path, alpha_dets)
        _write_text_lines(beta_path, beta_dets)

        command = [
            mpirun,
            "-np",
            str(mpi_processes),
            str(executable),
            "--fcidump",
            str(fcidump_path),
            "--adetfile",
            str(alpha_path),
            "--bdetfile",
            str(beta_path),
            "--method",
            "0",
            "--block",
            str(block),
            "--iteration",
            str(iteration),
            "--tolerance",
            str(tolerance),
            "--bit_length",
            str(_determinant_bit_length(alpha_dets, beta_dets)),
            "--task_comm_size",
            str(task_comm_size),
            "--adet_comm_size",
            str(adet_comm_size),
            "--bdet_comm_size",
            str(bdet_comm_size),
            "--use_precalculated_dets",
            "0",
            "--max_memory_gb_for_determinants",
            "-1",
            "--thrust_collapse_loops",
            "1",
            "--init",
            "0",
            "--shuffle",
            "0",
            "--carryover_type",
            "0",
            "--rdm",
            "0",
        ]

        if verbose:
            print(
                "[SBD-TPB executable] "
                f"mpi_processes={mpi_processes} task_comm_size={task_comm_size} "
                f"adet_comm_size={adet_comm_size} bdet_comm_size={bdet_comm_size} "
                f"h_comm_size={h_comm_size}",
                flush=True,
            )
            print("[SBD-TPB executable]", " ".join(command), flush=True)

        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
            bufsize=1,
        )
        output_lines = []
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            if verbose:
                print(line, end="", flush=True)
        return_code = process.wait()
        output = "".join(output_lines)

        if return_code != 0:
            if not verbose:
                print(output, end="")
            raise RuntimeError(
                f"SBD executable failed with exit code {return_code}"
            )

    return SBDTPBResult(
        energy=_parse_sbd_executable_energy(output),
        num_alpha_dets=len(alpha_dets),
        num_beta_dets=len(beta_dets),
        product_dimension=len(alpha_dets) * len(beta_dets),
    )
