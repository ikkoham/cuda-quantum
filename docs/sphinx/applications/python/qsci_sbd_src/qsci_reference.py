# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Notebook-compatible selected full determinant subspace diagonalization."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy


@dataclass(frozen=True)
class ReferenceQSCIResult:
    energy: float
    dimension: int
    nonzero_elements: int


def _as_bit_array(bits: str | np.ndarray) -> np.ndarray:
    """Return a boolean bit array in the tutorial's qubit order."""

    if isinstance(bits, str):
        return np.array([char == "1" for char in bits], dtype=bool)
    return np.asarray(bits, dtype=bool)


def _cudaq_spin_operator_element(
    operator: Any, b1: str | np.ndarray, b2: str | np.ndarray
) -> complex:
    """Compute <b1|operator|b2> for a CUDA-Q SpinOperator."""

    x, z, coeffs = _cudaq_spin_operator_terms(operator)
    return _cudaq_spin_operator_element_from_terms(x, z, coeffs, b1, b2)


def _cudaq_spin_operator_terms(
    operator: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CUDA-Q SpinOperator terms once for repeated matrix elements."""

    num_qubits = operator.qubit_count
    num_terms = operator.term_count

    bsv = np.zeros((num_terms, 2 * num_qubits), dtype=bool)
    coeffs = np.zeros(num_terms, dtype=np.complex128)
    for i, term in enumerate(operator):
        coeffs[i] = term.evaluate_coefficient()
        bsf = term.get_binary_symplectic_form()
        n = len(bsf) // 2
        for j, element in enumerate(bsf):
            if j < n:
                bsv[i, j] = element
            else:
                bsv[i, j - n + num_qubits] = element

    return bsv[:, :num_qubits], bsv[:, num_qubits:], coeffs


def _cudaq_spin_operator_element_from_terms(
    x: np.ndarray,
    z: np.ndarray,
    coeffs: np.ndarray,
    b1: str | np.ndarray,
    b2: str | np.ndarray,
) -> complex:
    """Compute <b1|operator|b2> from pre-extracted CUDA-Q Pauli terms."""

    b1 = _as_bit_array(b1)
    b2 = _as_bit_array(b2)

    delta = np.bitwise_xor(b1, b2)
    match = np.all(x == delta, axis=1)
    if not np.any(match):
        return 0.0

    matched_x = x[match]
    matched_z = z[match]
    matched_coeffs = coeffs[match]

    x_dot_z = np.sum(np.bitwise_and(matched_x, matched_z), axis=1) % 4
    i_phases = np.array([1, 1j, -1, -1j], dtype=np.complex128)[x_dot_z]

    b1_dot_z_parity = np.sum(np.bitwise_and(b1, matched_z), axis=1) % 2
    sign_phases = np.where(b1_dot_z_parity == 0, 1, -1)

    return np.sum(matched_coeffs * i_phases * sign_phases)


def _openfermion_qubit_operator_element(
    operator: Any, b1: str | np.ndarray, b2: str | np.ndarray
) -> complex:
    """Compute <b1|operator|b2> for an OpenFermion QubitOperator."""

    b1 = _as_bit_array(b1)
    b2 = _as_bit_array(b2)
    value = 0.0j

    for term, coefficient in operator.terms.items():
        transformed = b2.copy()
        phase = 1.0 + 0.0j
        for qubit, pauli in term:
            ket_bit = b2[qubit]
            if pauli == "X":
                transformed[qubit] = not transformed[qubit]
            elif pauli == "Y":
                phase *= -1.0j if ket_bit else 1.0j
                transformed[qubit] = not transformed[qubit]
            elif pauli == "Z":
                if ket_bit:
                    phase *= -1.0
            else:
                raise ValueError(f"unsupported Pauli operator {pauli!r}")

        if np.array_equal(transformed, b1):
            value += complex(coefficient) * phase

    return value


def pauli_element(operator: Any, b1: str | np.ndarray, b2: str | np.ndarray) -> complex:
    """Compute the matrix element <b1|operator|b2>."""

    if hasattr(operator, "terms"):
        return _openfermion_qubit_operator_element(operator, b1, b2)
    return _cudaq_spin_operator_element(operator, b1, b2)


def diagonalize_selected_full_space(
    hamiltonian: Any,
    bitstrings: list[str],
    threshold: float = 1.0e-6,
    dense_limit: int = 256,
) -> ReferenceQSCIResult:
    """Diagonalize the Hamiltonian in the selected full-determinant subspace."""

    subspace_dimension = len(bitstrings)
    if subspace_dimension == 0:
        raise ValueError("cannot diagonalize an empty determinant subspace")

    values = []
    row_ids = []
    column_ids = []
    bit_arrays = [_as_bit_array(bits) for bits in bitstrings]
    cudaq_terms = None
    if not hasattr(hamiltonian, "terms"):
        cudaq_terms = _cudaq_spin_operator_terms(hamiltonian)

    for i in range(subspace_dimension):
        for j in range(i, subspace_dimension):
            if cudaq_terms is None:
                elem = pauli_element(hamiltonian, bit_arrays[i], bit_arrays[j])
            else:
                elem = _cudaq_spin_operator_element_from_terms(
                    *cudaq_terms, bit_arrays[i], bit_arrays[j]
                )
            if abs(elem) > threshold:
                values.append(elem)
                row_ids.append(i)
                column_ids.append(j)
                if i != j:
                    values.append(elem.conjugate())
                    row_ids.append(j)
                    column_ids.append(i)

    values = np.real_if_close(values)
    truncated_hamiltonian = scipy.sparse.coo_array(
        (values, (row_ids, column_ids)),
        shape=(subspace_dimension, subspace_dimension),
    )

    if subspace_dimension <= dense_limit:
        eigvals = scipy.linalg.eigh(
            truncated_hamiltonian.toarray(), eigvals_only=True
        )
    elif subspace_dimension > 1:
        eigvals, _ = scipy.sparse.linalg.eigsh(truncated_hamiltonian, 1, which="SA")
    else:
        eigvals = scipy.linalg.eigh(truncated_hamiltonian.toarray(), eigvals_only=True)

    return ReferenceQSCIResult(
        energy=float(np.real(eigvals[0])),
        dimension=subspace_dimension,
        nonzero_elements=len(values),
    )
