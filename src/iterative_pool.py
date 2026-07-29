"""Interleaved GF(2)/Clifford pool extension and orbital optimization.

Each round greedily picks ``m_round`` independent weight-≤2 operators in the
current GF(2) parity frame (seniorities + quartets), pulls them back to the
original orbital basis, accumulates an independent set, optimizes the orbital
rotation for that accumulated pool, then uses the external Clifford
implementation to map the selected products to canonical single-Z axes.  The
next round's ``{Z_i, Z_i Z_j}`` pool can consequently pull back to higher-order
products in the original basis.

Scoring uses the same per-row callable as one-shot greedy (NC or variance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from openfermion import QubitOperator

from external_imports import Clifford
from src.gf2_utils import gf2_int_rref
from src.greedy_selection import (
    GreedySelectionResult,
    select_senquart_kruskal_from_cost_matrix,
)

SELECTION_RULE_ITERATIVE = "gf2_iterative_pool_extension"


def orbitals_from_mask(mask: int) -> tuple[int, ...]:
    """Sorted orbital indices set in a GF(2) support mask."""
    orbitals: list[int] = []
    value = int(mask)
    index = 0
    while value:
        if value & 1:
            orbitals.append(index)
        value >>= 1
        index += 1
    return tuple(orbitals)


def mask_from_orbitals(orbitals: Iterable[int]) -> int:
    """Bitmask of orbital support (column ``i`` ↔ bit ``i``)."""
    mask = 0
    for orbital in orbitals:
        mask |= 1 << int(orbital)
    return int(mask)


def mask_to_parity_row(mask: int, norb: int) -> np.ndarray:
    """Binary parity row of length ``norb`` for a packed mask."""
    row = np.zeros(norb, dtype=int)
    for orbital in orbitals_from_mask(mask):
        if orbital >= norb:
            raise ValueError(f"mask bit {orbital} exceeds norb={norb}")
        row[orbital] = 1
    return row


def parity_rows_from_masks(masks: Iterable[int], norb: int) -> np.ndarray:
    """Stack packed masks into a ``(k, norb)`` parity matrix."""
    rows = [mask_to_parity_row(int(mask), norb) for mask in masks]
    if not rows:
        return np.zeros((0, norb), dtype=int)
    return np.asarray(rows, dtype=int)


def gf2_rank_masks(masks: Iterable[int]) -> int:
    """GF(2) rank of packed parity-support masks."""
    rows = [int(mask) for mask in masks if int(mask)]
    if not rows:
        return 0
    n_bits = max(row.bit_length() for row in rows)
    rref, _ = gf2_int_rref(rows, n_bits)
    return len(rref)


def complete_gf2_basis(selected_masks: Iterable[int], n: int) -> list[int]:
    """Extend independent masks to a full basis of ``F_2^n`` (selected first)."""
    basis: list[int] = []
    for mask in selected_masks:
        value = int(mask)
        if value == 0:
            continue
        if gf2_rank_masks([*basis, value]) > len(basis):
            basis.append(value)
    for bit in range(n):
        candidate = 1 << bit
        if gf2_rank_masks([*basis, candidate]) > len(basis):
            basis.append(candidate)
        if len(basis) >= n:
            break
    if len(basis) != n:
        raise RuntimeError(
            f"Could only build GF(2) basis of size {len(basis)} for n={n}."
        )
    return basis


@dataclass
class Gf2ParityFrame:
    """New Z labels as GF(2) rows in the original orbital-parity basis."""

    n_spatial: int
    basis_rows: list[int] = field(default_factory=list)

    @classmethod
    def identity(cls, n_spatial: int) -> "Gf2ParityFrame":
        return cls(n_spatial=n_spatial, basis_rows=[1 << i for i in range(n_spatial)])

    def mask_for_single(self, index: int) -> int:
        return int(self.basis_rows[int(index)])

    def mask_for_quartet(self, edge: tuple[int, int]) -> int:
        p, q = int(edge[0]), int(edge[1])
        return int(self.basis_rows[p]) ^ int(self.basis_rows[q])


@dataclass(frozen=True)
class IterativeSelectionResult:
    """Outcome of :func:`select_iterative_pool`."""

    parity_matrix: np.ndarray
    accumulated_masks: tuple[int, ...]
    selected_costs: tuple[float, ...]
    history: dict[str, Any]
    optimized_parameters: np.ndarray | None = None

    def metadata(
        self,
        *,
        cost_function: str,
        parity_output: str,
        m_round: int,
    ) -> dict:
        """JSON-serializable selection record for OO output."""
        return {
            "selection": "iterative",
            "selection_rule": SELECTION_RULE_ITERATIVE,
            "candidates": "senquart_iterative",
            "cost_function": cost_function,
            "n_sym": int(self.parity_matrix.shape[0]),
            "m_round": int(m_round),
            "selected_costs": [float(c) for c in self.selected_costs],
            "accumulated_masks": [int(m) for m in self.accumulated_masks],
            "accumulated_orbitals": [
                list(orbitals_from_mask(m)) for m in self.accumulated_masks
            ],
            "rounds": self.history.get("rounds", []),
            "gf2_rank": int(self.history.get("gf2_rank", self.parity_matrix.shape[0])),
            "parity_output": parity_output,
        }


def cost_matrix_in_frame(
    frame: Gf2ParityFrame,
    score_row: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Additive cost matrix of ``{Z_i, Z_i Z_j}`` in a GF(2) parity frame."""
    n = frame.n_spatial
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        row = mask_to_parity_row(frame.mask_for_single(i), n)
        matrix[i, i] = float(score_row(row))
    for p in range(n):
        for q in range(p + 1, n):
            row = mask_to_parity_row(frame.mask_for_quartet((p, q)), n)
            matrix[p, q] = float(score_row(row))
    return matrix


def _z_operator_from_mask(mask: int) -> QubitOperator:
    support = orbitals_from_mask(mask)
    if not support:
        raise ValueError("Cannot construct a Clifford generator from identity.")
    return QubitOperator(tuple((orbital, "Z") for orbital in support), 1.0)


def _mask_from_z_operator(operator: QubitOperator, norb: int) -> int:
    if len(operator.terms) != 1:
        raise ValueError("Expected one Pauli term from Clifford inverse transform.")
    term, coefficient = next(iter(operator.terms.items()))
    if not np.isclose(abs(complex(coefficient)), 1.0, atol=1e-10):
        raise ValueError("Clifford image has a non-unit Pauli coefficient.")
    if any(pauli != "Z" for _, pauli in term):
        raise ValueError("Z-native Clifford produced a non-Z frame axis.")
    if any(qubit < 0 or qubit >= norb for qubit, _ in term):
        raise ValueError("Clifford frame axis lies outside the orbital register.")
    return mask_from_orbitals(qubit for qubit, _ in term)


def clifford_frame_from_masks(
    selected_masks: Iterable[int],
    norb: int,
) -> tuple[Gf2ParityFrame, dict[str, Any]]:
    """Canonicalize selected Z products with the external Clifford utility.

    The Clifford is synthesized on an abstract ``norb``-qubit spatial-parity
    register.  This is intentional: applying it physically to the
    Jordan--Wigner Hamiltonian would generally destroy the molecular
    Hamiltonian form expected by the orbital optimizer.  Pulling canonical
    Z-axis candidates back with ``inverse_transform`` is exactly equivalent
    for NC/variance scoring and keeps orbital optimization in its native
    representation.
    """
    masks = [int(mask) for mask in selected_masks]
    if not masks:
        frame = Gf2ParityFrame.identity(norb)
        return frame, {
            "backend": "identity",
            "basis_rows": list(frame.basis_rows),
            "symmetry_qubits": [],
            "factor_descriptions": [],
            "permutation": list(range(norb)),
        }
    if gf2_rank_masks(masks) != len(masks):
        raise ValueError("Clifford frame requires independent selected masks.")

    clifford = Clifford.from_symmetries(
        [_z_operator_from_mask(mask) for mask in masks],
        n_qubits=norb,
        symmetry_qubits_first=True,
        synthesis_basis="Z",
        generator_mapping="positive_z",
    )
    expected_qubits = tuple(range(len(masks)))
    if tuple(clifford.symmetry_qubits) != expected_qubits:
        raise RuntimeError(
            "External Clifford did not place selected generators on leading axes."
        )

    canonical_axes = [
        QubitOperator(((axis, "Z"),), 1.0) for axis in range(norb)
    ]
    basis_rows = [
        _mask_from_z_operator(clifford.inverse_transform(axis), norb)
        for axis in canonical_axes
    ]
    if gf2_rank_masks(basis_rows) != norb:
        raise RuntimeError("External Clifford pullback is not a full GF(2) basis.")
    if basis_rows[: len(masks)] != masks:
        raise RuntimeError(
            "External Clifford did not preserve selected-generator input order."
        )

    frame = Gf2ParityFrame(n_spatial=norb, basis_rows=basis_rows)
    return frame, {
        "backend": "external.QuasiSymmetries.Clifford",
        "basis_rows": [int(mask) for mask in basis_rows],
        "symmetry_qubits": [int(q) for q in clifford.symmetry_qubits],
        "mapped_qubits": [int(q) for q in clifford.mapped_qubits],
        "factor_descriptions": list(clifford.factor_descriptions),
        "permutation": [int(q) for q in clifford.permutation],
        "synthesis_basis": str(clifford.synthesis_basis),
        "generator_mapping": str(clifford.generator_mapping),
    }


def select_iterative_pool(
    norb: int,
    n_sym: int,
    score_row: Callable[[np.ndarray], float],
    *,
    m_round: int = 2,
    score_row_at: Callable[[np.ndarray, np.ndarray | None], float] | None = None,
    optimize_pool: Callable[
        [np.ndarray, np.ndarray | None, int],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    initial_parameters: np.ndarray | None = None,
) -> IterativeSelectionResult:
    """Run select -> orbital-optimize -> Clifford-canonicalize rounds.

    Parameters
    ----------
    norb:
        Number of spatial orbitals (ambient GF(2) dimension).
    n_sym:
        Target number of independent generators (``m_total``).
    score_row:
        Additive cost of a binary parity row (same metric as one-shot greedy).
    m_round:
        Operators requested per frame round (default 2).
    score_row_at:
        Optional dynamic scorer receiving the current optimized parameters.
        When omitted, the one-argument ``score_row`` is used.
    optimize_pool:
        Optional callback that optimizes the accumulated parity matrix from the
        previous parameters and returns ``(new_parameters, JSON metadata)``.
        CLI iterative mode supplies this callback, so optimization occurs after
        every selection round.
    """
    if n_sym < 0 or m_round < 1:
        raise ValueError("n_sym must be non-negative and m_round >= 1")
    if n_sym == 0:
        return IterativeSelectionResult(
            parity_matrix=np.zeros((0, norb), dtype=int),
            accumulated_masks=(),
            selected_costs=(),
            history={
                "selection_rule": SELECTION_RULE_ITERATIVE,
                "m_total": 0,
                "m_round": int(m_round),
                "accumulated_masks": [],
                "rounds": [],
                "gf2_rank": 0,
            },
            optimized_parameters=(
                None
                if initial_parameters is None
                else np.asarray(initial_parameters, dtype=float).copy()
            ),
        )
    if n_sym > norb:
        raise ValueError(f"Cannot select n_sym={n_sym} on n={norb} orbitals")

    frame = Gf2ParityFrame.identity(norb)
    accumulated: list[int] = []
    selected_costs: list[float] = []
    rounds: list[dict[str, Any]] = []
    current_parameters = (
        None
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).copy()
    )

    while gf2_rank_masks(accumulated) < n_sym:
        need = n_sym - gf2_rank_masks(accumulated)
        take = min(int(m_round), need)
        # After canonicalization, accumulated generators are frame axes 0..rank-1.
        rank = gf2_rank_masks(accumulated)
        prior_singles = tuple(range(rank))
        if score_row_at is None:
            round_scorer = score_row
        else:
            round_scorer = lambda row: score_row_at(row, current_parameters)
        cost_matrix = cost_matrix_in_frame(frame, round_scorer)
        singles, quartets, round_costs = select_senquart_kruskal_from_cost_matrix(
            cost_matrix, take, prior_singles=prior_singles
        )

        round_masks: list[int] = []
        for orbital in singles:
            round_masks.append(frame.mask_for_single(int(orbital)))
        for edge in quartets:
            round_masks.append(frame.mask_for_quartet(edge))

        added: list[int] = []
        added_costs: list[float] = []
        for mask, cost in zip(round_masks, round_costs):
            if gf2_rank_masks([*accumulated, mask]) > len(accumulated):
                accumulated.append(int(mask))
                added.append(int(mask))
                added_costs.append(float(cost))
                selected_costs.append(float(cost))

        round_record: dict[str, Any] = {
            "m_requested": take,
            "prior_singles": list(prior_singles),
            "singles": list(singles),
            "quartets": [list(edge) for edge in quartets],
            "masks": added,
            "orbitals": [list(orbitals_from_mask(mask)) for mask in added],
            "additive_cost": float(sum(round_costs)),
            "selected_costs": added_costs,
            "accumulated_rank": gf2_rank_masks(accumulated),
        }

        if not added:
            raise RuntimeError(
                "Iterative selection made no GF(2) progress; cannot reach "
                f"n_sym={n_sym}."
            )

        if optimize_pool is not None:
            parity_matrix = parity_rows_from_masks(accumulated, norb)
            current_parameters, optimization = optimize_pool(
                parity_matrix,
                current_parameters,
                len(rounds),
            )
            current_parameters = np.asarray(current_parameters, dtype=float).copy()
            round_record["optimization"] = optimization

        frame, clifford_metadata = clifford_frame_from_masks(accumulated, norb)
        round_record["clifford"] = clifford_metadata
        rounds.append(round_record)

    # Truncate to exact n_sym if a round overshot (should not with take=need).
    if len(accumulated) > n_sym:
        accumulated = accumulated[:n_sym]
        selected_costs = selected_costs[:n_sym]

    parity_matrix = parity_rows_from_masks(accumulated, norb)
    history: dict[str, Any] = {
        "selection_rule": SELECTION_RULE_ITERATIVE,
        "m_total": int(n_sym),
        "m_round": int(m_round),
        "accumulated_masks": [int(mask) for mask in accumulated],
        "accumulated_orbitals": [
            list(orbitals_from_mask(mask)) for mask in accumulated
        ],
        "rounds": rounds,
        "gf2_rank": gf2_rank_masks(accumulated),
    }
    return IterativeSelectionResult(
        parity_matrix=np.asarray(parity_matrix, dtype=int),
        accumulated_masks=tuple(int(m) for m in accumulated),
        selected_costs=tuple(selected_costs),
        history=history,
        optimized_parameters=current_parameters,
    )


def as_greedy_result(result: IterativeSelectionResult) -> GreedySelectionResult:
    """Adapt iterative output to the one-shot greedy result shape."""
    return GreedySelectionResult(
        parity_matrix=result.parity_matrix,
        selected_indices=tuple(range(result.parity_matrix.shape[0])),
        selected_costs=result.selected_costs,
    )
