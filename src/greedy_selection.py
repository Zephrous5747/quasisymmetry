"""Greedy / Kruskal selection of additive Z-type approximate symmetries.

Selects a minimum-weight functionally independent subset of size ``n_sym``
from seniority / quartet candidates. Independence is linear independence of
parity vectors over GF(2); for additive costs this is exact (matroid greedy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.gf2_utils import gf2_int_try_add_to_span, gf2_matrix_to_int_rows


@dataclass(frozen=True)
class GreedySelectionResult:
    """Outcome of :func:`greedy_select_independent` / quota selection."""

    parity_matrix: np.ndarray
    selected_indices: tuple[int, ...]
    selected_costs: tuple[float, ...]
    selection_rule: str = "kruskal"
    n_singles: int | None = None
    n_quartets: int | None = None
    singles: tuple[int, ...] = ()
    quartets: tuple[tuple[int, int], ...] = ()

    def metadata(self, *, candidates: str, cost_function: str, parity_output: str) -> dict:
        """JSON-serializable selection record for OO output."""
        out = {
            "selection": "greedy",
            "selection_rule": self.selection_rule,
            "candidates": candidates,
            "cost_function": cost_function,
            "n_sym": int(self.parity_matrix.shape[0]),
            "selected_indices": list(self.selected_indices),
            "selected_costs": [float(c) for c in self.selected_costs],
            "parity_output": parity_output,
        }
        if self.n_singles is not None:
            out["n_singles"] = int(self.n_singles)
        if self.n_quartets is not None:
            out["n_quartets"] = int(self.n_quartets)
        if self.singles:
            out["singles"] = list(self.singles)
        if self.quartets:
            out["quartets"] = [list(edge) for edge in self.quartets]
        return out


def seniority_candidates(norb: int) -> np.ndarray:
    """Local seniority parity rows: identity ``(norb, norb)``."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    return np.eye(norb, dtype=int)


def senquart_candidates(norb: int) -> np.ndarray:
    """Seniorities plus quartets: ``{e_i} ∪ {e_i + e_j}``, shape ``(N+binom(N,2), N)``."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    rows: list[np.ndarray] = [np.eye(norb, dtype=int)[i] for i in range(norb)]
    for i in range(norb):
        for j in range(i + 1, norb):
            row = np.zeros(norb, dtype=int)
            row[i] = 1
            row[j] = 1
            rows.append(row)
    return np.asarray(rows, dtype=int)


def candidate_pool(norb: int, candidates: str = "senquart") -> np.ndarray:
    """Return the binary candidate matrix for ``seniority`` or ``senquart``."""
    kind = candidates.lower()
    if kind == "seniority":
        return seniority_candidates(norb)
    if kind == "senquart":
        return senquart_candidates(norb)
    raise ValueError("candidates must be 'senquart' or 'seniority'")


def greedy_select_independent(
    vectors: np.ndarray | Sequence[Sequence[int]],
    costs: Sequence[float],
    n_sym: int,
    *,
    prior_vectors: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> GreedySelectionResult:
    """Select ``n_sym`` GF(2)-independent rows minimizing the sum of costs.

    Scans candidates in nondecreasing cost order and keeps a row whenever it
    increases the linear span (Kruskal / matroid greedy).

    ``prior_vectors`` (optional) seeds the span with already-chosen generators;
    they are not returned. The ambient dimension limit then applies to
    ``n_sym + rank(prior)``.
    """
    mat = np.atleast_2d(np.asarray(vectors, dtype=int))
    if mat.ndim != 2:
        raise ValueError("vectors must be a 2D array of binary rows")
    n_cand, n_bits = mat.shape
    costs_arr = np.asarray(costs, dtype=float).ravel()
    if costs_arr.size != n_cand:
        raise ValueError(
            f"costs length {costs_arr.size} does not match number of "
            f"candidates {n_cand}"
        )
    if n_sym <= 0:
        raise ValueError(f"n_sym must be positive, got {n_sym}")

    rref_rows: list[int] = []
    if prior_vectors is not None:
        prior = np.atleast_2d(np.asarray(prior_vectors, dtype=int))
        if prior.ndim != 2 or prior.shape[1] != n_bits:
            raise ValueError(
                "prior_vectors must be 2D with the same bit width as vectors"
            )
        for packed_prior in gf2_matrix_to_int_rows(prior):
            if packed_prior == 0:
                continue
            new_rref = gf2_int_try_add_to_span(packed_prior, rref_rows, n_bits)
            if new_rref is None:
                raise ValueError("prior_vectors are not GF(2)-independent")
            rref_rows = new_rref

    rank_prior = len(rref_rows)
    if n_sym + rank_prior > n_bits:
        raise ValueError(
            f"n_sym={n_sym} with {rank_prior} prior generators exceeds "
            f"ambient GF(2) dimension {n_bits}"
        )
    if n_sym > n_cand:
        raise ValueError(
            f"n_sym={n_sym} exceeds candidate pool size {n_cand}"
        )

    packed = gf2_matrix_to_int_rows(mat)
    order = np.argsort(costs_arr, kind="stable")
    selected_indices: list[int] = []
    selected_costs: list[float] = []

    for idx in order:
        idx = int(idx)
        new_rref = gf2_int_try_add_to_span(packed[idx], rref_rows, n_bits)
        if new_rref is None:
            continue
        rref_rows = new_rref
        selected_indices.append(idx)
        selected_costs.append(float(costs_arr[idx]))
        if len(selected_indices) >= n_sym:
            break

    if len(selected_indices) < n_sym:
        raise ValueError(
            f"could only find {len(selected_indices)} independent candidates "
            f"(requested n_sym={n_sym})"
        )

    parity_matrix = mat[selected_indices]
    return GreedySelectionResult(
        parity_matrix=np.asarray(parity_matrix, dtype=int),
        selected_indices=tuple(selected_indices),
        selected_costs=tuple(selected_costs),
    )


def select_senquart_kruskal_from_cost_matrix(
    cost_matrix: np.ndarray,
    m: int,
    *,
    prior_singles: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Kruskal on ``{Z_i, Z_i Z_j}`` with additive weights from a cost matrix.

    ``cost_matrix[i, i]`` / ``cost_matrix[p, q]`` (``p < q``) are the weights.
    ``prior_singles`` seeds already-chosen frame axes; they are not returned.
    """
    mat = np.asarray(cost_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("cost_matrix must be a square array")
    n = int(mat.shape[0])
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    prior = tuple(sorted({int(i) for i in prior_singles}))
    for orbital in prior:
        if orbital < 0 or orbital >= n:
            raise ValueError(f"prior single {orbital} out of range for n={n}")
    if m == 0:
        return (), (), ()

    pool = senquart_candidates(n)
    costs: list[float] = []
    for row in pool:
        support = np.flatnonzero(row)
        if support.size == 1:
            costs.append(float(mat[int(support[0]), int(support[0])]))
        elif support.size == 2:
            p, q = int(support[0]), int(support[1])
            costs.append(float(mat[p, q]))
        else:
            raise RuntimeError("senquart pool produced unexpected weight")

    prior_vectors = None
    if prior:
        prior_vectors = np.eye(n, dtype=int)[list(prior)]

    result = greedy_select_independent(
        pool, costs, m, prior_vectors=prior_vectors
    )
    singles: list[int] = []
    quartets: list[tuple[int, int]] = []
    selected_costs: list[float] = []
    for idx, cost in zip(result.selected_indices, result.selected_costs):
        support = np.flatnonzero(pool[idx])
        if support.size == 1:
            singles.append(int(support[0]))
        else:
            quartets.append((int(support[0]), int(support[1])))
        selected_costs.append(float(cost))
    return tuple(singles), tuple(quartets), tuple(selected_costs)


def score_parity_rows(
    rows: np.ndarray,
    score_row: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Evaluate ``score_row`` on each candidate row."""
    rows = np.atleast_2d(np.asarray(rows, dtype=int))
    return np.asarray([float(score_row(row)) for row in rows], dtype=float)


def select_from_pool(
    norb: int,
    n_sym: int,
    score_row: Callable[[np.ndarray], float],
    candidates: str = "senquart",
) -> GreedySelectionResult:
    """Build a candidate pool, score it, and run greedy selection."""
    pool = candidate_pool(norb, candidates)
    costs = score_parity_rows(pool, score_row)
    return greedy_select_independent(pool, costs, n_sym)


def select_senquart_quota(
    norb: int,
    score_row: Callable[[np.ndarray], float],
    n_singles: int,
    n_quartets: int,
) -> GreedySelectionResult:
    """Greedy fill of fixed seniority and quartet quotas (GF(2)-independent).

    Scans sen+quartet candidates in nondecreasing cost order. A seniority is
    accepted only while fewer than ``n_singles`` singles have been kept; a
    quartet only while fewer than ``n_quartets`` quartets have been kept; and
    only if the candidate increases the GF(2) span of the growing set.
    """
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    if n_singles < 0 or n_quartets < 0:
        raise ValueError("n_singles and n_quartets must be non-negative")
    if n_singles + n_quartets <= 0:
        raise ValueError("n_singles + n_quartets must be positive")
    if n_singles + n_quartets > norb:
        raise ValueError(
            f"n_singles+n_quartets={n_singles + n_quartets} exceeds norb={norb}"
        )

    pool = senquart_candidates(norb)
    costs = score_parity_rows(pool, score_row)
    packed = gf2_matrix_to_int_rows(pool)
    order = np.argsort(costs, kind="stable")

    rref_rows: list[int] = []
    selected_indices: list[int] = []
    selected_costs: list[float] = []
    singles: list[int] = []
    quartets: list[tuple[int, int]] = []

    for idx in order:
        idx = int(idx)
        row = pool[idx]
        support = np.flatnonzero(row)
        if support.size == 1:
            if len(singles) >= n_singles:
                continue
            kind = "single"
        elif support.size == 2:
            if len(quartets) >= n_quartets:
                continue
            kind = "quartet"
        else:
            raise RuntimeError("senquart pool produced unexpected weight")

        new_rref = gf2_int_try_add_to_span(packed[idx], rref_rows, norb)
        if new_rref is None:
            continue
        rref_rows = new_rref
        selected_indices.append(idx)
        selected_costs.append(float(costs[idx]))
        if kind == "single":
            singles.append(int(support[0]))
        else:
            quartets.append((int(support[0]), int(support[1])))
        if len(singles) >= n_singles and len(quartets) >= n_quartets:
            break

    if len(singles) < n_singles:
        raise ValueError(
            f"could only select {len(singles)} independent singles "
            f"(requested {n_singles})"
        )
    if len(quartets) < n_quartets:
        raise ValueError(
            f"could only select {len(quartets)} independent quartets "
            f"(requested {n_quartets})"
        )

    return GreedySelectionResult(
        parity_matrix=np.asarray(pool[selected_indices], dtype=int),
        selected_indices=tuple(selected_indices),
        selected_costs=tuple(selected_costs),
        selection_rule="senquart_quota",
        n_singles=int(n_singles),
        n_quartets=int(n_quartets),
        singles=tuple(singles),
        quartets=tuple(quartets),
    )


def default_parity_output_path(
    outname: str | None,
    *,
    select: str = "greedy",
) -> str:
    """Default path for writing a selected parity matrix."""
    suffix = "iterative" if select == "iterative" else "greedy"
    if outname:
        stem = Path(outname).stem
        parent = Path(outname).parent
        return str(parent / f"{stem}_parity.txt")
    return f"parity_{suffix}.txt"


def write_parity_matrix(path: str | Path, parity_matrix: np.ndarray) -> str:
    """Save an integer parity matrix and return the path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_2d(parity_matrix), fmt="%d")
    return str(path)
