"""Greedy / Kruskal selection of additive Z-type approximate symmetries.

Selects a minimum-weight functionally independent subset of size ``n_sym``
from seniority / quartet candidates. Independence is linear independence of
parity vectors over GF(2); for additive costs this is exact (matroid greedy).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.gf2_utils import (
    gf2_int_in_span,
    gf2_int_try_add_to_span,
    gf2_matrix_to_int_rows,
)


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
    selection_trace: tuple[dict, ...] = ()
    # Sec.10 item 3: the complete candidate library in ranked order with its NC
    # scores, so a mixed/greedy point is reproducible at candidate level rather
    # than only through the accepted rows.
    ranked_candidates: tuple[dict, ...] = ()
    # Sec.10 item 4: running GF(2) span of E u G after every acceptance.
    span_trace: tuple[dict, ...] = ()

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
        if self.selection_trace:
            out["selection_trace"] = list(self.selection_trace)
        if self.ranked_candidates:
            out["ranked_candidates"] = list(self.ranked_candidates)
            out["pool_size"] = len(self.ranked_candidates)
        if self.span_trace:
            out["span_trace"] = list(self.span_trace)
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
    record_trace: bool = False,
    top_alternatives: int = 5,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[float, ...],
    tuple[dict, ...],
]:
    """Kruskal on ``{Z_i, Z_i Z_j}`` with additive weights from a cost matrix.

    ``cost_matrix[i, i]`` / ``cost_matrix[p, q]`` (``p < q``) are the weights.
    ``prior_singles`` seeds already-chosen frame axes; they are not returned.
    When ``record_trace`` is true, also return a decision log with rejected
    near-misses around each acceptance.
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
        return (), (), (), ()

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

    trace: list[dict] = []
    if record_trace:
        selected_set = set(int(i) for i in result.selected_indices)
        order = np.argsort(np.asarray(costs, dtype=float), kind="stable")
        packed = gf2_matrix_to_int_rows(pool)
        rref_rows: list[int] = []
        if prior_vectors is not None:
            for packed_prior in gf2_matrix_to_int_rows(prior_vectors):
                if packed_prior == 0:
                    continue
                new_rref = gf2_int_try_add_to_span(packed_prior, rref_rows, n)
                if new_rref is None:
                    raise ValueError("prior_vectors are not GF(2)-independent")
                rref_rows = new_rref
        pending_rejects: list[dict] = []
        for idx in order:
            idx = int(idx)
            support = [int(x) for x in np.flatnonzero(pool[idx])]
            new_rref = gf2_int_try_add_to_span(packed[idx], rref_rows, n)
            if idx in selected_set:
                if new_rref is None:
                    raise RuntimeError("selected Kruskal candidate is GF(2)-dependent")
                trace.append(
                    {
                        "event": "accept",
                        "index": idx,
                        "support": support,
                        "cost": float(costs[idx]),
                        "near_misses_before": pending_rejects[-top_alternatives:],
                    }
                )
                rref_rows = new_rref
                pending_rejects = []
                continue
            if new_rref is None:
                pending_rejects.append(
                    {
                        "index": idx,
                        "support": support,
                        "cost": float(costs[idx]),
                        "reason": "gf2_dependent",
                    }
                )
        # Cheapest independent candidates not chosen for this round.
        start_rref: list[int] = []
        if prior_vectors is not None:
            for packed_prior in gf2_matrix_to_int_rows(prior_vectors):
                if packed_prior == 0:
                    continue
                new_rref = gf2_int_try_add_to_span(packed_prior, start_rref, n)
                if new_rref is not None:
                    start_rref = new_rref
        top_open: list[dict] = []
        for idx in order:
            idx = int(idx)
            if idx in selected_set:
                continue
            if gf2_int_try_add_to_span(packed[idx], start_rref, n) is None:
                continue
            top_open.append(
                {
                    "index": idx,
                    "support": [int(x) for x in np.flatnonzero(pool[idx])],
                    "cost": float(costs[idx]),
                }
            )
            if len(top_open) >= top_alternatives:
                break
        trace.append({"event": "top_unselected_independent", "candidates": top_open})

    return (
        tuple(singles),
        tuple(quartets),
        tuple(selected_costs),
        tuple(trace),
    )


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
    *,
    exact_masks: Sequence[int] | None = None,
) -> GreedySelectionResult:
    """Build a candidate pool, score it, and run greedy selection.

    ``exact_masks`` (optional) seeds the GF(2) span so selected LAS rows are
    independent of the exact / PG exact set (same rule as iterative).
    """
    pool = candidate_pool(norb, candidates)
    costs = score_parity_rows(pool, score_row)
    prior = None
    if exact_masks is not None:
        prior = parity_matrix_from_exact_masks(exact_masks, norb)
    return greedy_select_independent(
        pool, costs, n_sym, prior_vectors=prior
    )


def parity_matrix_from_exact_masks(
    exact_masks: Sequence[int],
    norb: int,
) -> np.ndarray:
    """Unpack nonzero exact masks into a ``(r, norb)`` binary prior matrix."""
    rows = []
    for mask in exact_masks:
        value = int(mask)
        if value == 0:
            continue
        row = np.zeros(norb, dtype=int)
        bit = 0
        while value:
            if value & 1:
                if bit >= norb:
                    raise ValueError(f"exact mask bit {bit} exceeds norb={norb}")
                row[bit] = 1
            value >>= 1
            bit += 1
        rows.append(row)
    if not rows:
        return np.zeros((0, norb), dtype=int)
    return np.asarray(rows, dtype=int)


def select_senquart_quota(
    norb: int,
    score_row: Callable[[np.ndarray], float],
    n_singles: int,
    n_quartets: int,
    *,
    disjoint_orbitals: bool = True,
    exact_masks: Sequence[int] | None = None,
) -> GreedySelectionResult:
    """Greedy fill of fixed seniority and quartet quotas (GF(2)-independent).

    Fills in two cost-ordered passes: **singles first**, then quartets. A single
    interleaved scan can accept cheap quartets before the singles quota is met
    and, with ``disjoint_orbitals`` + an exact prior (e.g. H2O ``Q_B1={4}``),
    leave too few independent singles — which is what PG-adapted NC rankings
    hit in practice.

    If the greedy singles set strands the quartet phase (common for N2 with
    exact ``Q_pix``/``Q_piy`` weight-2 rows), a small search over alternate
    single combinations completes the quotas when a feasible packing exists.

    When ``exact_masks`` is provided, the GF(2) span is seeded with those exact
    / PG rows first (LAS must be independent of ``E``, as in iterative).

    When ``disjoint_orbitals`` is true (default), a candidate is also rejected
    if its support shares any orbital with an already accepted *LAS* generator
    (forbids e.g. ``S_1`` together with ``S_{12}``). Exact rows do not mark
    orbitals as used for this orbital-disjoint check.
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
    min_orbitals = int(n_singles) + 2 * int(n_quartets)
    if disjoint_orbitals and min_orbitals > norb:
        raise ValueError(
            f"disjoint_orbitals=True needs at least {min_orbitals} orbitals "
            f"for n_singles={n_singles}, n_quartets={n_quartets} (norb={norb})"
        )

    pool = senquart_candidates(norb)
    costs = score_parity_rows(pool, score_row)
    packed = gf2_matrix_to_int_rows(pool)
    order = [int(i) for i in np.argsort(costs, kind="stable")]

    exact_rref: list[int] = []
    if exact_masks is not None:
        prior = parity_matrix_from_exact_masks(exact_masks, norb)
        for packed_prior in gf2_matrix_to_int_rows(prior):
            if packed_prior == 0:
                continue
            new_rref = gf2_int_try_add_to_span(packed_prior, exact_rref, norb)
            if new_rref is None:
                raise ValueError(
                    "exact_masks are not GF(2)-independent for Mixed prior"
                )
            exact_rref = new_rref
    rank_exact = len(exact_rref)
    if n_singles + n_quartets + rank_exact > norb:
        raise ValueError(
            f"n_singles+n_quartets={n_singles + n_quartets} with "
            f"rank(E)={rank_exact} exceeds norb={norb}"
        )

    single_idxs = [i for i in order if int(np.count_nonzero(pool[i])) == 1]
    quartet_idxs = [i for i in order if int(np.count_nonzero(pool[i])) == 2]

    def _try_pack(
        chosen_single_idxs: Sequence[int],
        *,
        note: str | None = None,
    ) -> GreedySelectionResult | None:
        rref = list(exact_rref)
        used = 0
        sel_idx: list[int] = []
        sel_cost: list[float] = []
        singles_out: list[int] = []
        quartets_out: list[tuple[int, int]] = []
        trace: list[dict] = []

        for idx in chosen_single_idxs:
            mask = int(packed[idx])
            if disjoint_orbitals and (mask & used):
                return None
            new_rref = gf2_int_try_add_to_span(mask, rref, norb)
            if new_rref is None:
                return None
            rref = new_rref
            used |= mask
            support = np.flatnonzero(pool[idx])
            sel_idx.append(idx)
            sel_cost.append(float(costs[idx]))
            singles_out.append(int(support[0]))
            ev = {
                "event": "accept",
                "kind": "single",
                "index": idx,
                "support": [int(x) for x in support],
                "cost": float(costs[idx]),
                "n_singles": len(singles_out),
                "n_quartets": 0,
                "near_misses_before": [],
            }
            if note:
                ev["repair"] = note
            trace.append(ev)

        if len(singles_out) != n_singles:
            return None

        for idx in quartet_idxs:
            if len(quartets_out) >= n_quartets:
                break
            mask = int(packed[idx])
            support_r = [int(x) for x in np.flatnonzero(pool[idx])]
            if disjoint_orbitals and (mask & used):
                trace.append({
                    "event": "reject", "reason": "orbital_overlap",
                    "kind": "quartet", "index": int(idx),
                    "support": support_r, "cost": float(costs[idx]),
                })
                continue
            new_rref = gf2_int_try_add_to_span(mask, rref, norb)
            if new_rref is None:
                trace.append({
                    "event": "reject",
                    "reason": ("exact_parity"
                               if gf2_int_in_span(mask, exact_rref)
                               else "gf2_dependent_or_exact_dressed"),
                    "kind": "quartet", "index": int(idx),
                    "support": support_r, "cost": float(costs[idx]),
                })
                continue
            rref = new_rref
            used |= mask
            support = np.flatnonzero(pool[idx])
            sel_idx.append(idx)
            sel_cost.append(float(costs[idx]))
            quartets_out.append((int(support[0]), int(support[1])))
            ev = {
                "event": "accept",
                "kind": "quartet",
                "index": idx,
                "support": [int(x) for x in support],
                "cost": float(costs[idx]),
                "n_singles": len(singles_out),
                "n_quartets": len(quartets_out),
                "near_misses_before": [],
            }
            if note:
                ev["repair"] = note
            trace.append(ev)

        if len(quartets_out) < n_quartets:
            return None

        rule = "senquart_quota_disjoint" if disjoint_orbitals else "senquart_quota"
        # Sec.10 item 3: the complete ranked library actually scored here.
        ranked = tuple(
            {
                "rank": int(pos),
                "index": int(i),
                "kind": ("single" if int(np.count_nonzero(pool[i])) == 1
                         else "quartet"),
                "support": [int(x) for x in np.flatnonzero(pool[i])],
                "cost": float(costs[i]),
                "accepted": bool(i in set(sel_idx)),
            }
            for pos, i in enumerate(order)
        )
        # Sec.10 item 4: running span of E u G after each acceptance.
        spans, running = [], list(exact_rref)
        for i in sel_idx:
            nxt = gf2_int_try_add_to_span(int(packed[i]), running, norb)
            if nxt is not None:
                running = nxt
            spans.append({
                "after_index": int(i),
                "rank": len(running),
                "span_rref": [int(v) for v in running],
            })
        return GreedySelectionResult(
            parity_matrix=np.asarray(pool[sel_idx], dtype=int),
            selected_indices=tuple(sel_idx),
            selected_costs=tuple(sel_cost),
            selection_rule=rule,
            n_singles=int(n_singles),
            n_quartets=int(n_quartets),
            singles=tuple(singles_out),
            quartets=tuple(quartets_out),
            selection_trace=tuple(trace),
            ranked_candidates=ranked,
            span_trace=tuple(spans),
        )

    greedy_singles: list[int] = []
    rref = list(exact_rref)
    used = 0
    for idx in single_idxs:
        if len(greedy_singles) >= n_singles:
            break
        mask = int(packed[idx])
        if disjoint_orbitals and (mask & used):
            continue
        new_rref = gf2_int_try_add_to_span(mask, rref, norb)
        if new_rref is None:
            continue
        rref = new_rref
        used |= mask
        greedy_singles.append(idx)

    result = _try_pack(greedy_singles)
    if result is not None:
        return result

    eligible_singles: list[int] = []
    for idx in single_idxs:
        if gf2_int_try_add_to_span(int(packed[idx]), exact_rref, norb) is not None:
            eligible_singles.append(idx)

    if n_singles == 0:
        result = _try_pack([])
        if result is not None:
            return result
    elif len(eligible_singles) >= n_singles:
        ranked_combos: list[tuple[float, tuple[int, ...]]] = []
        for combo in combinations(eligible_singles, n_singles):
            masks = [int(packed[i]) for i in combo]
            occupied = 0
            ok = True
            for m in masks:
                if disjoint_orbitals and (m & occupied):
                    ok = False
                    break
                occupied |= m
            if not ok:
                continue
            rref = list(exact_rref)
            for m in masks:
                new_rref = gf2_int_try_add_to_span(m, rref, norb)
                if new_rref is None:
                    ok = False
                    break
                rref = new_rref
            if not ok:
                continue
            ranked_combos.append((sum(float(costs[i]) for i in combo), combo))
        ranked_combos.sort(key=lambda t: t[0])
        for _cost, combo in ranked_combos:
            result = _try_pack(combo, note="single_combo_repair")
            if result is not None:
                return result

    if len(greedy_singles) < n_singles:
        raise ValueError(
            f"could only select {len(greedy_singles)} independent singles "
            f"(requested {n_singles}"
            + ("; orbital-disjoint constraint" if disjoint_orbitals else "")
            + ("; exact-parity prior" if rank_exact else "")
            + ")"
        )
    raise ValueError(
        f"could only select fewer than {n_quartets} independent quartets "
        f"(requested {n_quartets}"
        + ("; orbital-disjoint constraint" if disjoint_orbitals else "")
        + ("; exact-parity prior" if rank_exact else "")
        + ")"
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
