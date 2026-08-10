"""Iterative NC-ranked fixed-M LAS search with exact-parity quotienting.

Implements the revised procedure (see ``.cursor/memory/iterative_nc_ranked_las.md``):

Each macroiteration ranks all current-frame single-``Z`` and pair products by a
state-specific cost (NC / variance), accepts rows that enlarge
``span(E cup G)`` over GF(2), retains the first ``M`` accepted rows as LASs,
and continues until a complete quotient basis of ``N - r`` rows is built.  The
remaining ranked rows become auxiliary axes of the next Clifford frame.  Exact
parities (default: all-ones total particle parity) never consume the LAS budget.

This replaces the older growing-pool algorithm (add ``m_round`` generators until
raw rank ``M``, never drop earlier choices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from openfermion import QubitOperator

from external_imports import Clifford
from src.exact_parity import default_exact_masks
from src.gf2_utils import gf2_int_in_span, gf2_int_rref, gf2_int_try_add_to_span
from src.greedy_selection import GreedySelectionResult
from src.parity_rank import effective_parity_rank

SELECTION_RULE_ITERATIVE = "nc_ranked_las_fixed_m"
# Back-compat alias for older JSON / scripts.
SELECTION_RULE_ITERATIVE_LEGACY = "gf2_iterative_pool_extension"


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


def gf2_rank_masks(masks: Iterable[int], n_bits: int | None = None) -> int:
    """GF(2) rank of packed parity-support masks."""
    rows = [int(mask) for mask in masks if int(mask)]
    if not rows:
        return 0
    width = int(n_bits) if n_bits is not None else max(row.bit_length() for row in rows)
    rref, _ = gf2_int_rref(rows, width)
    return len(rref)


def span_key(masks: Iterable[int], n_bits: int) -> tuple[int, ...]:
    """Canonical RREF signature of a GF(2) span (for stopping / cycle checks)."""
    rows = [int(m) for m in masks if int(m)]
    rref, _ = gf2_int_rref(rows, n_bits)
    return tuple(sorted(int(r) for r in rref))


def complete_gf2_basis(selected_masks: Iterable[int], n: int) -> list[int]:
    """Extend independent masks to a full basis of ``F_2^n`` (selected first)."""
    basis: list[int] = []
    for mask in selected_masks:
        value = int(mask)
        if value == 0:
            continue
        if gf2_rank_masks([*basis, value], n_bits=n) > len(basis):
            basis.append(value)
    for bit in range(n):
        candidate = 1 << bit
        if gf2_rank_masks([*basis, candidate], n_bits=n) > len(basis):
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
        m_round: int | None = None,
    ) -> dict:
        """JSON-serializable selection record for OO output."""
        out = {
            "selection": "iterative",
            "selection_rule": SELECTION_RULE_ITERATIVE,
            "candidates": "senquart_iterative",
            "cost_function": cost_function,
            "n_sym": int(self.parity_matrix.shape[0]),
            "M": int(self.history.get("M", self.parity_matrix.shape[0])),
            "selected_costs": [float(c) for c in self.selected_costs],
            "accumulated_masks": [int(m) for m in self.accumulated_masks],
            "accumulated_orbitals": [
                list(orbitals_from_mask(m)) for m in self.accumulated_masks
            ],
            "las_masks": [int(m) for m in self.history.get("las_masks", self.accumulated_masks)],
            "auxiliary_masks": [int(m) for m in self.history.get("auxiliary_masks", [])],
            "ranked_basis_masks": [
                int(m) for m in self.history.get("ranked_basis_masks", [])
            ],
            "exact_masks": [int(m) for m in self.history.get("exact_masks", [])],
            "exact_rank": self.history.get("exact_rank"),
            "exact_sector": self.history.get("exact_sector"),
            "rounds": self.history.get("rounds", []),
            "gf2_rank": int(self.history.get("gf2_rank", self.parity_matrix.shape[0])),
            "M_eff": self.history.get("M_eff"),
            "stop_reason": self.history.get("stop_reason"),
            "parity_output": parity_output,
        }
        if m_round is not None:
            out["m_round_deprecated"] = int(m_round)
        return out


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


def _frame_candidates(
    frame: Gf2ParityFrame,
    cost_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """Enumerate singles/pairs with costs (stable order for tie-breaks)."""
    n = frame.n_spatial
    cands: list[dict[str, Any]] = []
    for i in range(n):
        cands.append(
            {
                "kind": "single",
                "frame_indices": (i,),
                "mask": int(frame.mask_for_single(i)),
                "cost": float(cost_matrix[i, i]),
                "order_key": (i, i),
            }
        )
    for p in range(n):
        for q in range(p + 1, n):
            cands.append(
                {
                    "kind": "quartet",
                    "frame_indices": (p, q),
                    "mask": int(frame.mask_for_quartet((p, q))),
                    "cost": float(cost_matrix[p, q]),
                    "order_key": (p, q),
                }
            )
    # Ascending NC; deterministic tie-break by frame indices.
    cands.sort(key=lambda c: (c["cost"], c["order_key"]))
    return cands


def ranked_quotient_scan(
    frame: Gf2ParityFrame,
    cost_matrix: np.ndarray,
    *,
    M: int,
    exact_masks: Sequence[int],
) -> dict[str, Any]:
    """NC-ranked scan with exact-parity quotient independence.

    Returns LAS pool (first ``M`` accepted), auxiliary rows, full ranked
    quotient basis ``B`` of size ``N - r``, and a decision trace.
    """
    n = frame.n_spatial
    exact = [int(m) for m in exact_masks if int(m)]
    rref_exact, _ = gf2_int_rref(exact, n)
    r = len(rref_exact)
    if M < 0 or M > n - r:
        raise ValueError(f"M={M} out of range for N={n}, r={r} (need 0 <= M <= N-r)")

    target = n - r
    accepted: list[int] = []
    accepted_costs: list[float] = []
    rref = list(rref_exact)
    trace: list[dict[str, Any]] = []

    for cand in _frame_candidates(frame, cost_matrix):
        mask = int(cand["mask"])
        if mask == 0:
            trace.append({**cand, "event": "reject", "reason": "identity"})
            continue
        new_rref = gf2_int_try_add_to_span(mask, rref, n)
        if new_rref is None:
            if gf2_int_in_span(mask, rref_exact):
                reason = "exact_parity"
            else:
                reason = "gf2_dependent_or_exact_dressed"
            trace.append(
                {
                    "event": "reject",
                    "reason": reason,
                    "kind": cand["kind"],
                    "frame_indices": list(cand["frame_indices"]),
                    "support": list(orbitals_from_mask(mask)),
                    "cost": cand["cost"],
                    "mask": mask,
                }
            )
            continue

        rref = new_rref
        accepted.append(mask)
        accepted_costs.append(float(cand["cost"]))
        event = "accept_las" if len(accepted) <= M else "accept_auxiliary"
        trace.append(
            {
                "event": event,
                "kind": cand["kind"],
                "frame_indices": list(cand["frame_indices"]),
                "support": list(orbitals_from_mask(mask)),
                "cost": cand["cost"],
                "mask": mask,
                "accepted_index": len(accepted) - 1,
            }
        )
        if len(accepted) >= target:
            break

    if len(accepted) < target:
        raise RuntimeError(
            f"Ranked quotient scan accepted only {len(accepted)} of {target} "
            f"rows (N={n}, r={r}, M={M})."
        )

    las = accepted[:M]
    aux = accepted[M:]
    return {
        "exact_masks": exact,
        "exact_rank": r,
        "ranked_basis_masks": accepted,
        "las_masks": las,
        "auxiliary_masks": aux,
        "las_costs": accepted_costs[:M],
        "ranked_costs": accepted_costs,
        "selection_trace": trace,
        "N": n,
        "M": M,
        "r": r,
    }


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
    register.  Pulling canonical Z-axis candidates back with
    ``inverse_transform`` keeps orbital optimization in its native
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
    if gf2_rank_masks(masks, n_bits=norb) != len(masks):
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
    if gf2_rank_masks(basis_rows, n_bits=norb) != norb:
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
    m_round: int | None = None,
    max_macroiterations: int | None = None,
    exact_masks: Sequence[int] | None = None,
    exact_sector: Sequence[int] | None = None,
    score_row_at: Callable[[np.ndarray, np.ndarray | None], float] | None = None,
    optimize_pool: Callable[
        [np.ndarray, np.ndarray | None, int],
        tuple[np.ndarray, dict[str, Any]],
    ]
    | None = None,
    initial_parameters: np.ndarray | None = None,
    stable_span_iters: int = 2,
    oo_stop_tol: float = 1e-6,
    oo_step_tol: float = 1e-5,
    rank_replace_tol: float = 1e-4,
) -> IterativeSelectionResult:
    """Fixed-``M`` NC-ranked LAS search with exact-parity quotienting.

    Parameters
    ----------
    norb:
        Ambient GF(2) dimension ``N`` (spatial orbitals).
    n_sym:
        LAS budget ``M`` (not raw growing-pool size).
    score_row / score_row_at:
        Candidate costs (NC or variance); lower is better.
    m_round:
        Deprecated (ignored). Kept so older CLI call sites do not break.
    max_macroiterations:
        Cap on discrete frame updates (default ``max(2*N, 8)``).
    exact_masks:
        Exact parity rows ``E``; default all-ones total particle parity.
    exact_sector:
        Optional exact eigenvalue label recorded in history.
    optimize_pool:
        Optional OO callback on the current LAS parity matrix.
    stable_span_iters:
        Stop after this many consecutive macroiterations where all five
        document stop conditions hold.
    oo_stop_tol / oo_step_tol / rank_replace_tol:
        Tolerances for orbital objective, orbital step, and ranking
        replacement significance.
    """
    if m_round is not None and int(m_round) < 1:
        raise ValueError("m_round must be >= 1 when provided (deprecated flag)")
    if n_sym < 0:
        raise ValueError("n_sym (M) must be non-negative")
    if n_sym == 0:
        return IterativeSelectionResult(
            parity_matrix=np.zeros((0, norb), dtype=int),
            accumulated_masks=(),
            selected_costs=(),
            history={
                "selection_rule": SELECTION_RULE_ITERATIVE,
                "M": 0,
                "m_total": 0,
                "accumulated_masks": [],
                "las_masks": [],
                "auxiliary_masks": [],
                "ranked_basis_masks": [],
                "exact_masks": list(exact_masks or default_exact_masks(norb)),
                "exact_rank": gf2_rank_masks(
                    exact_masks or default_exact_masks(norb), n_bits=norb
                ),
                "exact_sector": (
                    None if exact_sector is None else [int(b) for b in exact_sector]
                ),
                "rounds": [],
                "gf2_rank": 0,
                "M_eff": 0,
                "stop_reason": "empty_M",
            },
            optimized_parameters=(
                None
                if initial_parameters is None
                else np.asarray(initial_parameters, dtype=float).copy()
            ),
        )

    exact = list(exact_masks) if exact_masks is not None else list(default_exact_masks(norb))
    r = gf2_rank_masks(exact, n_bits=norb)
    if n_sym > norb - r:
        raise ValueError(
            f"Cannot select M={n_sym} LASs on N={norb} with exact rank r={r} "
            f"(need M <= N-r={norb - r})"
        )

    max_macro = (
        int(max_macroiterations)
        if max_macroiterations is not None
        else max(2 * norb, 8)
    )
    if max_macro < 1:
        raise ValueError("max_macroiterations must be >= 1")
    if int(stable_span_iters) < 1:
        raise ValueError("stable_span_iters must be >= 1")

    frame = Gf2ParityFrame.identity(norb)
    las: list[int] = []
    las_costs: list[float] = []
    aux: list[int] = []
    ranked_B: list[int] = []
    rounds: list[dict[str, Any]] = []
    current_parameters = (
        None
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float).copy()
    )
    prev_span_G: tuple[int, ...] | None = None
    prev_span_B: tuple[int, ...] | None = None
    prev_cost: float | None = None
    prev_x: np.ndarray | None = None
    prev_las_cost_sum: float | None = None
    prev_las_key: tuple[int, ...] | None = None
    consecutive_all = 0
    seen_cycles: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    best: dict[str, Any] | None = None
    stop_reason = "max_macroiterations"

    for macro in range(max_macro):
        # Step 3: OO on current LAS (skipped on first pass if empty).
        if las and optimize_pool is not None:
            parity_matrix = parity_rows_from_masks(las, norb)
            current_parameters, optimization_pre = optimize_pool(
                parity_matrix,
                current_parameters,
                macro,
            )
            current_parameters = np.asarray(current_parameters, dtype=float).copy()
        else:
            optimization_pre = None

        if score_row_at is None:
            round_scorer = score_row
        else:
            round_scorer = lambda row, _p=current_parameters: score_row_at(row, _p)

        cost_matrix = cost_matrix_in_frame(frame, round_scorer)
        scan = ranked_quotient_scan(
            frame,
            cost_matrix,
            M=int(n_sym),
            exact_masks=exact,
        )
        new_las = [int(m) for m in scan["las_masks"]]
        new_aux = [int(m) for m in scan["auxiliary_masks"]]
        new_B = [int(m) for m in scan["ranked_basis_masks"]]
        new_costs = [float(c) for c in scan["las_costs"]]
        las_cost_sum = float(sum(new_costs))
        las_key = tuple(sorted(new_las))

        span_G = span_key([*exact, *new_las], norb)
        span_B = span_key(new_B, norb)

        # --- five document stop conditions ---
        cond_span_G = prev_span_G is not None and span_G == prev_span_G
        cond_span_B = prev_span_B is not None and span_B == prev_span_B

        if optimize_pool is None:
            cond_oo_obj = True
            cond_oo_step = True
            oo_obj_delta = None
            oo_step_inf = None
        elif optimization_pre is None or prev_cost is None or prev_x is None:
            cond_oo_obj = False
            cond_oo_step = False
            oo_obj_delta = None
            oo_step_inf = None
        else:
            cost_now = float(optimization_pre["cost_after"])
            oo_obj_delta = abs(cost_now - float(prev_cost))
            cond_oo_obj = oo_obj_delta < float(oo_stop_tol)
            x_now = np.asarray(optimization_pre["parameters_after"], dtype=float)
            oo_step_inf = float(np.max(np.abs(x_now - prev_x)))
            cond_oo_step = oo_step_inf < float(oo_step_tol)

        # Ranking: meaningful cheaper LAS set resets; tiny change = stable.
        if prev_las_key is None:
            cond_ranking = False
            ranking_delta = None
        elif las_key == prev_las_key or span_G == prev_span_G:
            ranking_delta = (
                None
                if prev_las_cost_sum is None
                else abs(las_cost_sum - float(prev_las_cost_sum))
            )
            cond_ranking = True
        else:
            ranking_delta = float(prev_las_cost_sum) - las_cost_sum
            # Meaningful improvement (cheaper by more than tol) → not stable.
            cond_ranking = ranking_delta < float(rank_replace_tol)

        stop_checklist = {
            "span_G": bool(cond_span_G),
            "span_B": bool(cond_span_B),
            "oo_obj": bool(cond_oo_obj),
            "oo_step": bool(cond_oo_step),
            "ranking": bool(cond_ranking),
            "oo_obj_delta": oo_obj_delta,
            "oo_step_inf": oo_step_inf,
            "ranking_delta": ranking_delta,
            "las_cost_sum": las_cost_sum,
        }
        all_five = all(
            stop_checklist[k]
            for k in ("span_G", "span_B", "oo_obj", "oo_step", "ranking")
        )
        if all_five:
            consecutive_all += 1
        else:
            consecutive_all = 0

        # Full frame for Clifford: exact rows then ranked quotient basis B.
        frame_masks = [*exact, *new_B]
        if gf2_rank_masks(frame_masks, n_bits=norb) != norb:
            frame_masks = complete_gf2_basis(frame_masks, norb)

        frame, clifford_metadata = clifford_frame_from_masks(frame_masks, norb)

        round_record: dict[str, Any] = {
            "macroiteration": macro,
            "M": int(n_sym),
            "r": int(scan["r"]),
            "exact_masks": [int(m) for m in exact],
            "las_masks": new_las,
            "auxiliary_masks": new_aux,
            "ranked_basis_masks": new_B,
            "las_orbitals": [list(orbitals_from_mask(m)) for m in new_las],
            "auxiliary_orbitals": [list(orbitals_from_mask(m)) for m in new_aux],
            "selected_costs": new_costs,
            "additive_cost": las_cost_sum,
            "selection_trace": scan["selection_trace"],
            "span_key": list(span_G),
            "span_B_key": list(span_B),
            "stable_span_count": consecutive_all,
            "stop_checklist": stop_checklist,
            "frame_basis_rows": [int(x) for x in frame.basis_rows],
            "clifford": clifford_metadata,
        }
        if optimization_pre is not None:
            round_record["optimization"] = optimization_pre

        las = new_las
        las_costs = new_costs
        aux = new_aux
        ranked_B = new_B
        rounds.append(round_record)

        # Track best-so-far by sum of LAS costs (heuristic).
        if best is None or las_cost_sum < float(best["score"]):
            best = {
                "score": las_cost_sum,
                "las": list(las),
                "costs": list(las_costs),
                "aux": list(aux),
                "ranked_B": list(ranked_B),
                "parameters": (
                    None
                    if current_parameters is None
                    else np.asarray(current_parameters, dtype=float).copy()
                ),
            }

        cycle_key = (span_G, span_B)
        # Persistence of the same span is not a cycle; only a return after leaving.
        if cycle_key in seen_cycles and (
            not rounds[:-1]
            or (
                tuple(rounds[-2].get("span_key", [])),
                tuple(rounds[-2].get("span_B_key", [])),
            )
            != cycle_key
        ):
            stop_reason = "cycle"
            if best is not None:
                las = list(best["las"])
                las_costs = list(best["costs"])
                aux = list(best["aux"])
                ranked_B = list(best["ranked_B"])
                if best["parameters"] is not None:
                    current_parameters = np.asarray(best["parameters"], dtype=float)
            break
        seen_cycles.add(cycle_key)

        if consecutive_all >= int(stable_span_iters) and macro > 0:
            stop_reason = "stable_five"
            break

        # Advance previous-state trackers for the next macro.
        prev_span_G = span_G
        prev_span_B = span_B
        prev_las_cost_sum = las_cost_sum
        prev_las_key = las_key
        if optimization_pre is not None:
            prev_cost = float(optimization_pre["cost_after"])
            prev_x = np.asarray(optimization_pre["parameters_after"], dtype=float)

    # Final OO on retained LAS after last discrete update.
    if las and optimize_pool is not None:
        parity_matrix = parity_rows_from_masks(las, norb)
        current_parameters, optimization_final = optimize_pool(
            parity_matrix,
            current_parameters,
            len(rounds),
        )
        current_parameters = np.asarray(current_parameters, dtype=float).copy()
        if rounds:
            rounds[-1]["optimization_final"] = optimization_final

    meff = effective_parity_rank(
        parity_rows_from_masks(las, norb),
        exact_masks=exact,
    )["M_eff"]

    history: dict[str, Any] = {
        "selection_rule": SELECTION_RULE_ITERATIVE,
        "M": int(n_sym),
        "m_total": int(n_sym),
        "max_macroiterations": max_macro,
        "stable_span_iters": int(stable_span_iters),
        "oo_stop_tol": float(oo_stop_tol),
        "oo_step_tol": float(oo_step_tol),
        "rank_replace_tol": float(rank_replace_tol),
        "m_round_deprecated": None if m_round is None else int(m_round),
        "exact_masks": [int(m) for m in exact],
        "exact_rank": r,
        "exact_sector": (
            None if exact_sector is None else [int(b) for b in exact_sector]
        ),
        "las_masks": [int(m) for m in las],
        "auxiliary_masks": [int(m) for m in aux],
        "ranked_basis_masks": [int(m) for m in ranked_B],
        "accumulated_masks": [int(m) for m in las],
        "accumulated_orbitals": [list(orbitals_from_mask(m)) for m in las],
        "rounds": rounds,
        "gf2_rank": gf2_rank_masks(las, n_bits=norb),
        "M_eff": meff,
        "best_score": None if best is None else best["score"],
        "stop_reason": stop_reason,
    }
    return IterativeSelectionResult(
        parity_matrix=parity_rows_from_masks(las, norb),
        accumulated_masks=tuple(int(m) for m in las),
        selected_costs=tuple(float(c) for c in las_costs),
        history=history,
        optimized_parameters=current_parameters,
    )


def as_greedy_result(result: IterativeSelectionResult) -> GreedySelectionResult:
    """Adapt iterative output to the one-shot greedy result shape."""
    return GreedySelectionResult(
        parity_matrix=result.parity_matrix,
        selected_indices=tuple(range(result.parity_matrix.shape[0])),
        selected_costs=result.selected_costs,
        selection_rule=SELECTION_RULE_ITERATIVE,
    )
