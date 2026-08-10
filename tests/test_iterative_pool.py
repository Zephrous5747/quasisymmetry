"""Unit tests for iterative NC-ranked fixed-M LAS search."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.iterative_pool import (
    SELECTION_RULE_ITERATIVE,
    clifford_frame_from_masks,
    complete_gf2_basis,
    default_exact_masks,
    gf2_rank_masks,
    mask_from_orbitals,
    orbitals_from_mask,
    ranked_quotient_scan,
    select_iterative_pool,
    Gf2ParityFrame,
)
from src.parity_rank import total_parity_mask
from src.workflow_cli import validate_greedy_cli_args


class TestMaskHelpers:
    def test_orbitals_roundtrip(self):
        mask = mask_from_orbitals([0, 2, 4])
        assert orbitals_from_mask(mask) == (0, 2, 4)

    def test_complete_gf2_basis_selected_first(self):
        selected = [(1 << 0) ^ (1 << 1), 1 << 2]
        basis = complete_gf2_basis(selected, n=4)
        assert len(basis) == 4
        assert basis[0] == selected[0]
        assert basis[1] == selected[1]
        assert gf2_rank_masks(basis) == 4

    def test_external_clifford_maps_selected_product_to_leading_axis(self):
        frame, metadata = clifford_frame_from_masks(
            [mask_from_orbitals([0, 1])],
            norb=4,
        )
        assert frame.basis_rows[0] == mask_from_orbitals([0, 1])
        assert metadata["backend"] == "external.QuasiSymmetries.Clifford"
        assert metadata["symmetry_qubits"] == [0]
        assert orbitals_from_mask(frame.mask_for_quartet((0, 2))) == (0, 1, 2)

    def test_default_exact_is_all_ones(self):
        assert default_exact_masks(5) == (total_parity_mask(5),)


class TestRankedQuotientScan:
    def test_rejects_total_parity_and_fills_las(self):
        n = 4
        M = 2
        frame = Gf2ParityFrame.identity(n)
        cost = np.full((n, n), 10.0, dtype=float)
        # Cheap singles 0,1,2 then pair (0,1) etc.
        cost[0, 0] = 0.01
        cost[1, 1] = 0.02
        cost[2, 2] = 0.03
        cost[3, 3] = 0.04
        cost[0, 1] = 0.05
        # Make all-ones pair expensive so it appears later: for n=4 all-ones is
        # not a single/pair of identity frame except weight-2 only.
        exact = default_exact_masks(n)
        scan = ranked_quotient_scan(frame, cost, M=M, exact_masks=exact)
        assert len(scan["las_masks"]) == M
        assert len(scan["ranked_basis_masks"]) == n - 1  # N - r
        assert len(scan["auxiliary_masks"]) == n - 1 - M
        assert total_parity_mask(n) not in scan["las_masks"]
        assert total_parity_mask(n) not in scan["ranked_basis_masks"]
        # First accepted should be cheapest independent singles.
        assert scan["las_masks"][0] == (1 << 0)
        assert scan["las_masks"][1] == (1 << 1)

    def test_exact_candidate_rejected(self):
        n = 3
        frame = Gf2ParityFrame.identity(n)
        cost = np.full((n, n), 1.0, dtype=float)
        cost[0, 0] = 0.001  # exact single — must be skipped
        cost[1, 1] = 0.01
        cost[2, 2] = 0.02
        cost[1, 2] = 0.03
        exact = (1 << 0,)
        scan = ranked_quotient_scan(frame, cost, M=2, exact_masks=exact)
        assert len(scan["ranked_basis_masks"]) == n - 1
        assert (1 << 0) not in scan["ranked_basis_masks"]
        rejected = [t for t in scan["selection_trace"] if t["event"] == "reject"]
        assert any(t["reason"] == "exact_parity" and t["mask"] == 1 for t in rejected)


class TestIterativeSelection:
    def test_fixed_m_one_macro(self):
        n = 4
        M = 2
        cost = np.full((n, n), 1.0, dtype=float)
        cost[0, 0] = 0.01
        cost[1, 1] = 0.02
        cost[2, 2] = 0.03
        cost[3, 3] = 0.04

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                i = int(support[0])
                return float(cost[i, i])
            p, q = int(support[0]), int(support[1])
            return float(cost[p, q])

        with patch(
            "src.iterative_pool.cost_matrix_in_frame",
            return_value=cost,
        ):
            result = select_iterative_pool(
                n,
                M,
                score_row,
                max_macroiterations=1,
                stable_span_iters=99,
            )

        assert result.history["selection_rule"] == SELECTION_RULE_ITERATIVE
        assert result.parity_matrix.shape == (M, n)
        assert len(result.accumulated_masks) == M
        assert len(result.history["ranked_basis_masks"]) == n - 1
        assert len(result.history["auxiliary_masks"]) == n - 1 - M
        assert result.history["exact_masks"] == [total_parity_mask(n)]
        assert result.history["M_eff"] == M  # all-ones not in span of two singles

    def test_rejects_m_exceeding_quotient_dim(self):
        with pytest.raises(ValueError, match="N-r"):
            select_iterative_pool(3, 3, lambda row: 1.0)  # r=1 ⇒ max M=2

    def test_rejects_bad_deprecated_m_round(self):
        with pytest.raises(ValueError, match="m_round"):
            select_iterative_pool(3, 2, lambda row: 1.0, m_round=0)

    def test_five_condition_stop_requires_all(self):
        """With no OO, oo_obj/oo_step are vacuously true; span+ranking drive stop."""
        n = 4
        M = 2
        cost = np.full((n, n), 1.0, dtype=float)
        cost[0, 0] = 0.01
        cost[1, 1] = 0.02
        cost[2, 2] = 0.03
        cost[3, 3] = 0.04

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return float(cost[int(support[0]), int(support[0])])
            return float(cost[int(support[0]), int(support[1])])

        with patch(
            "src.iterative_pool.cost_matrix_in_frame",
            return_value=cost,
        ):
            result = select_iterative_pool(
                n,
                M,
                score_row,
                max_macroiterations=10,
                stable_span_iters=2,
            )

        assert result.history["stop_reason"] in ("stable_five", "max_macroiterations", "cycle")
        assert "stop_checklist" in result.history["rounds"][0]
        checklist = result.history["rounds"][0]["stop_checklist"]
        for key in ("span_G", "span_B", "oo_obj", "oo_step", "ranking"):
            assert key in checklist
        # No OO → first macro cannot be fully stable (ranking/span need prev).
        assert checklist["span_G"] is False
        assert checklist["ranking"] is False
        # Later macros should eventually hit stable_five for constant costs.
        if len(result.history["rounds"]) >= 3:
            assert result.history["stop_reason"] == "stable_five"

    def test_meaningful_replacement_blocks_stop(self):
        n = 4
        M = 1
        call = {"i": 0}

        def score_row(row: np.ndarray) -> float:
            support = tuple(int(x) for x in np.flatnonzero(row))
            # Alternate preferred single so LAS flips with large cost delta.
            prefer = 0 if call["i"] % 2 == 0 else 1
            call["i"] += 1
            if support == (prefer,):
                return 0.01
            if len(support) == 1:
                return 1.0
            return 2.0

        # Force changing cost matrices each macro via score_row_at side effect.
        matrices = []
        base = np.full((n, n), 2.0)
        for flip in (0, 1, 0, 1):
            m = base.copy()
            m[flip, flip] = 0.01
            m[1 - flip, 1 - flip] = 1.0
            matrices.append(m)

        idx = {"k": 0}

        def fake_cost_matrix(frame, scorer):
            k = min(idx["k"], len(matrices) - 1)
            idx["k"] += 1
            return matrices[k]

        with patch(
            "src.iterative_pool.cost_matrix_in_frame",
            side_effect=fake_cost_matrix,
        ):
            result = select_iterative_pool(
                n,
                M,
                lambda row: 1.0,
                max_macroiterations=4,
                stable_span_iters=2,
                rank_replace_tol=1e-4,
            )

        # Alternating LAS should prevent stable_five (or stop on cycle).
        assert result.history["stop_reason"] in ("cycle", "max_macroiterations")
        assert result.history["stop_reason"] != "stable_five"

    def test_interleaves_optimization(self):
        parameters_seen: list[float] = []
        optimized_pool_sizes: list[int] = []

        def score_at(row: np.ndarray, parameters: np.ndarray | None) -> float:
            assert parameters is not None
            parameters_seen.append(float(parameters[0]))
            support = np.flatnonzero(row)
            return float(10 * len(support) + sum(int(i) for i in support))

        def optimize_pool(parity, parameters, round_index):
            assert parameters is not None
            optimized_pool_sizes.append(len(parity))
            updated = np.asarray(parameters, dtype=float) + 1.0
            return updated, {
                "cost_before": float(round_index + 1),
                "cost_after": float(round_index),
                "parameters_before": np.asarray(parameters, dtype=float).tolist(),
                "parameters_after": updated.tolist(),
            }

        result = select_iterative_pool(
            4,
            2,
            lambda row: 0.0,
            score_row_at=score_at,
            optimize_pool=optimize_pool,
            initial_parameters=np.array([0.0]),
            max_macroiterations=2,
            stable_span_iters=2,
        )

        assert all(sz == 2 for sz in optimized_pool_sizes)
        assert optimized_pool_sizes  # at least final OO
        assert 0.0 in parameters_seen
        assert result.parity_matrix.shape[0] == 2
        assert "clifford" in result.history["rounds"][0]
        assert "stop_checklist" in result.history["rounds"][0]


class TestCliValidation:
    def test_iterative_requires_n_sym(self):
        with pytest.raises(ValueError, match="--n_sym"):
            validate_greedy_cli_args(
                select="iterative",
                n_sym=None,
                cost_function="NC",
            )

    def test_iterative_rejects_bad_m_round(self):
        with pytest.raises(ValueError, match="--m_round"):
            validate_greedy_cli_args(
                select="iterative",
                n_sym=2,
                cost_function="variance",
                m_round=0,
            )

    def test_iterative_ok(self):
        validate_greedy_cli_args(
            select="iterative",
            n_sym=3,
            cost_function="NC",
            m_round=2,
        )
