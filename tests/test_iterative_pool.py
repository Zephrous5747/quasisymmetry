"""Unit tests for iterative GF(2) pool extension."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.greedy_selection import select_senquart_kruskal_from_cost_matrix
from src.iterative_pool import (
    SELECTION_RULE_ITERATIVE,
    clifford_frame_from_masks,
    complete_gf2_basis,
    gf2_rank_masks,
    mask_from_orbitals,
    orbitals_from_mask,
    select_iterative_pool,
)
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
        # A frame-local quartet can pull back to a genuine higher-order term.
        assert orbitals_from_mask(frame.mask_for_quartet((0, 2))) == (0, 1, 2)


class TestKruskalPriorSingles:
    def test_skips_folded_axes(self):
        n = 4
        cost = np.full((n, n), 10.0, dtype=float)
        cost[0, 0] = 0.001
        cost[1, 1] = 0.002
        cost[2, 2] = 0.01
        cost[3, 3] = 0.02
        singles, quartets, _ = select_senquart_kruskal_from_cost_matrix(
            cost, m=2, prior_singles=(0, 1)
        )
        assert singles == (2, 3)
        assert quartets == ()


class TestIterativeSelection:
    def test_m_round_equals_n_sym_matches_one_shot(self):
        n = 4
        m = 3
        cost = np.full((n, n), 1.0, dtype=float)
        cost[0, 0] = 0.01
        cost[1, 1] = 0.02
        cost[0, 1] = 0.03
        cost[2, 2] = 0.04
        cost[2, 3] = 0.05

        kruskal_singles, kruskal_quartets, _ = select_senquart_kruskal_from_cost_matrix(
            cost, m=m
        )

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
            result = select_iterative_pool(n, m, score_row, m_round=m)

        assert result.history["selection_rule"] == SELECTION_RULE_ITERATIVE
        assert len(result.history["rounds"]) == 1
        assert result.parity_matrix.shape == (m, n)
        assert gf2_rank_masks(result.accumulated_masks) == m

        # Same identity-frame picks as one-shot Kruskal.
        singles = []
        quartets = []
        for mask in result.accumulated_masks:
            orbs = orbitals_from_mask(mask)
            if len(orbs) == 1:
                singles.append(orbs[0])
            elif len(orbs) == 2:
                quartets.append(orbs)
        assert frozenset(singles) == frozenset(kruskal_singles)
        assert frozenset(quartets) == frozenset(kruskal_quartets)

    def test_multi_round_accumulates_and_skips_prior(self):
        n = 4
        n_sym = 2
        m_round = 1

        round1 = np.full((n, n), 10.0, dtype=float)
        round1[0, 1] = 0.01
        round1[0, 0] = 0.5
        round1[1, 1] = 0.5

        round2 = np.full((n, n), 10.0, dtype=float)
        # Folded axis 0 looks cheapest; prior_singles must skip it.
        round2[0, 0] = 0.001
        round2[1, 1] = 0.02
        round2[2, 2] = 0.03

        matrices = [round1, round2]
        frames_seen: list[list[int]] = []

        def fake_cost_matrix(frame, score_row):
            frames_seen.append(list(frame.basis_rows))
            index = min(len(frames_seen) - 1, len(matrices) - 1)
            return matrices[index]

        def score_row(_row: np.ndarray) -> float:
            return 1.0

        with patch(
            "src.iterative_pool.cost_matrix_in_frame",
            side_effect=fake_cost_matrix,
        ):
            result = select_iterative_pool(n, n_sym, score_row, m_round=m_round)

        assert len(result.history["rounds"]) == 2
        assert result.history["gf2_rank"] == n_sym
        assert len(result.accumulated_masks) == n_sym
        assert result.history["rounds"][0]["quartets"] == [[0, 1]]
        assert result.history["rounds"][0]["prior_singles"] == []
        assert frames_seen[0] == [1 << i for i in range(n)]
        assert frames_seen[1][0] == (1 << 0) ^ (1 << 1)
        assert result.history["rounds"][1]["prior_singles"] == [0]
        assert 0 not in result.history["rounds"][1]["singles"]

    def test_rejects_bad_m_round(self):
        with pytest.raises(ValueError, match="m_round"):
            select_iterative_pool(3, 2, lambda row: 1.0, m_round=0)

    def test_interleaves_optimization_and_uses_updated_parameters(self):
        parameters_seen: list[float] = []
        optimized_pool_sizes: list[int] = []

        def score_at(row: np.ndarray, parameters: np.ndarray | None) -> float:
            assert parameters is not None
            parameters_seen.append(float(parameters[0]))
            support = np.flatnonzero(row)
            # Stable, nondegenerate additive costs.
            return float(10 * len(support) + sum(int(i) for i in support))

        def optimize_pool(parity, parameters, round_index):
            assert parameters is not None
            optimized_pool_sizes.append(len(parity))
            updated = np.asarray(parameters, dtype=float) + 1.0
            return updated, {
                "cost_before": float(round_index + 1),
                "cost_after": float(round_index),
            }

        result = select_iterative_pool(
            4,
            2,
            lambda row: 0.0,
            m_round=1,
            score_row_at=score_at,
            optimize_pool=optimize_pool,
            initial_parameters=np.array([0.0]),
        )

        assert optimized_pool_sizes == [1, 2]
        assert 0.0 in parameters_seen
        assert 1.0 in parameters_seen
        np.testing.assert_allclose(result.optimized_parameters, [2.0])
        assert all("optimization" in record for record in result.history["rounds"])
        assert all("clifford" in record for record in result.history["rounds"])


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
