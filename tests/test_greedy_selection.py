"""Unit tests for greedy / Kruskal Z-symmetry selection."""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb

import numpy as np
import pytest

from src.gf2_utils import gf2_int_in_span, gf2_int_rref, gf2_matrix_to_int_rows
from src.greedy_selection import (
    candidate_pool,
    default_parity_output_path,
    greedy_select_independent,
    select_senquart_quota,
    senquart_candidates,
    seniority_candidates,
)
from src.workflow_cli import validate_greedy_cli_args


def _is_independent(rows: np.ndarray) -> bool:
    packed = gf2_matrix_to_int_rows(rows)
    rref, _ = gf2_int_rref(packed, rows.shape[1])
    return len(rref) == len(rows)


def _brute_force_min_weight(vectors: np.ndarray, costs: np.ndarray, n_sym: int) -> float:
    best = np.inf
    n = len(vectors)
    for idxs in combinations(range(n), n_sym):
        subset = vectors[list(idxs)]
        if not _is_independent(subset):
            continue
        best = min(best, float(costs[list(idxs)].sum()))
    if not np.isfinite(best):
        raise AssertionError("no independent set found in brute force")
    return best


class TestCandidatePools:
    def test_seniority_shape(self):
        norb = 5
        pool = seniority_candidates(norb)
        assert pool.shape == (norb, norb)
        np.testing.assert_array_equal(pool, np.eye(norb, dtype=int))

    def test_senquart_shape(self):
        norb = 4
        pool = senquart_candidates(norb)
        assert pool.shape == (norb + comb(norb, 2), norb)
        # First norb rows are seniorities; remaining are weight-2.
        np.testing.assert_array_equal(pool[:norb], np.eye(norb, dtype=int))
        assert np.all(pool[norb:].sum(axis=1) == 2)

    def test_candidate_pool_dispatch(self):
        assert candidate_pool(3, "seniority").shape == (3, 3)
        assert candidate_pool(3, "senquart").shape == (6, 3)
        with pytest.raises(ValueError, match="senquart"):
            candidate_pool(3, "beam")


class TestGreedySelect:
    def test_triangle_kruskal(self):
        # Vertices 0,1,2 with edges (0,1), (0,2), (1,2) as binary rows e_i+e_j.
        # Costs 1, 2, 10 → pick first two; third forms a cycle.
        vectors = np.array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=int,
        )
        costs = [1.0, 2.0, 10.0]
        result = greedy_select_independent(vectors, costs, n_sym=2)
        assert result.selected_indices == (0, 1)
        assert result.selected_costs == (1.0, 2.0)
        assert _is_independent(result.parity_matrix)

        # Dependent third edge lies in the span of the first two.
        packed = gf2_matrix_to_int_rows(result.parity_matrix)
        rref, _ = gf2_int_rref(packed, 3)
        assert gf2_int_in_span(gf2_matrix_to_int_rows(vectors[2:3])[0], rref)

    def test_exact_optimality_senquart(self):
        norb = 4
        n_sym = 3
        vectors = senquart_candidates(norb)
        rng = np.random.default_rng(0)
        costs = rng.random(len(vectors))
        result = greedy_select_independent(vectors, costs, n_sym)
        assert result.parity_matrix.shape == (n_sym, norb)
        assert _is_independent(result.parity_matrix)
        greedy_sum = sum(result.selected_costs)
        brute_sum = _brute_force_min_weight(vectors, costs, n_sym)
        assert greedy_sum == pytest.approx(brute_sum)

    def test_exact_optimality_seniority_is_cheapest_m(self):
        # Seniorities are already a basis; greedy must pick the M cheapest.
        norb = 5
        vectors = seniority_candidates(norb)
        costs = np.array([4.0, 1.0, 3.0, 2.0, 5.0])
        result = greedy_select_independent(vectors, costs, n_sym=3)
        assert set(result.selected_indices) == {1, 3, 2}
        assert sum(result.selected_costs) == pytest.approx(6.0)

    def test_n_sym_too_large(self):
        vectors = seniority_candidates(3)
        with pytest.raises(ValueError, match="exceeds ambient"):
            greedy_select_independent(vectors, [1.0, 2.0, 3.0], n_sym=4)

    def test_n_sym_nonpositive(self):
        vectors = seniority_candidates(2)
        with pytest.raises(ValueError, match="positive"):
            greedy_select_independent(vectors, [1.0, 2.0], n_sym=0)

    def test_cost_length_mismatch(self):
        vectors = seniority_candidates(2)
        with pytest.raises(ValueError, match="costs length"):
            greedy_select_independent(vectors, [1.0], n_sym=1)

    def test_insufficient_independent(self):
        # Three copies of the same vector: rank 1 only.
        vectors = np.array([[1, 0], [1, 0], [1, 0]], dtype=int)
        with pytest.raises(ValueError, match="could only find"):
            greedy_select_independent(vectors, [1.0, 2.0, 3.0], n_sym=2)


class TestSenquartQuota:
    def test_fills_quota_counts(self):
        norb = 7

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return float(support[0] + 1)
            p, q = int(support[0]), int(support[1])
            return 10.0 + float(p) + 0.1 * float(q)

        result = select_senquart_quota(norb, score_row, n_singles=3, n_quartets=2)
        assert result.selection_rule == "senquart_quota_disjoint"
        assert result.n_singles == 3
        assert result.n_quartets == 2
        assert len(result.singles) == 3
        assert len(result.quartets) == 2
        assert result.parity_matrix.shape == (5, norb)
        assert _is_independent(result.parity_matrix)
        used: set[int] = set(result.singles)
        for p, q in result.quartets:
            assert p not in used and q not in used
            used.add(p)
            used.add(q)
        meta = result.metadata(
            candidates="senquart", cost_function="variance", parity_output="p.txt"
        )
        assert meta["selection_rule"] == "senquart_quota_disjoint"
        assert meta["n_singles"] == 3
        assert meta["n_quartets"] == 2

    def test_forbids_orbital_overlap(self):
        norb = 5

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                # Prefer single 0, then 1.
                return float(support[0])
            p, q = int(support[0]), int(support[1])
            # Cheapest quartet overlaps orbital 0; next is disjoint.
            if (p, q) == (0, 2):
                return 10.0
            if (p, q) == (3, 4):
                return 11.0
            return 20.0 + float(p) + float(q)

        result = select_senquart_quota(norb, score_row, n_singles=1, n_quartets=1)
        assert result.singles == (0,)
        assert result.quartets == ((3, 4),)
        assert (0, 2) not in result.quartets

    def test_rejects_exact_span_candidates(self):
        """LAS must enlarge span(E); exact-dressed generators are skipped."""
        norb = 5
        # Exact: orbital 0 seniority.
        exact = (1 << 0,)

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                # Prefer exact-dressed single 0 first, then 1, 2.
                return float(support[0])
            p, q = int(support[0]), int(support[1])
            return 10.0 + float(p) + 0.1 * float(q)

        result = select_senquart_quota(
            norb,
            score_row,
            n_singles=2,
            n_quartets=1,
            disjoint_orbitals=True,
            exact_masks=exact,
        )
        assert 0 not in result.singles
        packed = gf2_matrix_to_int_rows(result.parity_matrix)
        rref_e, _ = gf2_int_rref(list(exact), norb)
        rref_u, _ = gf2_int_rref([*exact, *packed], norb)
        assert len(rref_u) == len(rref_e) + len(packed)

    def test_h2o_disjoint_independent_of_sto3g_exact(self):
        from src.sto3g_exact_symmetries import sto3g_exact_masks

        norb = 7
        exact = sto3g_exact_masks("h2o")

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return float(support[0] + 1)
            p, q = int(support[0]), int(support[1])
            return 10.0 + float(p) + 0.1 * float(q)

        result = select_senquart_quota(
            norb,
            score_row,
            n_singles=3,
            n_quartets=2,
            disjoint_orbitals=True,
            exact_masks=exact,
        )
        packed = gf2_matrix_to_int_rows(result.parity_matrix)
        rref_e, _ = gf2_int_rref(list(exact), norb)
        rref_u, _ = gf2_int_rref([*exact, *packed], norb)
        assert len(rref_u) == len(rref_e) + result.parity_matrix.shape[0]
        assert result.parity_matrix.shape == (5, norb)

    def test_h2o_singles_before_cheap_quartets_with_exact(self):
        """PG-adapted NC can rank quartets cheaper than singles; still fill 3+2."""
        from src.sto3g_exact_symmetries import sto3g_exact_masks

        norb = 7
        exact = sto3g_exact_masks("h2o")

        def score_quartets_first(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return 100.0 + float(support[0])
            return float(support[0]) + 0.1 * float(support[1])

        result = select_senquart_quota(
            norb,
            score_quartets_first,
            n_singles=3,
            n_quartets=2,
            disjoint_orbitals=True,
            exact_masks=exact,
        )
        assert len(result.singles) == 3
        assert len(result.quartets) == 2
        # Accepts are singles-then-quartets even when quartets score lower.
        kinds = [ev["kind"] for ev in result.selection_trace if ev["event"] == "accept"]
        assert kinds[:3] == ["single", "single", "single"]
        assert kinds[3:] == ["quartet", "quartet"]

    def test_n2_disjoint_exact_survives_adversarial_costs(self):
        """Greedy singles can strand quartets; repair must still find 4+3."""
        from src.sto3g_exact_symmetries import sto3g_exact_masks

        norb = 10
        exact = sto3g_exact_masks("n2")
        fails = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)

            def score_row(row: np.ndarray, rng=rng) -> float:
                support = np.flatnonzero(row)
                if support.size == 1:
                    return float(rng.random())
                return 10.0 + float(rng.random())

            try:
                result = select_senquart_quota(
                    norb,
                    score_row,
                    n_singles=4,
                    n_quartets=3,
                    disjoint_orbitals=True,
                    exact_masks=exact,
                )
            except ValueError:
                fails += 1
                continue
            assert len(result.singles) == 4
            assert len(result.quartets) == 3
        assert fails == 0

    def test_allows_overlap_when_disabled(self):
        norb = 5

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return float(support[0] + 1)
            p, q = int(support[0]), int(support[1])
            return 10.0 + float(p) + 0.1 * float(q)

        result = select_senquart_quota(
            norb, score_row, n_singles=3, n_quartets=2, disjoint_orbitals=False
        )
        assert result.selection_rule == "senquart_quota"
        assert result.singles == (0, 1, 2)
        # With overlap allowed, cheapest quartets reuse low indices.
        assert any(0 in edge for edge in result.quartets)

    def test_skips_dependent_quartet(self):
        norb = 3

        def score_row(row: np.ndarray) -> float:
            support = np.flatnonzero(row)
            if support.size == 1:
                return float(support[0])
            p, q = int(support[0]), int(support[1])
            if (p, q) == (0, 1):
                return 10.0
            return 20.0 + float(p) + float(q)

        result = select_senquart_quota(
            norb, score_row, n_singles=2, n_quartets=1, disjoint_orbitals=False
        )
        assert result.singles == (0, 1)
        assert result.quartets == ((0, 2),) or result.quartets == ((1, 2),)
        assert _is_independent(result.parity_matrix)

    def test_disjoint_underfill_raises(self):
        norb = 5
        # 3 singles + 2 quartets need 7 orbitals when disjoint.
        with pytest.raises(ValueError, match="disjoint_orbitals"):
            select_senquart_quota(
                norb, lambda row: 1.0, n_singles=3, n_quartets=2
            )

    def test_underfill_raises(self):
        norb = 2

        def score_row(row: np.ndarray) -> float:
            return 1.0

        with pytest.raises(ValueError, match="independent quartets|disjoint"):
            select_senquart_quota(norb, score_row, n_singles=0, n_quartets=2)


class TestCliValidation:
    def test_greedy_requires_n_sym_or_quota(self):
        with pytest.raises(ValueError, match="--n_sym|--n_singles"):
            validate_greedy_cli_args(
                select="greedy",
                n_sym=None,
                cost_function="NC",
            )

    def test_greedy_quota_without_n_sym_ok(self):
        validate_greedy_cli_args(
            select="greedy",
            n_sym=None,
            cost_function="NC",
            n_singles=3,
            n_quartets=2,
        )

    def test_greedy_quota_from_argv_when_kwargs_omitted(self, monkeypatch):
        """Stale optimize_*.py may forget to forward quota kwargs."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "optimize_symmetries.py",
                "mol.chk",
                "--select",
                "greedy",
                "--n_singles",
                "3",
                "--n_quartets",
                "2",
            ],
        )
        validate_greedy_cli_args(
            select="greedy",
            n_sym=None,
            cost_function="NC",
        )

    def test_greedy_quota_rejects_seniority_candidates(self):
        with pytest.raises(ValueError, match="senquart"):
            validate_greedy_cli_args(
                select="greedy",
                n_sym=None,
                cost_function="NC",
                n_singles=1,
                n_quartets=1,
                candidates="seniority",
            )

    def test_iterative_still_requires_n_sym(self):
        with pytest.raises(ValueError, match="--n_sym"):
            validate_greedy_cli_args(
                select="iterative",
                n_sym=None,
                cost_function="NC",
                n_singles=3,
                n_quartets=2,
            )

    def test_greedy_rejects_sector_costs(self):
        with pytest.raises(ValueError, match="NC or variance"):
            validate_greedy_cli_args(
                select="greedy",
                n_sym=2,
                cost_function="decoupled",
            )

    def test_greedy_rejects_parity_positional(self):
        with pytest.raises(ValueError, match="omit the parity"):
            validate_greedy_cli_args(
                select="greedy",
                n_sym=2,
                cost_function="variance",
                parity="parity.txt",
            )

    def test_greedy_rejects_seniority_flag(self):
        with pytest.raises(ValueError, match="--candidates seniority"):
            validate_greedy_cli_args(
                select="greedy",
                n_sym=2,
                cost_function="NC",
                seniority=True,
            )

    def test_none_requires_source(self):
        with pytest.raises(ValueError, match="parity matrix"):
            validate_greedy_cli_args(
                select="none",
                n_sym=None,
                cost_function="NC",
            )

    def test_none_accepts_parity(self):
        validate_greedy_cli_args(
            select="none",
            n_sym=None,
            cost_function="NC",
            parity="parity.txt",
        )

    def test_greedy_ok(self):
        validate_greedy_cli_args(
            select="greedy",
            n_sym=3,
            cost_function="variance",
        )

    def test_iterative_ok(self):
        validate_greedy_cli_args(
            select="iterative",
            n_sym=3,
            cost_function="NC",
            m_round=2,
        )


class TestDefaultParityOutput:
    def test_with_outname(self):
        assert default_parity_output_path("OO_mol.json") == "OO_mol_parity.txt"

    def test_without_outname_greedy(self):
        assert default_parity_output_path(None) == "parity_greedy.txt"

    def test_without_outname_iterative(self):
        assert (
            default_parity_output_path(None, select="iterative")
            == "parity_iterative.txt"
        )
