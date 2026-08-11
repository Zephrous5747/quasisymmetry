"""Unit tests for effective parity rank (mod total particle parity)."""

from __future__ import annotations

import numpy as np
import pytest

from src.parity_rank import (
    effective_parity_rank,
    m_eff,
    raw_parity_rank,
    total_parity_row,
)


class TestEffectiveParityRank:
    def test_disjoint_cover_drops_one(self):
        # Five disjoint generators covering all 7 orbitals of H2O-like:
        # three singles + two quartets that partition {0..6}.
        rows = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ]
        mat = np.asarray(rows, dtype=int)
        info = effective_parity_rank(mat)
        assert info["raw_rank"] == 5
        assert info["contains_total_parity"] is True
        assert info["M_eff"] == 4
        assert m_eff(mat) == 4

    def test_missing_orbital_keeps_full_eff(self):
        # Four generators; orbital 6 uncovered ⇒ all-ones not in span.
        rows = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ]
        mat = np.asarray(rows, dtype=int)
        info = effective_parity_rank(mat)
        assert info["raw_rank"] == 4
        assert info["contains_total_parity"] is False
        assert info["M_eff"] == 4

    def test_all_ones_alone(self):
        mat = np.atleast_2d(total_parity_row(5))
        assert raw_parity_rank(mat) == 1
        assert m_eff(mat) == 0

    def test_n2_disjoint_seven(self):
        # 4 singles + 3 quartets covering 10 orbitals.
        rows = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
        info = effective_parity_rank(np.asarray(rows, dtype=int))
        assert info["raw_rank"] == 7
        assert info["M_eff"] == 6
        assert info["contains_total_parity"] is True


class TestFciRotationChecks:
    def test_exact_eigenpair(self):
        from src.fci_rotation_checks import verify_rotated_fci

        e = -1.5
        psi = np.array([0.0, 1.0, 0.0], dtype=np.complex128)

        def h_apply(v):
            out = np.zeros_like(v)
            out[1] = e * v[1]
            return out

        report = verify_rotated_fci(
            h_apply=h_apply, rotated_state=psi, e_fci=e, fresh_state=psi
        )
        assert report["ok"]
        assert report["residual_norm"] < 1e-12
        assert report["overlap_vs_fresh"] == pytest.approx(1.0)
