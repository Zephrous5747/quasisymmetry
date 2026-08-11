"""Tests for exact parity loading and Route A exact taper helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.exact_parity import (
    default_exact_masks,
    filter_labels_fixed_exact,
    load_exact_parity_matrix,
    parse_exact_sector_label,
    resolve_exact_las_split,
    resolve_exact_masks,
)
from src.iterative_pool import ranked_quotient_scan, Gf2ParityFrame
from src.parity_rank import effective_parity_rank, total_parity_mask


class TestExactParity:
    def test_default_all_ones(self):
        assert default_exact_masks(4) == (total_parity_mask(4),)

    def test_load_and_resolve(self, tmp_path: Path):
        path = tmp_path / "exact.txt"
        path.write_text("1 0 0\n0 1 1\n", encoding="utf-8")
        masks, mat = load_exact_parity_matrix(path)
        assert mat.shape == (2, 3)
        assert resolve_exact_masks(3, exact_parity_path=path) == masks

    def test_parse_sector(self):
        assert parse_exact_sector_label("01", 2) == (0, 1)
        with pytest.raises(ValueError):
            parse_exact_sector_label("0", 2)

    def test_custom_exact_in_quotient_scan(self):
        n = 3
        frame = Gf2ParityFrame.identity(n)
        cost = np.full((n, n), 1.0)
        cost[0, 0] = 0.001
        cost[1, 1] = 0.01
        cost[2, 2] = 0.02
        exact = (1 << 0,)
        scan = ranked_quotient_scan(frame, cost, M=1, exact_masks=exact)
        assert (1 << 0) not in scan["ranked_basis_masks"]
        assert len(scan["ranked_basis_masks"]) == n - 1

    def test_effective_rank_custom_exact(self):
        mat = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=int)
        info = effective_parity_rank(mat, exact_masks=[1 << 0])
        assert info["M_eff"] == 1
        assert info["exact_parity_rank"] == 1


class TestExactLasSplit:
    def test_no_exact_masks(self):
        parity = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=int)
        split = resolve_exact_las_split({}, parity, 3)
        assert split["exact_tapered"] is False
        assert split["n_las"] == 2
        assert split["n_exact"] == 0

    def test_exact_union_las(self):
        norb = 4
        exact = [total_parity_mask(norb)]
        las = [1 << 0, 1 << 1]
        parity = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=int)
        split = resolve_exact_las_split(
            {
                "exact_masks": exact,
                "las_masks": las,
                "exact_sector": [0],
            },
            parity,
            norb,
        )
        assert split["exact_tapered"] is True
        assert split["n_exact"] == 1
        assert split["n_exact_spatial"] == 1
        assert split["n_spin_exact"] == 0
        assert split["n_las"] == 2
        assert split["n_tail"] == 2 * norb - 1 - 2
        assert split["combined_matrix"].shape == (3, norb)
        assert split["exact_sector"] == (0,)
        assert split["las_masks_dropped"] == []

    def test_document_spin_number_exact(self):
        norb = 7
        from src.sto3g_exact_symmetries import sto3g_exact_masks

        exact = list(sto3g_exact_masks("h2o"))
        las = [1 << 0]
        parity = np.asarray([[1, 0, 0, 0, 0, 0, 0]], dtype=int)
        split = resolve_exact_las_split(
            {
                "exact_masks": exact,
                "las_masks": las,
                "include_spin_number_exact": True,
            },
            parity,
            norb,
        )
        assert split["n_exact_spatial"] == 2
        assert split["n_spin_exact"] == 2
        assert split["n_exact"] == 4  # document r for H2O
        assert split["n_las"] == 1
        assert split["n_tail"] == 2 * norb - 4 - 1

    def test_drops_las_dependent_on_exact_cover(self):
        """Disjoint Mixed covering all orbitals ⇒ all-ones ∈ span(LAS)."""
        norb = 7
        exact = [total_parity_mask(norb)]
        # 3 singles + 2 quartets partitioning {0..6}
        las = [
            1 << 0,
            1 << 1,
            1 << 2,
            (1 << 3) | (1 << 4),
            (1 << 5) | (1 << 6),
        ]
        parity = np.zeros((5, norb), dtype=int)
        for i, m in enumerate(las):
            for b in range(norb):
                if m & (1 << b):
                    parity[i, b] = 1
        split = resolve_exact_las_split(
            {"exact_masks": exact, "las_masks": las},
            parity,
            norb,
        )
        assert split["n_exact"] == 1
        assert split["n_las"] == 4  # one LAS dropped as exact-dressed
        assert len(split["las_masks_dropped"]) == 1
        assert split["combined_matrix"].shape == (5, norb)
        assert split["combined_rank"] == 5

    def test_filter_labels(self):
        labels = [(0, 0), (0, 1), (1, 0), (1, 1)]
        kept = filter_labels_fixed_exact(labels, 1, (0,))
        assert kept == [(0, 0), (0, 1)]


class TestExactTaperContext:
    def test_prepare_context_normalized(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("openfermion")
        from src.exact_taper import (
            _residual_gs_to_clifford_frame_state,
            jw_state_to_ci_vector,
            prepare_exact_taper_context,
        )

        norb = 2
        nelec = (1, 1)
        masks = default_exact_masks(norb)
        ctx = prepare_exact_taper_context(masks, norb, nelec, exact_sector="0")
        assert ctx["n_symmetries"] == 1
        assert ctx["exact_sector"] == (0,)
        label = ctx["exact_sector"]
        residual = ctx["physical_sectors"][label]
        if not residual:
            pytest.skip("no physical residual for this sector")
        vec = np.zeros(len(residual), dtype=complex)
        vec[0] = 1.0
        cliff_state = _residual_gs_to_clifford_frame_state(
            vec,
            label,
            residual,
            2 * norb - ctx["n_symmetries"],
        )
        jw = ctx["clifford"].inverse_transform_state(cliff_state)
        assert pytest.approx(np.linalg.norm(jw), abs=1e-10) == 1.0
        ci = jw_state_to_ci_vector(jw, norb, nelec)
        assert pytest.approx(np.linalg.norm(ci), abs=1e-10) == 1.0
