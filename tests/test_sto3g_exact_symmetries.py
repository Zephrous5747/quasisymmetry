"""Tests for STO-3G H2O/N2 exact spatial + document qubit families."""

from __future__ import annotations

from src.sto3g_exact_symmetries import (
    STO3G_NORB,
    document_exact_rank,
    exact_masks_info,
    sto3g_document_exact_qubit_masks,
    sto3g_exact_masks,
    sto3g_exact_parity_matrix,
    write_sto3g_exact_parity,
)


class TestSto3gExact:
    def test_h2o_masks(self):
        masks = sto3g_exact_masks("h2o")
        # Q_B1={4} → 16; Q_B2={2,6} → 4|64=68
        assert masks == (16, 68)
        info = exact_masks_info("h2o")
        assert info["exact_rank_spatial"] == 2
        assert info["document_exact_rank"] == 4
        assert document_exact_rank("h2o") == 4
        mat = sto3g_exact_parity_matrix("h2o")
        assert mat.shape == (2, 7)
        qubit = sto3g_document_exact_qubit_masks("h2o")
        assert len(qubit) == 4

    def test_n2_masks(self):
        masks = sto3g_exact_masks("n2")
        # pix=16|128=144, piy=32|256=288, u=2|8|16|32|512=570
        assert masks == (144, 288, 570)
        assert exact_masks_info("n2")["exact_rank_spatial"] == 3
        assert document_exact_rank("n2") == 5
        assert sto3g_exact_parity_matrix("n2").shape == (3, 10)
        assert len(sto3g_document_exact_qubit_masks("n2")) == 5

    def test_write_roundtrip(self, tmp_path):
        path, mat = write_sto3g_exact_parity("n2", tmp_path / "n2.txt")
        assert path.is_file()
        assert mat.shape == (3, STO3G_NORB["n2"])
