"""GF(2) parity ranks with quotient by an exact-parity span E.

Effective independent approximate labels:
    M_eff = rank(E cup G) - rank(E)
Default E is span{all-ones} (total particle parity in the spatial register).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.gf2_utils import (
    gf2_int_in_span,
    gf2_int_rref,
    gf2_matrix_to_int_rows,
    gf2_rank,
)


def total_parity_row(norb: int) -> np.ndarray:
    """All-ones spatial parity vector of length ``norb``."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    return np.ones(norb, dtype=int)


def total_parity_mask(norb: int) -> int:
    """Packed all-ones mask (bit i set for orbital i)."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    return (1 << norb) - 1


def raw_parity_rank(parity_matrix: np.ndarray) -> int:
    """GF(2) rank of generator rows in the full spatial-parity space."""
    mat = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if mat.size == 0:
        return 0
    return int(gf2_rank(mat))


def effective_parity_rank(
    parity_matrix: np.ndarray,
    *,
    exact_masks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Raw rank and ``M_eff`` modulo an exact-parity span.

    Parameters
    ----------
    parity_matrix :
        Shape ``(m, norb)`` binary generator matrix (spatial ``Zbar`` rows).
    exact_masks :
        Packed exact parity rows.  Default: all-ones total particle parity.
    """
    mat = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if mat.size == 0:
        return {
            "norb": 0,
            "raw_rank": 0,
            "exact_parity_rank": 0,
            "rank_mod_total_parity": 0,
            "M_eff": 0,
            "contains_total_parity": False,
            "contains_exact_span": False,
        }
    norb = int(mat.shape[1])
    if exact_masks is None:
        exact = [total_parity_mask(norb)]
    else:
        exact = [int(m) for m in exact_masks if int(m)]
    rref_e, _ = gf2_int_rref(exact, norb)
    r_e = len(rref_e)

    packed = gf2_matrix_to_int_rows(mat)
    rref_g, _ = gf2_int_rref(packed, norb)
    raw = len(rref_g)

    rref_union, _ = gf2_int_rref([*exact, *packed], norb)
    rank_union = len(rref_union)
    m_eff = int(rank_union - r_e)

    ones = total_parity_mask(norb)
    contains_ones = bool(gf2_int_in_span(ones, rref_g))
    # Exact span contained in G iff rank(E cup G) == rank(G).
    contains_exact = rank_union == raw

    return {
        "norb": norb,
        "raw_rank": int(raw),
        "exact_parity_rank": int(r_e),
        "rank_mod_total_parity": m_eff,
        "M_eff": m_eff,
        "contains_total_parity": contains_ones,
        "contains_exact_span": contains_exact,
        "exact_masks": [int(m) for m in exact],
    }


def m_eff(
    parity_matrix: np.ndarray,
    *,
    exact_masks: Sequence[int] | None = None,
) -> int:
    """Convenience wrapper for :func:`effective_parity_rank` ``M_eff``."""
    return int(effective_parity_rank(parity_matrix, exact_masks=exact_masks)["M_eff"])
