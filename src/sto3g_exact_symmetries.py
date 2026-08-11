"""STO-3G H2O / N2 exact symmetries (PDF / sector-dim script).

Document (qubit) exact set, ambient ``N = 2 n_orb``:
  * H2O: ``r = 4`` = 2 spin-resolved number (``N_alpha``, ``N_beta``) + 2 C2v
  * N2:  ``r = 5`` = 2 spin-resolved number + 3 D2h

Spatial supports (LAS / GF(2) spatial register of length ``n_orb``):
  * H2O: ``Q_B1={4}``, ``Q_B2={2,6}``
  * N2:  ``Q_pix={4,7}``, ``Q_piy={5,8}``, ``Q_u={1,3,4,5,9}``

On-disk exact matrices store the **spatial PG rows only**. Spin-resolved
number is enforced by the fixed-``(N_alpha, N_beta)`` FCI space and, when
``include_spin_number_exact`` is set, as leading Clifford Z generators in the
interleaved JW register so metrics ``n_exact`` matches the document ``r``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.exact_parity import exact_rank, parity_matrix_from_masks
from src.gf2_utils import gf2_int_rref

# (label, spatial orbital indices) — 0-based, matching the LaTeX / counting script.
STO3G_EXACT_SPATIAL: dict[str, tuple[tuple[str, frozenset[int]], ...]] = {
    "h2o": (
        ("Q_B1", frozenset({4})),
        ("Q_B2", frozenset({2, 6})),
    ),
    "n2": (
        ("Q_pix", frozenset({4, 7})),
        ("Q_piy", frozenset({5, 8})),
        ("Q_u", frozenset({1, 3, 4, 5, 9})),
    ),
}

STO3G_NORB: dict[str, int] = {"h2o": 7, "n2": 10}

# The index sets above are ONLY valid at the geometry the PDF was written for.
# MO energies cross along both scans (H2O 3a1/1b1 swaps at R ~ 1.47 A; N2 3sigma_g
# sits at index 4 already at R = 1.2 A), so at other bond lengths those indices
# name different irreps and the resulting Z products do not commute with H.
# Derive supports from the irrep labels instead -- see
# ``exact_spatial_sets_from_orbsym``.
PG_GENERATOR_IRREPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "c2v": (
        ("Q_B1", ("B1",)),
        ("Q_B2", ("B2",)),
    ),
    "d2h": (
        ("Q_pix", ("B3u", "B2g")),
        ("Q_piy", ("B2u", "B3g")),
        ("Q_u", ("B1u", "B2u", "B3u", "Au")),
    ),
}

MOLECULE_POINT_GROUP: dict[str, str] = {"h2o": "c2v", "n2": "d2h"}


def normalize_point_group(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in PG_GENERATOR_IRREPS:
        raise ValueError(
            f"no document generator recipe for point group {name!r}; "
            f"known: {sorted(PG_GENERATOR_IRREPS)}"
        )
    return key


def exact_spatial_sets_from_orbsym(
    orbsym: Sequence[str],
    point_group: str,
) -> tuple[tuple[str, frozenset[int]], ...]:
    """Document generator supports from actual MO irrep labels.

    ``orbsym`` must be per-orbital irrep *names* (e.g. ``['A1','A1','B2',...]``),
    in the same order as the orbitals of the Hamiltonian being used.
    """
    pg = normalize_point_group(point_group)
    labels = [str(s).strip() for s in orbsym]
    out: list[tuple[str, frozenset[int]]] = []
    for gname, irreps in PG_GENERATOR_IRREPS[pg]:
        support = frozenset(i for i, s in enumerate(labels) if s in irreps)
        if not support:
            continue
        out.append((gname, support))
    if not out:
        raise ValueError(
            f"no {pg} generator has any support for orbsym={labels}"
        )
    return tuple(out)


def sto3g_exact_masks_from_orbsym(
    orbsym: Sequence[str],
    point_group: str,
) -> tuple[int, ...]:
    """GF(2) masks of the document generators, derived per geometry."""
    sets = exact_spatial_sets_from_orbsym(orbsym, point_group)
    return spatial_sets_to_masks(sets, len(list(orbsym)))


def sto3g_exact_parity_matrix_from_orbsym(
    orbsym: Sequence[str],
    point_group: str,
) -> np.ndarray:
    """Binary ``(r_spatial, norb)`` exact matrix derived from irrep labels."""
    norb = len(list(orbsym))
    return parity_matrix_from_masks(
        sto3g_exact_masks_from_orbsym(orbsym, point_group), norb
    )


def hardcoded_supports_valid(orbsym: Sequence[str], molecule: str) -> bool:
    """True when the stale index sets happen to agree with the irrep labels."""
    key = normalize_molecule(molecule)
    derived = {
        name: sorted(support)
        for name, support in exact_spatial_sets_from_orbsym(
            orbsym, MOLECULE_POINT_GROUP[key]
        )
    }
    hard = {name: sorted(support) for name, support in STO3G_EXACT_SPATIAL[key]}
    return derived == hard

# Document: two spin-resolved number parities always accompany the spatial PG.
PDF_SPIN_NUMBER_RANK = 2


def normalize_molecule(name: str) -> str:
    key = str(name).strip().lower().replace("_", "")
    if key in ("h2o", "water"):
        return "h2o"
    if key in ("n2", "nitrogen"):
        return "n2"
    raise ValueError(f"unknown sto-3g molecule {name!r}; expected h2o or n2")


def spatial_sets_to_masks(
    sets: Sequence[tuple[str, frozenset[int] | set[int] | Sequence[int]]],
    norb: int,
) -> tuple[int, ...]:
    """Pack spatial orbital supports into GF(2) masks of width ``norb``."""
    masks: list[int] = []
    for _label, support in sets:
        mask = 0
        for p in support:
            ip = int(p)
            if ip < 0 or ip >= norb:
                raise ValueError(f"orbital {ip} out of range for norb={norb}")
            mask |= 1 << ip
        if mask:
            masks.append(mask)
    # Drop GF(2)-dependent duplicates while preserving order.
    rref: list[int] = []
    kept: list[int] = []
    for mask in masks:
        trial = [*rref, int(mask)]
        new_rref, _ = gf2_int_rref(trial, norb)
        if len(new_rref) <= len(rref):
            continue
        rref = new_rref
        kept.append(int(mask))
    return tuple(kept)


def sto3g_exact_masks(molecule: str) -> tuple[int, ...]:
    """Exact spatial PG masks for STO-3G H2O or N2 (LAS / on-disk E)."""
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    return spatial_sets_to_masks(STO3G_EXACT_SPATIAL[key], norb)


def sto3g_exact_parity_matrix(molecule: str) -> np.ndarray:
    """Binary ``(r_spatial, norb)`` exact parity matrix for STO-3G H2O / N2."""
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    return parity_matrix_from_masks(sto3g_exact_masks(key), norb)


def expand_spatial_mask_to_interleaved_qubits(mask: int, norb: int) -> int:
    """Map a spatial Z-bar mask to interleaved JW qubits (both spins)."""
    out = 0
    value = int(mask)
    for p in range(norb):
        if (value >> p) & 1:
            out |= (1 << (2 * p)) | (1 << (2 * p + 1))
    return out


def interleaved_spin_number_masks(norb: int) -> tuple[int, int]:
    """``P_alpha``, ``P_beta`` as masks on interleaved JW qubits (``2*norb`` bits)."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    p_alpha = sum(1 << (2 * p) for p in range(norb))
    p_beta = sum(1 << (2 * p + 1) for p in range(norb))
    return int(p_alpha), int(p_beta)


def sto3g_document_exact_qubit_masks(molecule: str) -> tuple[int, ...]:
    """Full PDF exact set in interleaved JW: ``(P_alpha, P_beta) + spatial PG``.

    Rank is 4 (H2O) / 5 (N2), matching ``exact_sector_dimension_H2O_N2.py``
    (block spin ordering there; interleaved here — same GF(2) rank).
    """
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    spin = list(interleaved_spin_number_masks(norb))
    spatial = [
        expand_spatial_mask_to_interleaved_qubits(m, norb)
        for m in sto3g_exact_masks(key)
    ]
    return tuple(spin + spatial)


def document_exact_rank(molecule: str) -> int:
    """PDF exact rank ``r = r_spatial + 2``."""
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    return exact_rank(sto3g_exact_masks(key), norb) + PDF_SPIN_NUMBER_RANK


def default_sto3g_exact_path(molecule: str, *, repo: Path | None = None) -> Path:
    """Canonical on-disk path for the STO-3G spatial exact matrix."""
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    root = repo if repo is not None else Path(".")
    return Path(root) / "exact" / f"{key}_norb{norb}_sto3g_exact.txt"


def write_sto3g_exact_parity(
    molecule: str,
    path: str | Path,
) -> tuple[Path, np.ndarray]:
    """Write the STO-3G spatial exact matrix and return ``(path, matrix)``."""
    key = normalize_molecule(molecule)
    mat = sto3g_exact_parity_matrix(key)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, mat, fmt="%d")
    return out, mat


def exact_masks_info(molecule: str) -> dict:
    """Diagnostic dict: spatial E, document r, qubit masks."""
    key = normalize_molecule(molecule)
    norb = STO3G_NORB[key]
    masks = sto3g_exact_masks(key)
    spatial_rank = exact_rank(masks, norb)
    return {
        "molecule": key,
        "norb": norb,
        "n_qubits": 2 * norb,
        "labels": [lab for lab, _ in STO3G_EXACT_SPATIAL[key]],
        "supports": [sorted(s) for _, s in STO3G_EXACT_SPATIAL[key]],
        "exact_masks": list(masks),
        "exact_rank_spatial": spatial_rank,
        "exact_rank": spatial_rank,
        "document_exact_rank": spatial_rank + PDF_SPIN_NUMBER_RANK,
        "document_exact_qubit_masks": list(sto3g_document_exact_qubit_masks(key)),
        "spin_number_rank": PDF_SPIN_NUMBER_RANK,
    }
