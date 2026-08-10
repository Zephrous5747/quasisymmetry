"""Energy-sector diagnostics: decoupled and coupled (PT or reference) K."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.coupled_energy_core import (
    CHEMICAL_PRECISION,
    COUPLED_ENERGY_DEGENERACY_FLOOR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TAU_PT,
    all_sector_eigenpair_candidates,
    one_shot_coupled_energy,
    reference_coupled_energy,
)


def diagonalize_sector_blocks(h_apply, sectors_dict, full_dim: int):
    """
    Diagonalize each symmetry block independently.

    Returns sector_data mapping sector key -> {idxs, evals, evecs_full}.
    """
    sector_data = {}
    for key, idxs in sectors_dict.items():
        h_local = _subspace_matrix(h_apply, idxs, full_dim)
        h_local = 0.5 * (h_local + h_local.conj().T)
        evals, evecs = np.linalg.eigh(h_local)

        evecs_full = []
        for j in range(evecs.shape[1]):
            v = np.zeros(full_dim, dtype=np.complex128)
            v[np.asarray(idxs, dtype=int)] = evecs[:, j]
            evecs_full.append(v)

        sector_data[key] = {
            "idxs": idxs,
            "evals": evals,
            "evecs_full": evecs_full,
        }
    return sector_data


def sector_data_from_gs_pairs(sectors_dict, sector_gs_pairs, full_dim: int):
    """Build sector_data from precomputed (evals, local_evecs) per sector."""
    sector_data = {}
    for key, idxs in sectors_dict.items():
        evals, evecs_local = sector_gs_pairs[key]
        evecs_full = []
        for j in range(evecs_local.shape[1]):
            v = np.zeros(full_dim, dtype=np.complex128)
            v[idxs] = evecs_local[:, j]
            evecs_full.append(v)
        sector_data[key] = {
            "idxs": idxs,
            "evals": evals,
            "evecs_full": evecs_full,
        }
    return sector_data


def sector_dimension_stats(sectors_dict) -> dict:
    """Exact max / aggregate determinant dimensions (PDF ``D_max``).

    For a fixed exact-target filter already applied to ``sectors_dict``,
    ``D_max = max_s dim(s)`` matches the PDF §13 definition of the largest
    joint exact+approximate sector. ``total_dim`` is the sum over sectors
    (legacy ``relevant_sectors_total_dim`` when restricted to K-used sectors).

    Values may be index sequences (``len`` = dim) or integer dimensions.
    """
    dims: list[int] = []
    for value in sectors_dict.values():
        if isinstance(value, (int, np.integer)):
            dims.append(int(value))
        elif isinstance(value, dict) and "dimension" in value:
            dims.append(int(value["dimension"]))
        else:
            dims.append(len(value))
    if not dims:
        return {
            "D_max": 0,
            "D_min": 0,
            "n_sectors": 0,
            "total_dim": 0,
        }
    return {
        "D_max": int(max(dims)),
        "D_min": int(min(dims)),
        "n_sectors": int(len(dims)),
        "total_dim": int(sum(dims)),
    }


def decoupled_energy_test(sectors_dict, sector_gs_pairs):
    """E_dec_min = min_s lambda_min(H(s))."""
    best_e = None
    best_key = None
    best_dim = 0
    for key, idxs in sectors_dict.items():
        e0 = float(np.min(sector_gs_pairs[key][0]))
        if best_e is None or e0 < best_e:
            best_e = e0
            best_key = key
            best_dim = len(idxs)
    return best_e, best_key, best_dim


def coupled_energy_perturbation(
    h_apply: Callable[[np.ndarray], np.ndarray],
    sector_data,
    *,
    e_exact: float | None = None,
    tol: float = CHEMICAL_PRECISION,
    max_total_vectors: int | None = None,
    tau_pt: float = DEFAULT_TAU_PT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    degeneracy_floor: float = COUPLED_ENERGY_DEGENERACY_FLOOR,
    backward_prune: bool = True,
):
    """One-shot PT ordering + nested variational coupled dimension."""
    candidates = all_sector_eigenpair_candidates(sector_data)
    return one_shot_coupled_energy(
        candidates,
        h_apply,
        e_exact=e_exact,
        tol=tol,
        tau_pt=tau_pt,
        block_size=block_size,
        degeneracy_floor=degeneracy_floor,
        max_total_vectors=max_total_vectors,
        backward_prune=backward_prune,
    ).as_tuple()


def reference_coupled_energy_k(
    h_apply: Callable[[np.ndarray], np.ndarray],
    full_space_vectors_cat: np.ndarray,
    reference_vector: np.ndarray,
    e_exact: float,
    *,
    chemical_precision: float = CHEMICAL_PRECISION,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """
    Reference-overlap ordering + nested variational ``K``.

    ``full_space_vectors_cat`` columns are sector eigenstates. Returns
    ``(K, e_coupled, converged, order_indices, reference_weights)`` where
    ``order_indices`` indexes those columns in decreasing ``|<psi|Psi_ref>|^2``
    order and ``reference_weights`` are the per-column overlap weights.
    """
    n = full_space_vectors_cat.shape[1]
    candidates = [
        (0.0, j, full_space_vectors_cat[:, j], 0) for j in range(n)
    ]
    result = reference_coupled_energy(
        candidates,
        h_apply,
        reference_vector,
        e_exact=e_exact,
        tol=chemical_precision,
        block_size=block_size,
    )
    order = np.asarray(result.order_indices, dtype=int)
    weights = np.asarray(result.reference_weights, dtype=float)
    if result.K is None:
        return None, result.e_coupled, False, order, weights
    return result.K, result.e_coupled, result.converged, order, weights



def state_labels_for_columns(sector_gs_pairs):
    """(sector_label, block_index) for each column of the concatenated basis."""
    labels = []
    for sector_label, sector_gs in sector_gs_pairs.items():
        for i in range(sector_gs[1].shape[1]):
            labels.append((sector_label, i))
    return labels


def _subspace_matrix(h_apply, support, full_dim: int):
    dim = len(support)
    h_sub = np.zeros((dim, dim), dtype=np.complex128)
    for i, big_index in enumerate(support):
        x = np.zeros(full_dim, dtype=np.complex128)
        x[big_index] = 1.0
        y = h_apply(x)
        h_sub[:, i] = y[support]
    return h_sub
