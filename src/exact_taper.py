"""Exact-parity Clifford tapering (Route A reference state).

Uses QuasiSymmetries ``Clifford`` + ``taper_hamiltonian`` via
:mod:`src.clifford_sectors` to restrict ``H(U)`` to a fixed exact sector and
return an approximate ground state for NC / variance scoring.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.clifford_sectors import (
    bits_to_index,
    ci_vector_to_jw_state,
    molecular_hamiltonian_to_jw,
    occupation_bits,
    physical_clifford_basis,
    prepare_clifford_context,
    solve_tapered_sector,
    transform_hamiltonian_in_context,
)
from src.exact_parity import (
    choose_default_exact_sector,
    parity_matrix_from_masks,
    parse_exact_sector_label,
)
from src.orbital_rotation import params_to_U


def exact_symmetries_from_masks(
    exact_masks: Sequence[int],
    norb: int,
    *,
    include_spin_number: bool = False,
):
    """OpenFermion Z-products for exact spatial rows (+ optional spin number)."""
    from src.clifford_sectors import clifford_symmetries_from_spatial

    matrix = parity_matrix_from_masks(exact_masks, norb)
    if matrix.shape[0] == 0 and not include_spin_number:
        raise ValueError("exact_masks must be non-empty for tapering")
    built = clifford_symmetries_from_spatial(
        matrix if matrix.shape[0] else np.zeros((0, norb), dtype=int),
        norb,
        include_spin_number=include_spin_number,
    )
    if not built["symmetries"]:
        raise ValueError("no independent exact generators after spin/spatial filter")
    return built["symmetries"]


def prepare_exact_taper_context(
    exact_masks: Sequence[int],
    norb: int,
    nelec: tuple[int, int],
    *,
    exact_sector: str | tuple[int, ...] | None = None,
    include_spin_number: bool = False,
) -> dict[str, Any]:
    """Clifford context for tapering exact symmetries only."""
    from src.exact_parity import expand_exact_sector_with_spin

    symmetries = exact_symmetries_from_masks(
        exact_masks, norb, include_spin_number=include_spin_number
    )
    context = prepare_clifford_context(symmetries, norb, nelec)
    basis = physical_clifford_basis(
        norb,
        nelec,
        context["clifford"],
        context["n_symmetries"],
    )
    context["physical_positions"] = basis["physical_positions"]
    context["full_indices"] = basis["full_indices"]
    context["exact_masks"] = [int(m) for m in exact_masks]
    context["include_spin_number_exact"] = bool(include_spin_number)
    n_exact_spatial = len([m for m in exact_masks if int(m)])
    n_exact = int(context["n_symmetries"])

    label: tuple[int, ...] | None
    if isinstance(exact_sector, str):
        try:
            label = parse_exact_sector_label(exact_sector, n_exact_spatial)
        except ValueError:
            label = parse_exact_sector_label(exact_sector, n_exact)
    elif exact_sector is None:
        label = None
    else:
        label = tuple(int(b) for b in exact_sector)

    if include_spin_number and label is not None:
        label = expand_exact_sector_with_spin(
            label,
            nelec,
            include_spin_number=True,
            n_exact_spatial=n_exact_spatial,
        )

    if label is not None and len(label) != n_exact:
        if len(label) > n_exact:
            label = label[:n_exact]
        if label is not None and len(label) != n_exact:
            label = None

    if label is None:
        label = choose_default_exact_sector(context["physical_sectors"])
    if label not in context["physical_sectors"]:
        raise ValueError(
            f"exact sector {label} not present among physical sectors "
            f"{list(context['physical_sectors'])}"
        )
    context["exact_sector"] = label
    return context


def _residual_gs_to_clifford_frame_state(
    vector: np.ndarray,
    label: tuple[int, ...],
    residual_indices: Sequence[int],
    n_residual_qubits: int,
) -> np.ndarray:
    """Embed a residual GS into the full Clifford-frame computational basis."""
    n_sym = len(label)
    n_qubits = n_sym + int(n_residual_qubits)
    state = np.zeros(1 << n_qubits, dtype=complex)
    offset = bits_to_index(list(label)) << int(n_residual_qubits)
    vec = np.asarray(vector, dtype=complex).reshape(-1)
    if vec.size != len(residual_indices):
        raise ValueError("residual GS length does not match physical support")
    for amp, res_idx in zip(vec, residual_indices):
        state[offset + int(res_idx)] = amp
    return state


def jw_state_to_ci_vector(
    jw_state: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    threshold: float = 1e-14,
) -> np.ndarray:
    """Project a JW computational-basis state onto the fixed-spin CI vector."""
    import pyscf.fci.cistring

    alpha_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), int(nelec[0])), dtype=np.int64
    )
    beta_strings = np.asarray(
        pyscf.fci.cistring.make_strings(range(norb), int(nelec[1])), dtype=np.int64
    )
    ci = np.zeros((len(alpha_strings), len(beta_strings)), dtype=complex)
    state = np.asarray(jw_state, dtype=complex).reshape(-1)
    n_qubits = 2 * norb
    if state.size != 1 << n_qubits:
        raise ValueError(
            f"JW state length {state.size} != 2^{n_qubits}"
        )

    # Reverse of ci_vector_to_jw_state phase convention.
    for alpha_address, alpha_string in enumerate(alpha_strings):
        alpha = [
            orbital for orbital in range(norb) if (int(alpha_string) >> orbital) & 1
        ]
        for beta_address, beta_string in enumerate(beta_strings):
            beta = [
                orbital for orbital in range(norb) if (int(beta_string) >> orbital) & 1
            ]
            inversions = sum(
                beta_orbital < alpha_orbital
                for alpha_orbital in alpha
                for beta_orbital in beta
            )
            phase = -1.0 if inversions % 2 else 1.0
            idx = bits_to_index(occupation_bits(alpha, beta, norb))
            amp = state[idx]
            if abs(amp) <= threshold:
                continue
            ci[alpha_address, beta_address] = phase * amp
    flat = ci.reshape(-1)
    nrm = float(np.linalg.norm(flat))
    if nrm == 0.0:
        raise RuntimeError("exact-sector GS has zero overlap with fixed-(Na,Nb) space")
    return flat / nrm


def approx_ground_state_exact_sector(
    moldata,
    U: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Solve ApproxGroundState of ``H(U)`` in the fixed exact tapered sector.

    Returns a CI vector in the rotated orbital frame (do not rotate again for
    NC) plus diagnostics.
    """
    rotated_h = moldata.hamiltonian.rotated(np.asarray(U, dtype=float))
    jw_h = molecular_hamiltonian_to_jw(rotated_h, moldata.nelec)
    frame = transform_hamiltonian_in_context(jw_h, context)
    label = tuple(context["exact_sector"])
    residual_indices = context["physical_sectors"][label]
    solved = solve_tapered_sector(frame, label, residual_indices, n_roots=1)
    clifford_state = _residual_gs_to_clifford_frame_state(
        solved["vectors"][:, 0],
        label,
        residual_indices,
        frame["n_residual_qubits"],
    )
    jw_state = context["clifford"].inverse_transform_state(clifford_state)
    ci = jw_state_to_ci_vector(jw_state, moldata.norb, moldata.nelec)
    return {
        "ci_vector": ci,
        "energy": float(np.real(solved["energies"][0])),
        "exact_sector": list(label),
        "dimension": int(solved["dimension"]),
        "solver": solved["solver"],
    }


def approx_ground_state_from_params(
    moldata,
    x: np.ndarray,
    context: dict[str, Any],
    *,
    pairs=None,
) -> dict[str, Any]:
    """Route A helper: map orbital parameters ``x`` → exact-sector GS."""
    U = params_to_U(np.asarray(x, dtype=float), moldata.norb, pairs)
    return approx_ground_state_exact_sector(moldata, U, context)


def _psi_and_h_at(
    moldata,
    x: np.ndarray,
    context: dict[str, Any],
    *,
    pairs=None,
    cache: dict | None = None,
):
    """Return ``(U, psi, h_linop)`` for Route A at parameters ``x`` (cached)."""
    import ffsim

    x_arr = np.asarray(x, dtype=float)
    key = tuple(x_arr.tolist())
    if cache is not None and key in cache:
        return cache[key]
    U = params_to_U(x_arr, moldata.norb, pairs)
    psi = approx_ground_state_exact_sector(moldata, U, context)["ci_vector"]
    h = ffsim.linear_operator(
        moldata.hamiltonian.rotated(U),
        norb=moldata.norb,
        nelec=moldata.nelec,
    )
    packed = (U, psi, h)
    if cache is not None:
        cache[key] = packed
    return packed


def nc_on_state(h, symmetries: list, psi: np.ndarray) -> float:
    """``sum_s ||[H, S] |psi>||^2`` with ``psi`` already in the H frame."""
    total_nc = 0.0
    h_psi = h @ psi
    for s in symmetries:
        term1 = h @ (s @ psi)
        term2 = s @ h_psi
        commutator_on_state = term1 - term2
        total_nc += float(np.vdot(commutator_on_state, commutator_on_state).real)
    return total_nc


def variance_on_state(symmetries: list, psi: np.ndarray) -> float:
    """``sum_s (1 - <S>^2)`` on a fixed state."""
    total_var = 0.0
    for s in symmetries:
        total_var += 1.0 - float(((psi.T.conj() @ s @ psi) ** 2).real)
    return total_var


def make_exact_taper_nc_cost(
    moldata,
    symmetries: list,
    context: dict[str, Any],
    *,
    pairs=None,
):
    """NC cost using Route A exact-tapered GS at each trial ``U`` (no extra rotation)."""
    cache: dict = {}

    def f(x: np.ndarray) -> float:
        _U, psi, h = _psi_and_h_at(moldata, x, context, pairs=pairs, cache=cache)
        return nc_on_state(h, symmetries, psi)

    return f


def make_exact_taper_variance_cost(
    moldata,
    symmetries: list,
    context: dict[str, Any],
    *,
    pairs=None,
):
    """Variance cost using Route A exact-tapered GS at each trial ``U``."""
    cache: dict = {}

    def f(x: np.ndarray) -> float:
        _U, psi, _h = _psi_and_h_at(moldata, x, context, pairs=pairs, cache=cache)
        return variance_on_state(symmetries, psi)

    return f


def make_exact_taper_row_scorer(
    moldata,
    context: dict[str, Any],
    *,
    cost_function: str = "NC",
    pairs=None,
    row_to_operators,
):
    """Per-row NC/variance scorer for ranked scan (caches ``Psi_A(U)``)."""
    if cost_function not in ("NC", "variance"):
        raise ValueError(f"unsupported cost_function {cost_function!r}")
    cache: dict = {}

    def score_at(row: np.ndarray, parameters: np.ndarray | None) -> float:
        if parameters is None:
            raise RuntimeError("exact-taper scoring requires orbital parameters")
        _U, psi, h = _psi_and_h_at(
            moldata, parameters, context, pairs=pairs, cache=cache
        )
        ops = row_to_operators(row)
        if cost_function == "NC":
            return nc_on_state(h, ops, psi)
        return variance_on_state(ops, psi)

    return score_at
