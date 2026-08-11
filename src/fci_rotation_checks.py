"""Verification helpers for rotated FCI states used in OO / metrics.

Orbital optimization rotates a stored FCI vector rather than recomputing FCI
at every trial ``U``.  These checks document that the rotated pair
``(H(U), Psi(U))`` remains consistent with the reference eigenvalue.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def verify_rotated_fci(
    *,
    h_apply: Callable[[np.ndarray], np.ndarray],
    rotated_state: np.ndarray,
    e_fci: float,
    fresh_state: np.ndarray | None = None,
) -> dict[str, Any]:
    """Norm, energy, residual, and optional overlap vs a fresh FCI vector.

    Parameters
    ----------
    h_apply :
        Linear map ``v |-> H(U) v`` in the same orbital frame as ``rotated_state``.
    rotated_state :
        Stored FCI vector after ``apply_orbital_rotation(..., U)``.
    e_fci :
        Reference FCI energy (orbital-invariant).
    fresh_state :
        Optional independently recomputed FCI eigenvector in the same frame
        (e.g. FCI on rotated integrals).  When provided, reports overlap.
    """
    psi = np.asarray(rotated_state, dtype=np.complex128).ravel()
    nrm = float(np.linalg.norm(psi))
    if nrm == 0.0:
        return {
            "norm": 0.0,
            "energy": None,
            "energy_error": None,
            "residual_norm": None,
            "overlap_vs_fresh": None,
            "ok": False,
            "message": "zero state",
        }
    psi_u = psi / nrm
    hpsi = np.asarray(h_apply(psi_u), dtype=np.complex128).ravel()
    energy = float(np.real(np.vdot(psi_u, hpsi)))
    residual = hpsi - float(e_fci) * psi_u
    residual_norm = float(np.linalg.norm(residual))
    overlap = None
    if fresh_state is not None:
        phi = np.asarray(fresh_state, dtype=np.complex128).ravel()
        pn = float(np.linalg.norm(phi))
        if pn > 0.0:
            overlap = float(abs(np.vdot(psi_u, phi / pn)))
    energy_error = abs(energy - float(e_fci))
    # Loose default: residual should be tiny for exact FCI + unitary rotation.
    ok = (
        abs(nrm - 1.0) < 1e-8
        and energy_error < 1e-6
        and residual_norm < 1e-5
        and (overlap is None or overlap > 1.0 - 1e-6)
    )
    return {
        "norm": nrm,
        "energy": energy,
        "energy_error": energy_error,
        "residual_norm": residual_norm,
        "overlap_vs_fresh": overlap,
        "ok": bool(ok),
        "message": "ok" if ok else "check failed",
    }
