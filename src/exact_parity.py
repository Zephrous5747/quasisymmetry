"""Exact parity set E for quotienting and Clifford tapering.

Default: all-ones total particle parity in the spatial ``Zbar`` register.
Extra exact rows (point-group, etc.) can be loaded from a binary matrix file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.gf2_utils import gf2_matrix_to_int_rows, gf2_int_rref
from src.parity_rank import total_parity_mask


def default_exact_masks(norb: int) -> tuple[int, ...]:
    """Default exact parity: all-ones total particle parity."""
    if norb < 1:
        raise ValueError(f"norb must be positive, got {norb}")
    return (total_parity_mask(norb),)


def masks_from_parity_matrix(parity_matrix: np.ndarray) -> tuple[int, ...]:
    """Pack ``(r, norb)`` binary rows into GF(2) masks."""
    mat = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    if mat.size == 0:
        return ()
    return tuple(int(m) for m in gf2_matrix_to_int_rows(mat) if int(m))


def parity_matrix_from_masks(masks: Sequence[int], norb: int) -> np.ndarray:
    """Unpack masks into a ``(r, norb)`` binary matrix."""
    rows = []
    for mask in masks:
        row = np.zeros(norb, dtype=int)
        value = int(mask)
        bit = 0
        while value:
            if value & 1:
                if bit >= norb:
                    raise ValueError(f"mask bit {bit} exceeds norb={norb}")
                row[bit] = 1
            value >>= 1
            bit += 1
        rows.append(row)
    if not rows:
        return np.zeros((0, norb), dtype=int)
    return np.asarray(rows, dtype=int)


def load_exact_parity_matrix(path: str | Path) -> tuple[tuple[int, ...], np.ndarray]:
    """Load exact parity rows from a text matrix (whitespace-separated 0/1)."""
    path = Path(path)
    mat = np.atleast_2d(np.loadtxt(path, dtype=int))
    if mat.ndim != 2:
        raise ValueError(f"exact parity file must be 2-D, got shape {mat.shape}")
    if not np.all((mat == 0) | (mat == 1)):
        raise ValueError("exact parity matrix must be binary")
    masks = masks_from_parity_matrix(mat)
    # Drop GF(2)-dependent duplicates while preserving order.
    norb = int(mat.shape[1])
    rref: list[int] = []
    kept: list[int] = []
    for mask in masks:
        from src.gf2_utils import gf2_int_try_add_to_span

        new = gf2_int_try_add_to_span(int(mask), rref, norb)
        if new is None:
            continue
        rref = new
        kept.append(int(mask))
    return tuple(kept), parity_matrix_from_masks(kept, norb)


def parse_exact_sector_label(text: str | None, n_exact: int) -> tuple[int, ...] | None:
    """Parse a single binary sector label of length ``n_exact`` (e.g. ``01``)."""
    if text is None or not str(text).strip():
        return None
    item = str(text).strip()
    if "," in item:
        item = item.split(",")[0].strip()
    if len(item) != n_exact or any(bit not in "01" for bit in item):
        raise ValueError(f"exact sector must be a {n_exact}-bit string, got {text!r}")
    return tuple(int(bit) for bit in item)


def resolve_exact_masks(
    norb: int,
    *,
    exact_parity_path: str | Path | None = None,
    exact_masks: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve exact masks from explicit list, file, or default all-ones."""
    if exact_masks is not None:
        return tuple(int(m) for m in exact_masks if int(m))
    if exact_parity_path is not None:
        masks, mat = load_exact_parity_matrix(exact_parity_path)
        if mat.shape[1] != norb:
            raise ValueError(
                f"exact parity file has norb={mat.shape[1]}, expected {norb}"
            )
        return masks
    return default_exact_masks(norb)


def exact_rank(masks: Sequence[int], norb: int) -> int:
    """GF(2) rank of exact masks."""
    rows = [int(m) for m in masks if int(m)]
    if not rows:
        return 0
    rref, _ = gf2_int_rref(rows, norb)
    return len(rref)


def choose_default_exact_sector(
    physical_sectors: dict[tuple[int, ...], list],
) -> tuple[int, ...]:
    """Pick the densest physical tapered sector when no label is provided."""
    if not physical_sectors:
        raise ValueError("no physical exact sectors available")
    if len(physical_sectors) == 1:
        return next(iter(physical_sectors.keys()))
    return max(physical_sectors.items(), key=lambda kv: len(kv[1]))[0]


def exact_metadata(
    masks: Sequence[int],
    norb: int,
    *,
    exact_sector: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """JSON-serializable exact-parity record."""
    return {
        "exact_masks": [int(m) for m in masks],
        "exact_rank": exact_rank(masks, norb),
        "exact_sector": None if exact_sector is None else list(exact_sector),
        "norb": int(norb),
    }


def spin_parity_sector_bits(nelec: Sequence[int]) -> tuple[int, int]:
    """``(N_alpha mod 2, N_beta mod 2)`` for document spin-number exact sectors."""
    return (int(nelec[0]) % 2, int(nelec[1]) % 2)


def expand_exact_sector_with_spin(
    exact_sector: tuple[int, ...] | list[int] | None,
    nelec: Sequence[int] | None,
    *,
    include_spin_number: bool,
    n_exact_spatial: int,
) -> tuple[int, ...] | None:
    """Prepend spin-parity bits to a spatial-only exact sector label.

    CLI / OO JSON may store the spatial PG sector only (length
    ``n_exact_spatial``). With document spin-number generators leading the
    Clifford list, the physical label is ``(n_a%2, n_b%2) + spatial``.
    """
    if exact_sector is None:
        return None
    sector = tuple(int(b) for b in exact_sector)
    if not include_spin_number or nelec is None:
        return sector
    spin = spin_parity_sector_bits(nelec)
    if len(sector) == n_exact_spatial + 2 and tuple(sector[:2]) == spin:
        return sector
    if len(sector) == n_exact_spatial:
        return spin + sector
    if len(sector) == n_exact_spatial + 2:
        # Spin bits present but may not match this geometry's (Na, Nb).
        return spin + sector[2:]
    return sector


def _iota(mask: int, norb: int) -> int:
    """Embed a spatial parity row into the interleaved qubit register.

    ``iota(x)_{2p} = iota(x)_{2p+1} = x_p``: each spatial bit is duplicated onto
    both spin qubits of that orbital, i.e. Zbar_p = Z_{p,alpha} Z_{p,beta}.
    Defined locally to avoid a circular import with sto3g_exact_symmetries.
    """
    out = 0
    for p in range(int(norb)):
        if (int(mask) >> p) & 1:
            out |= (1 << (2 * p)) | (1 << (2 * p + 1))
    return out


def _spin_number_qubit_masks(norb: int) -> tuple[int, int]:
    """``P_alpha``, ``P_beta`` as interleaved-JW qubit masks."""
    n = int(norb)
    return (sum(1 << (2 * p) for p in range(n)),
            sum(1 << (2 * p + 1) for p in range(n)))


def qubit_exact_rank(
    exact_spatial_masks: Sequence[int],
    norb: int,
    *,
    include_spin: bool,
) -> int:
    """Rank of the exact group in the QUBIT register ``F_2^(2 norb)``.

    This is the ``r`` that sets the exact sector-label length, and it is NOT
    ``len(exact_spatial_masks) + 2``: the only spatial row whose image under
    :func:`_iota` lies in ``span{P_alpha, P_beta}`` is all-ones, since
    ``iota(1) = P_alpha ^ P_beta``. Computed here by direct GF(2) elimination
    rather than by special-casing that row.
    """
    n_qubits = 2 * int(norb)
    rows: list[int] = []
    if include_spin:
        rows.extend(_spin_number_qubit_masks(norb))
    rows.extend(_iota(int(m), norb) for m in exact_spatial_masks if int(m))
    rref, _ = gf2_int_rref([m for m in rows if m], n_qubits)
    return len(rref)


def resolve_exact_las_split(input_data: dict, parity_matrix, norb: int) -> dict[str, Any]:
    """Split OO exact / LAS rows for tapered Heff and K.

    Builds a GF(2)-independent Clifford generator list with exact rows first,
    then LAS that enlarge the span.  Orbital-disjoint Mixed often has
    all-ones ∈ span(LAS), so naively stacking exact ∪ LAS is dependent and
    ``Clifford.from_symmetries`` rejects it — those dependent LAS rows are
    dropped from the tapered generator list (recorded in ``las_masks_dropped``).

    When ``include_spin_number_exact`` is True (PDF / STO-3G workflow), Clifford
    also prepends ``N_alpha`` / ``N_beta`` Z generators so ``n_exact`` matches
    document ``r = r_spatial + 2``. Spatial ``exact_masks`` are unchanged for
    LAS GF(2) selection.
    """
    from src.gf2_utils import gf2_int_try_add_to_span

    parity_matrix = np.atleast_2d(np.asarray(parity_matrix, dtype=int))
    raw_exact = input_data.get("exact_masks")
    if not raw_exact:
        las = list(masks_from_parity_matrix(parity_matrix))
        return {
            "exact_tapered": False,
            "n_exact": 0,
            "n_exact_spatial": 0,
            "n_spin_exact": 0,
            "include_spin_number_exact": False,
            "n_las": len(las),
            "n_tail": 2 * norb - len(las),
            "exact_sector": None,
            "exact_masks": [],
            "las_masks": las,
            "las_masks_kept": las,
            "las_masks_dropped": [],
            "combined_matrix": parity_matrix,
        }

    exact_masks_in = [int(m) for m in raw_exact if int(m)]
    las_raw = input_data.get("las_masks") or input_data.get("accumulated_masks")
    if las_raw is not None:
        las_masks = [int(m) for m in las_raw if int(m)]
    else:
        las_masks = list(masks_from_parity_matrix(parity_matrix))

    # Exact first (keep only independent exact rows), then independent LAS.
    rref: list[int] = []
    exact_kept: list[int] = []
    for mask in exact_masks_in:
        new = gf2_int_try_add_to_span(int(mask), rref, norb)
        if new is None:
            continue
        rref = new
        exact_kept.append(int(mask))

    las_kept: list[int] = []
    las_dropped: list[int] = []
    for mask in las_masks:
        new = gf2_int_try_add_to_span(int(mask), rref, norb)
        if new is None:
            las_dropped.append(int(mask))
            continue
        rref = new
        las_kept.append(int(mask))

    combined_masks = [*exact_kept, *las_kept]
    if combined_masks:
        combined = parity_matrix_from_masks(combined_masks, norb)
    else:
        combined = np.zeros((0, norb), dtype=int)

    include_spin = bool(input_data.get("include_spin_number_exact", False))
    n_exact_spatial = len(exact_kept)
    n_spin = 2 if include_spin else 0
    n_las = len(las_kept)

    # Two DIFFERENT ranks, previously conflated as n_exact = n_exact_spatial +
    # n_spin. They coincide only while all-ones is absent from the exact set.
    #
    #   r_sp  = dim of the exact group inside the SPATIAL register F_2^n.
    #           Governs the GF(2) independence test on LAS candidates and the
    #           budget bound M <= n - r_sp.
    #   r     = rank of the exact group in the QUBIT register F_2^(2n).
    #           Governs the Clifford tapering and the length of the exact
    #           sector label.
    #
    # They differ because iota(all-ones) = P_alpha ^ P_beta: adding all-ones to
    # the spatial set raises r_sp by one but leaves r unchanged. Using
    # n_exact_spatial + n_spin as the sector-label length would then over-length
    # the label and mis-slice filter_labels_fixed_exact.
    r_sp = n_exact_spatial
    r_qubit = qubit_exact_rank(exact_kept, norb, include_spin=include_spin)
    n_exact = r_qubit if include_spin else r_sp
    m_max_spatial = norb - r_sp
    exact_sector = input_data.get("exact_sector")
    if exact_sector is not None:
        exact_sector = tuple(int(b) for b in exact_sector)
        # Accept spatial-only or full (spin+spatial) labels; finalize with nelec later.
        if len(exact_sector) not in (n_exact_spatial, n_exact_spatial + n_spin):
            if len(exact_sector) > n_exact_spatial:
                exact_sector = exact_sector[: n_exact_spatial + n_spin]
            else:
                exact_sector = None
    return {
        "exact_tapered": True,
        "n_exact": n_exact,
        "n_exact_spatial": n_exact_spatial,
        "n_spin_exact": n_spin,
        "include_spin_number_exact": include_spin,
        "n_las": n_las,
        "n_tail": 2 * norb - n_exact - n_las,
        "exact_sector": exact_sector,
        "exact_masks": exact_kept,
        "exact_masks_input": exact_masks_in,
        "las_masks": las_masks,
        "las_masks_kept": las_kept,
        "las_masks_dropped": las_dropped,
        "combined_matrix": combined,
        "combined_rank": n_exact_spatial + n_las,
        "r_sp": r_sp,
        "r_qubit": r_qubit,
        "M_max_spatial": m_max_spatial,
    }


def filter_labels_fixed_exact(labels, n_exact: int, exact_sector):
    """Keep sector labels whose leading ``n_exact`` bits match ``exact_sector``."""
    if n_exact <= 0 or exact_sector is None:
        return list(labels)
    e = tuple(int(b) for b in exact_sector)
    if len(e) != n_exact:
        raise ValueError(
            f"exact_sector length {len(e)} != n_exact={n_exact}"
        )
    kept = [label for label in labels if tuple(label[:n_exact]) == e]
    if not kept:
        raise ValueError(
            f"no physical sectors match exact_sector={e} among {list(labels)}"
        )
    return kept
