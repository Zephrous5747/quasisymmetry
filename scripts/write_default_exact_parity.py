#!/usr/bin/env python
"""Write exact-parity matrix E for iterative / Mixed / metrics.

Defaults:
  * ``--molecule h2o|n2`` → STO-3G report exact **spatial** PG rows
    (H2O: Q_B1, Q_B2; N2: Q_pix, Q_piy, Q_u).
    Document qubit rank is ``r_spatial + 2`` (``N_alpha``, ``N_beta``),
    wired in Clifford via ``include_spin_number_exact`` — not stored here.
  * otherwise ``--norb N`` → all-ones total particle parity.

Optional ``--extra PATH`` merges additional GF(2)-independent binary rows.

Examples::

    python scripts/write_default_exact_parity.py --molecule h2o \\
        -o exact/h2o_norb7_sto3g_exact.txt
    python scripts/write_default_exact_parity.py --molecule n2 \\
        -o exact/n2_norb10_sto3g_exact.txt
    python scripts/write_default_exact_parity.py --norb 7 -o exact/h2o_E.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.exact_parity import (
    default_exact_masks,
    load_exact_parity_matrix,
    parity_matrix_from_masks,
)
from src.gf2_utils import gf2_int_rref, gf2_int_try_add_to_span
from src.sto3g_exact_symmetries import (
    STO3G_NORB,
    normalize_molecule,
    sto3g_exact_masks,
)


def _orbsym_from_chk(chk: str | Path) -> tuple[list[str], str, int]:
    """(irrep labels, point group, norb) from a symmetry-adapted PySCF chk."""
    from pyscf import symm
    from pyscf.scf import chkfile

    chk = Path(chk)
    mol = chkfile.load_mol(str(chk))
    if not getattr(mol, "symmetry", None):
        raise SystemExit(
            f"{chk} carries no point-group symmetry; rebuild it with "
            "make_pyscf_hamiltonian.py --point_group"
        )
    mo_coeff = np.asarray(chkfile.load(str(chk), "scf")["mo_coeff"])
    labels = [
        str(s)
        for s in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    ]
    return labels, str(mol.groupname), int(mo_coeff.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=("h2o", "n2"),
        default=None,
        help="use STO-3G report exact spatial rows for this molecule",
    )
    parser.add_argument(
        "--norb",
        type=int,
        default=None,
        help="spatial orbitals (required unless --molecule is set)",
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--extra",
        default=None,
        help="optional extra binary exact rows to merge (GF(2)-independent)",
    )
    parser.add_argument(
        "--chk",
        default=None,
        help=(
            "PySCF .chk for this geometry; derive the generator supports from "
            "the actual MO irrep labels instead of the hardcoded indices. "
            "STRONGLY preferred -- the hardcoded indices are only valid at the "
            "PDF reference geometry."
        ),
    )
    parser.add_argument(
        "--include-total-parity",
        dest="include_total_parity",
        action="store_true",
        default=True,
        help="include all-ones (total particle parity) in the stored exact "
             "matrix so the GF(2) independence test quotients by it (default)",
    )
    parser.add_argument(
        "--no-include-total-parity",
        dest="include_total_parity",
        action="store_false",
        help="legacy behaviour: point-group rows only (overstates M_max by one)",
    )
    args = parser.parse_args()

    if args.chk is not None:
        from src.sto3g_exact_symmetries import (
            exact_spatial_sets_from_orbsym,
            hardcoded_supports_valid,
            sto3g_exact_masks_from_orbsym,
        )

        orbsym, point_group, norb = _orbsym_from_chk(args.chk)
        if args.norb is not None and int(args.norb) != norb:
            raise SystemExit(f"--norb={args.norb} but chk has norb={norb}")
        masks = list(sto3g_exact_masks_from_orbsym(orbsym, point_group))
        sets = exact_spatial_sets_from_orbsym(orbsym, point_group)
        print(f"[exact] chk={args.chk} pg={point_group}")
        print(f"[exact] orbsym={list(orbsym)}")
        for name, support in sets:
            print(f"[exact]   {name:8s} = {sorted(support)}")
        if args.molecule is not None:
            key = normalize_molecule(args.molecule)
            if not hardcoded_supports_valid(orbsym, key):
                print(
                    "[exact] NOTE: hardcoded STO3G_EXACT_SPATIAL indices are "
                    "WRONG at this geometry; using the derived supports."
                )
    elif args.molecule is not None:
        key = normalize_molecule(args.molecule)
        norb = STO3G_NORB[key]
        if args.norb is not None and int(args.norb) != norb:
            raise SystemExit(
                f"--norb={args.norb} incompatible with --molecule {key} "
                f"(expected {norb})"
            )
        print(
            "[exact][warn] using hardcoded orbital indices; these are only "
            "valid at the PDF reference geometry. Pass --chk to derive them "
            "from the actual irrep labels."
        )
        masks = list(sto3g_exact_masks(key))
    else:
        if args.norb is None:
            raise SystemExit("provide --molecule or --norb")
        norb = int(args.norb)
        masks = list(default_exact_masks(norb))

    # Total particle parity 1 = prod_p Zbar_p = (-1)^N = P_alpha ^ P_beta is an
    # exact symmetry and is already implied whenever P_alpha, P_beta are in the
    # exact set. Omitting it from the stored matrix does not remove it from the
    # physics, but it does remove it from the GF(2) independence test, which
    # overstates the admissible LAS budget by one row (M <= n - r_sp).
    # It raises r_sp but NOT the qubit rank r, since iota(1) = P_alpha ^ P_beta.
    if args.include_total_parity:
        all_ones = (1 << norb) - 1
        probe, _ = gf2_int_rref([*masks, all_ones], norb)
        base, _ = gf2_int_rref(list(masks), norb)
        if len(probe) > len(base):
            masks = [all_ones, *masks]
            print(f"[exact] total particle parity added: mask={all_ones} "
                  f"(r_sp {len(base)} -> {len(probe)}, M_max {norb - len(base)} "
                  f"-> {norb - len(probe)})")
        else:
            print("[exact] total particle parity already in the span; not added")

    rref = list(masks)
    if args.extra:
        extra, _mat = load_exact_parity_matrix(args.extra)
        if _mat.shape[1] != norb:
            raise SystemExit(
                f"--extra has norb={_mat.shape[1]}, expected {norb}"
            )
        for mask in extra:
            new = gf2_int_try_add_to_span(int(mask), rref, norb)
            if new is None:
                continue
            rref = new
            masks.append(int(mask))
    mat = parity_matrix_from_masks(masks, norb)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: up to 12 array tasks share one geometry and all write this file.
    # np.savetxt truncates first, so a plain write leaves a window in which a
    # concurrent reader sees an empty or half-written matrix.
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
    np.savetxt(tmp, mat, fmt="%d")
    os.replace(tmp, out)
    print(f"[ok] wrote {out} shape={mat.shape} masks={masks}")


if __name__ == "__main__":
    main()
