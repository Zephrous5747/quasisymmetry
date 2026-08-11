#!/usr/bin/env python
"""Write the STO-3G exact parity matrix from the ACTUAL orbital irrep labels.

`src/sto3g_exact_symmetries.py` hardcodes orbital *indices*
(H2O: Q_B1={4}, Q_B2={2,6};  N2: Q_pix={4,7}, Q_piy={5,8}, Q_u={1,3,4,5,9}).
Those indices are only correct at the geometry the PDF was written for. MO
energies cross along the scan, so at other bond lengths the same indices name
different irreps and the resulting Z products are not symmetries of H at all.

This derives the supports per geometry from the point-group labels of the
Hartree-Fock MOs in the run's own .chk, so the generators are exact everywhere.

Usage
-----
    python scripts/audit/gen_exact_from_orbsym.py \
        --chk hamiltonians/N2_bond1.8000sto-3g_D2h.chk \
        -o exact/n2_R1.8000_exact.txt

    # also prints the sector label that holds the RHF reference, for --exact_sector
    python scripts/audit/gen_exact_from_orbsym.py --chk ... --print_sector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Which irreps make up each document generator.
GENERATORS = {
    "c2v": {"Q_B1": ("B1",), "Q_B2": ("B2",)},
    "d2h": {
        "Q_pix": ("B3u", "B2g"),
        "Q_piy": ("B2u", "B3g"),
        "Q_u": ("B1u", "B2u", "B3u", "Au"),
    },
}


def orbsym_from_chk(chk: Path) -> tuple[list[str], object, np.ndarray, int]:
    from pyscf import scf, symm
    from pyscf.scf import chkfile

    mol = chkfile.load_mol(str(chk))
    scf_rec = chkfile.load(str(chk), "scf")
    mo_coeff = np.asarray(scf_rec["mo_coeff"])
    mo_occ = np.asarray(scf_rec["mo_occ"])
    if not getattr(mol, "symmetry", None):
        raise SystemExit(
            f"{chk} was built without point-group symmetry; regenerate with "
            "make_pyscf_hamiltonian.py --point_group"
        )
    labels = [str(s) for s in
              symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)]
    del scf
    return labels, mol, mo_occ, mo_coeff.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chk", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--print_sector", action="store_true",
                    help="also print the RHF reference sector label")
    args = ap.parse_args()

    chk = Path(args.chk)
    labels, mol, mo_occ, norb = orbsym_from_chk(chk)
    pg = str(mol.groupname).lower()
    if pg not in GENERATORS:
        raise SystemExit(f"no document generator recipe for point group {pg!r}")

    rows, names, supports = [], [], []
    for gname, irreps in GENERATORS[pg].items():
        idx = [i for i, s in enumerate(labels) if s in irreps]
        if not idx:
            print(f"[warn] {gname}: no orbitals of irrep {irreps} -- skipped")
            continue
        row = np.zeros(norb, dtype=int)
        row[idx] = 1
        rows.append(row)
        names.append(gname)
        supports.append(idx)

    mat = np.asarray(rows, dtype=int) if rows else np.zeros((0, norb), dtype=int)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, mat, fmt="%d")

    print(f"[ok] {chk.name}  pg={mol.groupname}")
    print(f"     orbsym  = {labels}")
    for n, s in zip(names, supports):
        print(f"     {n:8s} = {s}")
    print(f"[ok] wrote {out} shape={mat.shape}")

    if args.print_sector:
        # RHF reference: doubly occupied orbitals -> every Q has even count -> +1
        occ = [i for i, o in enumerate(np.atleast_1d(mo_occ)) if o > 1.5]
        bits = [len(set(occ) & set(s)) % 2 for s in supports]   # per spin
        spin = (mol.nelec[0] % 2, mol.nelec[1] % 2)
        sector = list(spin) + [0 for _ in bits]   # Q = (-1)^(n_a+n_b) = +1
        print(f"[ok] RHF occupied = {occ}")
        print(f"[ok] per-spin parities = {bits} (both spins equal => all Q=+1)")
        print(f"[ok] pass this to the drivers:  --exact_sector "
              f"{','.join(map(str, sector))}")


if __name__ == "__main__":
    main()
