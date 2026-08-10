"""Run SCF for a given geometry and save a PySCF checkpoint file.

Example::

    python make_pyscf_hamiltonian.py h2o 0.958 --basis sto-3g
    python make_pyscf_hamiltonian.py h2o 0.958 --basis sto-3g --point_group C2v
    python make_pyscf_hamiltonian.py n2 1.1 --basis 6-31g --point_group D2h

``--point_group`` enables PySCF molecular symmetry so the checkpoint carries
irrep-adapted MOs (needed for ``optimize_*.py --orbital_rotation irrep``).
Incompatible with ``--localized`` (Pipek–Mezey), which breaks irrep blocks.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyscf
from pyscf import lo
from pyscf.lib import chkfile as pyscf_chk

from chemistry import get_geometry_and_description
from src.orbital_rotation import load_orbital_irreps, save_orbsym_sidecar


def _atomic_chk_path(final: str) -> tuple[str, str]:
    """Write SCF to a per-PID temp chk, then os.replace onto ``final``.

    Avoids HDF5 lock races when many array tasks target the same geometry.
    """
    final_path = Path(final)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}"))
    return tmp, str(final_path)


def _persist_orbsym(chk_path: str, mf) -> np.ndarray:
    """Store MO irrep ids in HDF5 ``/orbsym`` and a ``.orbsym.npy`` sidecar."""
    if not hasattr(mf, "get_orbsym"):
        raise RuntimeError(
            "SCF object has no get_orbsym(); use a point-group adapted RHF"
        )
    orbsym = np.asarray(mf.get_orbsym(), dtype=np.int32).ravel()
    if int(np.unique(orbsym).size) < 2:
        raise RuntimeError(f"MO irreps not distinct: {orbsym}")
    pyscf_chk.save(chk_path, "orbsym", orbsym)
    save_orbsym_sidecar(chk_path, orbsym)
    return orbsym


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RHF and write a PySCF .chk under hamiltonians/"
    )
    parser.add_argument(
        "mol",
        help="one of: lih, h2o, h4_linear, h4_square, h4_rectangle, h2, n2",
    )
    parser.add_argument("bond", type=float, help="bond length (angstrom)")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--mol_parameter_2",
        type=float,
        help="Additional geometry parameter (e.g. H2O H–O–H angle in degrees)",
    )
    parser.add_argument("--localized", action="store_true")
    parser.add_argument(
        "--point_group",
        default=None,
        metavar="GROUP",
        help=(
            "Enable PySCF point-group symmetry (e.g. C2v, D2h, or auto). "
            "Produces symmetry-adapted MOs for irrep-restricted orbital rotations. "
            "Incompatible with --localized."
        ),
    )

    args = parser.parse_args()

    if args.localized and args.point_group is not None:
        raise SystemExit(
            "--localized (Pipek–Mezey) breaks irrep blocks; "
            "do not combine it with --point_group"
        )

    if args.mol == "h2o":
        geometry, description = get_geometry_and_description(
            args.mol, args.bond, hoh_angle_deg=args.mol_parameter_2
        )
    else:
        geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = pyscf.M()
    pg_tag = ""
    if args.point_group is not None:
        group = args.point_group.strip()
        mol.symmetry = True if group.lower() == "auto" else group
        # Separate chk from non-symmetry builds so irrep jobs do not clobber
        # the shared default hamiltonians used by --orbital_rotation full.
        pg_tag = "_auto" if group.lower() == "auto" else f"_{group}"
    mol.build(atom=geometry, basis=args.basis)

    mf = pyscf.scf.RHF(mol)
    if not args.localized:
        final = "hamiltonians/" + description + str(args.basis) + pg_tag + ".chk"
        tmp, final = _atomic_chk_path(final)
        mf.chkfile = tmp
        orbsym = None
        try:
            mf.kernel()
            if args.point_group is not None:
                orbsym = _persist_orbsym(tmp, mf)
            os.replace(tmp, final)
            # Sidecar was written next to tmp; move/rewrite next to final.
            if orbsym is not None:
                save_orbsym_sidecar(final, orbsym)
                for side in (
                    Path(tmp + ".orbsym.npy"),
                    Path(tmp + ".orbsym.txt"),
                    Path(tmp).with_suffix(Path(tmp).suffix + ".orbsym.npy"),
                ):
                    if side.is_file():
                        try:
                            side.unlink()
                        except OSError:
                            pass
        finally:
            if os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            for side in (
                Path(tmp + ".orbsym.npy"),
                Path(tmp + ".orbsym.txt"),
                Path(tmp).with_suffix(Path(tmp).suffix + ".orbsym.npy"),
            ):
                if side.is_file():
                    try:
                        side.unlink()
                    except OSError:
                        pass
        mf.chkfile = final
        if args.point_group is not None:
            print("point group:", mol.groupname)
            print("chkfile:", mf.chkfile)
            print("MO irreps:", orbsym)
            loaded = load_orbital_irreps(final)
            if loaded is None:
                raise SystemExit(
                    f"round-trip load_orbital_irreps failed for {final}"
                )
            print("reload irreps:", loaded)
            print("sidecar txt:", str(Path(final)) + ".orbsym.txt")
    else:
        final = "hamiltonians/" + description + str(args.basis) + "_Pipek.chk"
        tmp, final = _atomic_chk_path(final)
        mf.chkfile = tmp
        try:
            mf.kernel()
            os.replace(tmp, final)
        finally:
            if os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        mf.chkfile = final
        localizer = lo.PipekMezey(mol, mf.mo_coeff[:, mf.mo_occ > 0])
        loc_orbs_occ = localizer.kernel()

        mf.mo_coeff[:, mf.mo_occ > 0] = loc_orbs_occ
        print(mf.mo_coeff)
        plt.imshow(mf.mo_coeff, cmap="PuOr", vmin=-1, vmax=1)
        plt.yticks(range(mf.mo_coeff.shape[0]), mol.ao_labels())
        plt.show()
