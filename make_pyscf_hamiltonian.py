"""Run SCF for a given geometry and save a PySCF checkpoint file"""

import pyscf
import argparse
import matplotlib.pyplot as plt

from pyscf import lo

from chemistry import  get_geometry_and_description

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
                        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle, h2, n2")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--mol_parameter_2", type=float,
                        help="Additional geometry parameter of the molecule (if any)")
    parser.add_argument("--localized", action="store_true")

    args = parser.parse_args()

    if args.mol=="h2o":
        geometry, description = get_geometry_and_description(args.mol, args.bond,
                                                             hoh_angle_deg=args.mol_parameter_2)
    else:
        geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = pyscf.M()
    mol.build(atom=geometry, basis=args.basis)

    mf = pyscf.scf.RHF(mol)
    if not args.localized:
        mf.chkfile = "hamiltonians/" + description + ".chk"
        mf.kernel()
    else:
        mf.chkfile = "hamiltonians/" + description + "_Pipek.chk"
        mf.kernel()
        # localizer = lo.Boys(mol, mf.mo_coeff[:, mf.mo_occ > 0])
        localizer = lo.PipekMezey(mol, mf.mo_coeff[:, mf.mo_occ > 0])
        loc_orbs_occ = localizer.kernel()

        mf.mo_coeff[:, mf.mo_occ > 0] = loc_orbs_occ
        print(mf.mo_coeff)
        plt.imshow(mf.mo_coeff, cmap="PuOr", vmin=-1, vmax=1)
        plt.yticks(range(mf.mo_coeff.shape[0]), mol.ao_labels())
        plt.show()

        # new_mo_coeff = mf.mo_coeff.copy()
        # new_mo_coeff = loc_orbs
        # mf.chkfile = "hamiltonians/" + description + "_LMO.chk"
        # pyscf.lib.chkfile.dump_scf(mol, "hamiltonians/" + description + "_LMO.chk",
        #                            mf.e_tot, mf.mo_energy,
        #                            mf.mo_coeff, mf.mo_occ)
        # pyscf.lib.chkfile.save_mol(mf, "hamiltonians/" + description + "_LMO.chk")


