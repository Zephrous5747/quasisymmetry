"""Generate fermionic Hamiltonians which have known exact symmetries.
Construct the symmetries from local parities, then fill eligible entries with random stuff,
enforce the 8-fold symmetry of the tensor, then rotate by a random SO(norb) orbital rotation.

The parity of a spin-orbital k is (-1)^(n_k). A multi-orbital parity is a product of these.
Each (multi-orbital) parity is a product of local parities. A collection of parities can then be
described by an incidence matrix, where rows are parities, columns are spin-orbitals.

Throughout the module the functions deal with condensed parity matrices. We assume that for each
molecular orbital, its alpha column is the same as its beta column."""


from math import comb

import ffsim
import numpy as np
import argparse
from scipy.stats import special_ortho_group
from uuid import uuid4
import pyscf
import networkx as nx

from itertools import product

from cost_functions import rotation_to_x, x_to_rotation
from optimize import SENIORITY_ANGLES


def generic_spinful_h(condensed_parity_table, rng=None):
    """Assuming that every alpha column of the parity table is equal to its beta counterpart"""
    if rng is None:
        rng = np.random.default_rng()
    norb = condensed_parity_table.shape[1]
    one_body_tensor = np.zeros((norb, norb))
    two_body_tensor = np.zeros((norb, norb, norb, norb))

    for p, q in product(range(norb), repeat=2):
        if (condensed_parity_table[:, p] == condensed_parity_table[:, q]).all():
            one_body_tensor[p, q] = rng.normal()

    one_body_tensor += one_body_tensor.T
    one_body_tensor = one_body_tensor / 2

    for p, q, r, s in product(range(norb), repeat=4):
        if ((condensed_parity_table[:, p] + condensed_parity_table[:, r]) % 2
            == (condensed_parity_table[:, s] + condensed_parity_table[:, q]) % 2).all():
            two_body_tensor[p, q, r, s] = rng.normal()

    two_body_tensor += np.transpose(two_body_tensor)
    two_body_tensor = two_body_tensor / 2

    tbt_compressed = pyscf.ao2mo.restore(8, two_body_tensor.real, norb)

    tbt_uncompressed = pyscf.ao2mo.restore(1, tbt_compressed, norb)

    h = ffsim.hamiltonians.MolecularHamiltonian(
        one_body_tensor, tbt_uncompressed
    )

    return h


def edge_list_to_condensed_parity_matrix(edges: list, norb) -> np.array:
    parity_matrix = np.zeros((len(edges), norb), dtype=int)
    for i, e in enumerate(edges):
        parity_matrix[i, int(e[0])] = 1
        parity_matrix[i, int(e[1])] = 1
    return parity_matrix


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate test Hamiltonians with known orbital parity symmetries",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--norb", help="Number of atomic orbitals; \n"
                                       "the number of spin-orbitals is 2x that. \n"
                                       "Required when h_type is 'seniority'",
                        type=int)
    parser.add_argument("outname")
    parser.add_argument("--h_type",
                        help="type of Hamiltonian to generate",
                        default="seniority")
    parser.add_argument("--seed", help="RNG seed",
                        default=None, type=int)
    parser.add_argument("--no_U", action="store_true")
    parser.add_argument("--U_close_to_id", action="store_true")
    parser.add_argument("--edgelist",
                        help="path to a list of edges. Required when h_type='parity_graph'.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.h_type == "seniority":
        condensed_parity = np.eye(args.norb, dtype=int)
        norb = args.norb
    elif args.h_type == "parity_graph":
        G = nx.read_edgelist(args.edgelist)
        condensed_parity = edge_list_to_condensed_parity_matrix(list(G.edges), G.number_of_nodes())
        norb = G.number_of_nodes()
    else:
        raise ValueError("h_type must be 'seniority' or 'parity_graph'")

    h = generic_spinful_h(condensed_parity, rng)

    if args.no_U:
        U = np.eye(norb)
    elif args.U_close_to_id:
        x0 = rng.normal(size=comb(norb, 2), scale=1e-3)
        U = x_to_rotation(x0, norb)
    else:
        U = special_ortho_group.rvs(dim=norb, random_state=rng)

    rotated_h = h.rotated(U)

    rot_moldata = ffsim.MolecularData(
        spin=0,
        nelec=(norb // 2, norb // 2),
        one_body_integrals=rotated_h.one_body_tensor.real,
        two_body_integrals=rotated_h.two_body_tensor.real,
        norb=norb,
        core_energy=0,
        mo_occ=np.array([2.] * (norb // 2) + [0.] * (norb - norb // 2))
    )
    x = rotation_to_x(U.T) # the optimizer will have to rotate H back to the original frame, hence the inverse

    x_with_sen = np.concatenate([x, SENIORITY_ANGLES])

    stamp = uuid4().hex[:4]
    h_name = args.outname + "_" + stamp

    rot_moldata.to_fcidump(h_name + ".FCIDUMP")
    np.savetxt(h_name + "_U.txt", U)
    np.savetxt(h_name + "_x0.txt", x)
    np.savetxt(h_name + "_x0_sen.txt", x_with_sen)