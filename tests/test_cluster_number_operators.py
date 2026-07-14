"""Tests for functions in src/cluster_number_operators.py on cluster numbers as quasisymmetries"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pytest
import scipy
import ffsim
from src.cluster_number_operators import build_one_orb_num_operators, build_two_orb_num_operators, number_matrix_to_operators, from_num_operator_to_expnum_operator, number_and_parity_symmetry_sectors, integers_to_phases_polynomial
from scipy.sparse.linalg import LinearOperator
from src.sector_utils import symmetry_sectors

def test_spin_orbital_occupations():
    """Testing to refresh scipy/ffsim basis ordering and operator usage"""
    norb = 3
    nelec = (1, 2) # alpha, beta
    # generate spin-orbital number operators
    alpha_number_operators = []
    beta_number_operators = []
    for i in range(norb):
        n_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): +1 # 1 * a_ialpha^dag a_ialpha
            }
        )
        n_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): +1 # 1 * a_ibeta^dag a_ibeta
            }
        )
        alpha_number_operators.append(ffsim.linear_operator(n_alpha, norb, nelec))
        beta_number_operators.append(ffsim.linear_operator(n_beta, norb, nelec))

    spin_orb_number_operators = alpha_number_operators + beta_number_operators

    # expected basis ordering: |alpha occupations -> 100; beta occupations -> 110>, |100; 101>, ...
    expected_occupations = [[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1], 
                            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], 
                            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]]
    # later, when using bitstrings, watch out for possible (intra-alpha/beta) flips.
    # get actual basis ordering
    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in spin_orb_number_operators]
        assert list(occupations) == expected_occupations[i]

def test_build_oneortwo_orb_num_operators():
    norb = 3
    nelec = (1, 2) # alpha, beta
    # get list of one-orbital alpha+beta occ number operators
    one_orbital_num_operators = build_one_orb_num_operators(norb, nelec)

    expected_occupations = [[2, 1, 0], [2, 0, 1], [1, 1, 1], 
                            [1, 2, 0], [1, 1, 1], [0, 2, 1], 
                            [1, 1, 1], [1, 0, 2], [0, 1, 2]]
    
    assert len(one_orbital_num_operators) == norb

    assert type(one_orbital_num_operators[0]) == scipy.sparse.linalg._interface._CustomLinearOperator

    assert np.allclose(one_orbital_num_operators[0] @ np.eye(9)[0] - 2 * np.eye(9)[0], np.zeros(9))

    two_orbital_num_operators = build_two_orb_num_operators(norb, nelec) # orbital pairs: 01, 02, 12

    assert len(two_orbital_num_operators) == (norb**2 - norb)/2

    expected_two_orb_occupations = [[3, 2, 1], [2, 3, 1], [2, 2, 2], 
                                [3, 1, 2], [2, 2, 2], [2, 1, 3], 
                                [2, 2, 2], [1, 3, 2], [1, 2, 3]]

    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in one_orbital_num_operators]
        two_orb_occupations = [np.vdot(basis_elem, op @ basis_elem) for op in two_orbital_num_operators]
        assert list(occupations) == expected_occupations[i]
        assert list(two_orb_occupations) == expected_two_orb_occupations[i]

def test_integers_to_phases_polynomial():
    def P(x, N):
        coeffs = integers_to_phases_polynomial(N)
        result = sum([coeffs[i] * x**i for i in range(N+1)])
        return(result)

    for N in range(10):
        for n in range(N + 1):
            omega = np.exp(1j * 2 * np.pi / (N+1))
            assert (omega ** n - P(n, N)).round(10) == 0

def test_from_num_operator_to_expnum_operator():
    integers = [0, 4, 6, 31, 19, 2]
    matrix = np.diag(integers)
    max_num_eval = max(integers)
    dim = len(integers)
    exp_integers = [np.exp(1.j * 2 * np.pi * n / (max_num_eval+1)) for n in integers]
    num_operator = LinearOperator((dim, dim), matvec=lambda v: matrix @ v)
    expnum_operator = from_num_operator_to_expnum_operator(num_operator, max_num_eval)
    for i in range(dim):
        basis_el = np.eye(dim)[i]
        assert np.allclose(expnum_operator @ basis_el, exp_integers[i] * basis_el, atol = 1e-15, rtol = 1e-10)

def test_number_matrix_to_operators():
    norb = 3
    nelec = (1, 2)
    cluster_matrix = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1]])
    # number operators
    cluster_num_operators = number_matrix_to_operators(cluster_matrix, norb, nelec)
    expected_cluster_num_operators = [build_one_orb_num_operators(norb, nelec)[0], build_two_orb_num_operators(norb, nelec)[1], LinearOperator((9, 9), matvec=lambda v: 3 * v)]
    # exponentiated versions
    cluster_expnum_operators = number_matrix_to_operators(cluster_matrix, norb, nelec, expnum=True)
    expected_cluster_expnum_operators = [from_num_operator_to_expnum_operator(expected_cluster_num_operators[i], 2 * sum(cluster_matrix[i])) for i in range(3)]
    for i in range(9):
        basis_elem = np.eye(9)[i]
        for j in range(3):
            assert np.allclose(cluster_num_operators[j] @ basis_elem, expected_cluster_num_operators[j] @ basis_elem)
            assert np.allclose(cluster_expnum_operators[j] @ basis_elem, expected_cluster_expnum_operators[j] @ basis_elem)

def test_number_and_parity_symmetry_sectors():
    # both parities and numbers
    norb = 3
    nelec = (1, 2)
    cluster_number_matrix = np.array([[1, 0, 0], [1, 0, 1]])
    cluster_parity_matrix = np.array([[0, 0, 1]])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    expected_sectors = {((2,2), (0,)): [0],
                        ((2,3), (1,)): [1],
                        ((1,2), (1,)): [2,4,6],
                        ((1,1), (0,)): [3],
                        ((0,1), (1,)): [5],
                        ((1,3), (0,)): [7],
                        ((0,2), (0,)): [8]}
    assert sectors == expected_sectors
    
    # only parities
    norb = 4
    nelec = (2, 3)
    cluster_number_matrix = np.array([])
    cluster_parity_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    aleksey_sectors = symmetry_sectors(cluster_parity_matrix, norb, nelec)
    expected_sectors = {((), k): v for k, v in aleksey_sectors.items()}
    assert sectors == expected_sectors

    # only numbers
    norb = 3
    nelec = (1, 1)
    cluster_number_matrix = np.array([[0, 1, 1]])
    cluster_parity_matrix = np.array([])
    sectors = number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)
    expected_sectors = {((0,), ()): [0],
                        ((1,), ()): [1,2,3,6],
                        ((2,), ()): [4,5,7,8]}
    assert sectors == expected_sectors