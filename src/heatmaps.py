"""Utilities for computing and visualizing heatmap representations of
quasisymmetry commutator-based, variance-based and eigenvalue equation-based scores.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# compare with show_symmetries.py
"""Computes a norb×norb matrix of noncommutativity scores ||[H, O]|psi>||^2 for diagonal and off-diagonal symmetry operators, with optional scaling factor for diagonal entries."""
def get_heatmap_data_nc_score(h, ref_state, norb, diag_operators, off_diag_operators, upscale_factor=1):
    # Initialize heatmap data
    nc_scores = np.zeros((norb, norb))

    # precompute H|psi>
    h_on_ref_state = h @ ref_state

    # Diagonal: Possibility to scale up with upscale_factor, for better visualization
    for i in range(norb):
        term1 = h @ (diag_operators[i] @ ref_state)
        term2 = diag_operators[i] @ h_on_ref_state # using the wrapper variable type. Commutator has same/related type
        commutator_on_state = term1 - term2
        nc_scores[i, i] = upscale_factor * np.linalg.norm(commutator_on_state)**2

    # Off-diagonal
    # Need to arefully handle indices (e.g., do inverse of index mapping of what is done in cluster_number.build_two_orb_num_operators)
    for i in range(norb):
        for j in range(i+1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1 # checked
            op_ij = off_diag_operators[flat_index]
            term1 = h @ (op_ij @ ref_state)
            term2 = op_ij @ h_on_ref_state
            commutator_on_state = term1 - term2
            nc_scores[i, j] = np.linalg.norm(commutator_on_state)**2

    return nc_scores

"""Computes a norb×norb matrix of variance scores <psi | O^2 |psi> - <psi|O|psi>^2."""
def get_heatmap_data_variance_score(ref_state, norb, diag_operators, off_diag_operators, upscale_factor=1):
    variance_scores = np.zeros((norb, norb))

    for i in range(norb):
        variance_scores[i, i] = upscale_factor * ( (ref_state.T.conj() @ (diag_operators[i] @ (diag_operators[i] @ ref_state))) - ((ref_state.T.conj() @ (diag_operators[i] @ ref_state)) ** 2).real)

    for i in range(norb):
        for j in range(i + 1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1
            op_ij = off_diag_operators[flat_index]
            variance_scores[i, j] = (ref_state.T.conj() @ (op_ij @ (op_ij @ ref_state)) - (ref_state.T.conj() @ (op_ij @ ref_state)) ** 2).real

    return variance_scores

"""Computes a norb×norb matrix of eigenvalue equation-based scores ||O|psi> - lambda|psi>||^2."""
def get_heatmap_data_eval_eq_score(ref_state, norb, diag_operators, off_diag_operators, diag_evals, off_diag_evals, upscale_factor=1):
    """Returns matrix of eigenvalue equation-based scores, with entries ||(quasisymmetry operator - eval * I)|Ψ>||²"""
    # Initialize heatmap data
    eval_scores = np.zeros((norb, norb))

    # Diagonal: Possibility to scale up with upscale_factor, for better visualization
    for i in range(norb):
        eval_scores[i, i] = upscale_factor * np.linalg.norm(diag_operators[i] @ ref_state - diag_evals[i] * ref_state)**2

    # Off-diagonal
    # Need to arefully handle indices (e.g., do inverse of index mapping of what is done in cluster_number.build_two_orb_num_operators)
    for i in range(norb):
        for j in range(i+1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1 # checked
            op_ij = off_diag_operators[flat_index]
            off_diag_eval = off_diag_evals[flat_index]
            eval_scores[i, j] = np.linalg.norm(op_ij @ ref_state - off_diag_eval * ref_state)**2

    return eval_scores

# compare with show_symmetries.py
"""Visualizes a quasisymmetry score matrix as a log-scale heatmap with colorbar, value annotations (upper triangle), and customizable bounds and title."""
def show_heatmap(nc_scores, vmin=None, vmax=None, title='Quasisymmetry discovery'):
    """Visualize the non-commutativity heatmap"""
    norb = np.shape(nc_scores)[0]
    plt.figure(figsize=(8, 6))
    plt.imshow(nc_scores, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    plt.colorbar(label='Non-commutativity norm (log scale)')
    plt.title(title)

    # Add text annotations for exact values
    for i in range(norb):
        for j in range(norb):
            if i <= j:  # Only show upper triangle
                plt.text(j, i, f'{nc_scores[i,j]:.2e}',
                        ha='center', va='center', color='white' if nc_scores[i,j] > 1e-2 else 'black')

    plt.tight_layout()
    plt.show()