#!/usr/bin/env python
"""Sec.4 exhaustive quotient-pool check for STO-3G H2O / N2.

Implements the control calculation of
``plans/Iterative_NC_Ranked_LAS_Search_Procedure_poly+exhaustive.pdf`` Sec.4:
instead of the polynomial current-frame library of N(N+1)/2 singles and pairs,
score EVERY nontrivial parity class modulo the exact symmetries,

    C_full = { gbar in F_2^(N-r) : gbar != 0 },   |C_full| = 2^(N-r) - 1,

redo the greedy independence scan on the full ranking, and compare the result
with the polynomial search at the same orbitals (Sec.4.3 items 1-4).

Cost identity
-------------
Z(g) is diagonal in the determinant basis with signs z_i = +-1, so

    ([H, Z(g)] Psi)_i = (H (z * Psi))_i - z_i (H Psi)_i,

and the NC score of Eq.(3) is the squared norm of that vector. H Psi is
computed once; each candidate then costs a single Hamiltonian application.

Scope
-----
``--scope spatial`` (default) enumerates the quotient of the SPATIAL parity
register F_2^n / E_spatial, which is the space the implemented search actually
selects from (LAS are Zbar_p = Z_p,alpha Z_p,beta products). Sizes: 31 (H2O),
127 (N2).

``--scope qubit`` enumerates the PDF's full register F_2^(2n) / E_qubit:
1023 (H2O) and 32767 (N2). This is the literal Sec.4.1 pool and it is strictly
larger than anything the current selection code can return, because
spin-resolved rows such as Z_p,alpha alone are not representable as spatial
Zbar products. Running it therefore also measures how much of the PDF candidate
space the implementation cannot reach.

What this does and does not do
------------------------------
This is the Sec.4.3 VALIDATION at the polynomial search's converged orbitals:
it removes the restricted-neighbourhood approximation of Sec.7.1 at fixed U.
It is NOT the full Sec.4.2 self-consistent exhaustive fixed point, which would
re-optimise the orbitals against the exhaustive pool and re-rank until the LAS
span stops moving. Use --selfconsistent-note to print what remains.

Usage
-----
    python scripts/exhaustive_quotient_check.py \
        --oo results/n2_endpoint_grid/bond_1p8000/U_irrep/iterative/NC/oo.json

    python scripts/exhaustive_quotient_check.py --oo ... --scope qubit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "audit"))

from _qs_json import load_oo                                    # noqa: E402
from src.gf2_utils import gf2_int_rref, gf2_int_try_add_to_span  # noqa: E402


# --------------------------------------------------------------------------
# GF(2) helpers
# --------------------------------------------------------------------------
def in_span(mask: int, rref: list[int]) -> bool:
    v = int(mask)
    for p in rref:
        v = min(v, v ^ p)
    return v == 0


def coset_representatives(n_bits: int, exact_rref: list[int]) -> list[int]:
    """One representative per nonzero class of F_2^n / span(exact).

    Canonical choice: the lexicographically smallest member of each coset,
    obtained by reducing every vector against the exact RREF and keeping the
    distinct reduced values.
    """
    seen: set[int] = set()
    reps: list[int] = []
    for v in range(1, 1 << n_bits):
        red = v
        for p in exact_rref:
            red = min(red, red ^ p)
        if red == 0 or red in seen:
            continue
        seen.add(red)
        reps.append(red)
    return reps


# --------------------------------------------------------------------------
# Hamiltonian / state in the rotated frame
# --------------------------------------------------------------------------
def build_rotated_problem(chk_path: str, rotation_params, orbital_rotation, norb,
                          *, transpose: bool = False,
                          reference: str = "fci_rotate", root: int = 0):
    from pyscf import ao2mo, fci, scf
    from pyscf.scf import chkfile

    from src.orbital_rotation import params_to_U, resolve_orbital_rotation

    mol = chkfile.load_mol(str(chk_path))
    mo = np.asarray(chkfile.load(str(chk_path), "scf")["mo_coeff"])
    nelec = (mol.nelectron // 2, mol.nelectron // 2)

    pairs, _irreps = resolve_orbital_rotation(
        str(orbital_rotation or "full"), str(chk_path), norb)
    x = np.asarray(rotation_params, dtype=float) if rotation_params is not None \
        else np.zeros(0)
    U = params_to_U(x, norb, pairs) if x.size else np.eye(norb)
    # ffsim's MolecularHamiltonian.rotated(U) convention is the TRANSPOSE of
    # mo @ U (established empirically: every point whose gate passes selects
    # mo @ U.T). E_FCI is invariant under either, so only the NC/variance
    # scores can distinguish them.
    if transpose == "identity":
        mo_rot = mo[:, :norb]
    else:
        mo_rot = mo[:, :norb] @ (U.T if transpose else U)

    hcore = scf.hf.get_hcore(mol)
    h1e = mo_rot.T @ hcore @ mo_rot
    eri = ao2mo.kernel(mol, mo_rot)
    ecore = float(mol.energy_nuc())

    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)

    # Reference state. The runs use iterative_reference=fci_rotate: FCI is
    # solved ONCE in the original MO basis and the CI vector is then rotated.
    # Re-solving FCI in the rotated basis gives the same state only when the
    # ground state is non-degenerate; at stretched H2O (R = 2.3-2.5 A) the
    # low-lying manifold is near-degenerate and the two routes can land on
    # different vectors, which is what broke the gate at those geometries.
    R = np.eye(norb) if transpose == "identity" else (U.T if transpose else U)
    if reference == "fci_rotate":
        h1e0 = mo[:, :norb].T @ hcore @ mo[:, :norb]
        eri0 = ao2mo.kernel(mol, mo[:, :norb])
        e_list, ci_list = fci.direct_spin1.kernel(
            h1e0, eri0, norb, nelec, ecore=ecore, nroots=max(2, root + 1))
        e0 = float(np.atleast_1d(e_list)[0])
        gap = float(np.atleast_1d(e_list)[1] - np.atleast_1d(e_list)[0])
        _roots = ci_list if isinstance(ci_list, (list, tuple)) else [ci_list]
        ci0 = np.asarray(_roots[min(root, len(_roots) - 1)])
        civec = fci.addons.transform_ci_for_orbital_rotation(
            ci0, norb, nelec, R)
        e_fci = e0
    else:
        e_list, ci_list = fci.direct_spin1.kernel(
            h1e, eri, norb, nelec, ecore=ecore, nroots=max(2, root + 1))
        e_fci = float(np.atleast_1d(e_list)[0])
        gap = float(np.atleast_1d(e_list)[1] - np.atleast_1d(e_list)[0])
        _roots = ci_list if isinstance(ci_list, (list, tuple)) else [ci_list]
        civec = np.asarray(_roots[min(root, len(_roots) - 1)])
    civec = np.asarray(civec)
    civec = civec / np.linalg.norm(civec)
    return dict(mol=mol, nelec=nelec, h2e=h2e, ci=civec, e_fci=float(e_fci),
                ecore=ecore, norb=norb, gap=gap)


def sign_tables(norb, nelec, scope):
    """Per-string occupation bit patterns for fast sign construction."""
    from pyscf.fci import cistring
    sa = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sb = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)
    return sa, sb


def parity_of(strings: np.ndarray, mask: int) -> np.ndarray:
    """popcount(string & mask) mod 2, vectorised."""
    v = strings & np.int64(mask)
    par = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        par ^= (v & 1)
        v >>= 1
    return par


def _z_signs(g, sa, sb, norb, scope):
    if scope == "spatial":
        pa, pb = parity_of(sa, g), parity_of(sb, g)
    else:
        ma = mb = 0
        for p in range(norb):
            if (g >> (2 * p)) & 1:
                ma |= 1 << p
            if (g >> (2 * p + 1)) & 1:
                mb |= 1 << p
        pa, pb = parity_of(sa, ma), parity_of(sb, mb)
    return 1.0 - 2.0 * ((pa[:, None] ^ pb[None, :]).astype(float))


def nc_scores(problem, reps, scope, cost_function="NC"):
    """Score every candidate with the objective the run used.

    NC       : ||[H, Z(g)] Psi||^2                      (one H application each)
    variance : 1 - <Psi|Z(g)|Psi>^2, matching
               src/metrics.py::variance. No H application needed.

    Mixing these silently was the reason 63/63 variance points failed the gate:
    their recorded selected_costs are variance values (one was -6.2e-15, which
    a squared norm can never be).
    """
    from pyscf import fci

    norb, nelec = problem["norb"], problem["nelec"]
    ci, h2e = problem["ci"], problem["h2e"]
    sa, sb = sign_tables(norb, nelec, scope)

    if str(cost_function).lower().startswith("var"):
        p2 = np.abs(ci) ** 2
        out = np.empty(len(reps), dtype=float)
        for k, g in enumerate(reps):
            expz = float(np.sum(_z_signs(g, sa, sb, norb, scope) * p2))
            out[k] = 1.0 - expz * expz
        return out

    hci = fci.direct_spin1.contract_2e(h2e, ci, norb, nelec)

    out = np.empty(len(reps), dtype=float)
    t0 = time.time()
    for k, g in enumerate(reps):
        z = _z_signs(g, sa, sb, norb, scope)
        zc = z * ci
        comm = fci.direct_spin1.contract_2e(h2e, zc, norb, nelec) - z * hci
        out[k] = float(np.sum(comm * comm))
        if (k + 1) % 2000 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"    scored {k+1}/{len(reps)}  ({rate:.0f}/s)", flush=True)
    return out


# --------------------------------------------------------------------------
# Greedy independence scan on the full ranking (Sec.4.2 step 2)
# --------------------------------------------------------------------------
def ranked_scan(reps, scores, exact_rref, n_bits, M, target):
    order = sorted(range(len(reps)),
                   key=lambda i: (float(scores[i]), int(reps[i])))
    rref = list(exact_rref)
    accepted, acc_cost, trace = [], [], []
    first_rejected_independent = None
    for pos in order:
        g = int(reps[pos])
        new = gf2_int_try_add_to_span(g, rref, n_bits)
        if new is None:
            reason = ("exact_parity" if in_span(g, exact_rref)
                      else "gf2_dependent_or_exact_dressed")
            trace.append({"event": "reject", "reason": reason,
                          "mask": g, "cost": float(scores[pos])})
            continue
        if len(accepted) >= target and first_rejected_independent is None:
            first_rejected_independent = {"mask": g, "cost": float(scores[pos])}
        rref = new
        accepted.append(g)
        acc_cost.append(float(scores[pos]))
        trace.append({"event": "accept_las" if len(accepted) <= M
                      else "accept_auxiliary",
                      "mask": g, "cost": float(scores[pos])})
        if len(accepted) >= target:
            break
    # the best independent candidate NOT taken into the first M
    boundary = None
    if len(accepted) > M:
        boundary = {"mask": accepted[M], "cost": acc_cost[M]}
    return dict(order_size=len(order), accepted=accepted,
                accepted_costs=acc_cost, las=accepted[:M],
                las_costs=acc_cost[:M], auxiliary=accepted[M:],
                first_rejected_competitor=boundary,
                first_independent_beyond_basis=first_rejected_independent,
                trace=trace)


def span_signature(masks, n_bits):
    rows = [int(m) for m in masks if int(m)]
    rref, _ = gf2_int_rref(rows, n_bits)
    return sorted(int(r) for r in rref)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oo", required=True, help="oo.json of the polynomial run")
    ap.add_argument("--scope", default="spatial", choices=("spatial", "qubit"))
    ap.add_argument("--reference-root", type=int, default=0,
                    help="which FCI root to use as the reference. At "
                         "near-degenerate geometries (E1-E0 < 1 mHa, e.g. H2O "
                         "R=2.5 A) the ground state is not uniquely defined by "
                         "energy; rerun with --reference-root 1 to check "
                         "whether the span conclusion is robust to the choice")
    ap.add_argument("--reference", default="fci_rotate",
                    choices=("fci_rotate", "resolve"),
                    help="fci_rotate reproduces the runs exactly (FCI in the "
                         "ORIGINAL basis, CI vector then rotated); resolve "
                         "re-solves FCI in the rotated basis")
    ap.add_argument("--u-convention", default="auto",
                    choices=("auto", "direct", "transpose"),
                    help="mo @ U vs mo @ U.T; auto picks whichever reproduces "
                         "the recorded NC scores")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    oo_path = Path(args.oo)
    oo = load_oo(oo_path)
    molpath = oo.get("molpath")
    norb = 7 if "h2o" in str(oo_path).lower() or "H2O" in str(molpath) else 10
    exact_spatial = [int(m) for m in (oo.get("exact_masks") or [])]
    # Iterative stores las_masks / accumulated_masks. The greedy/mixed path
    # stores neither -- its rows live in the parity matrix written to
    # parity_output (oo_parity.txt). Without this fallback las_poly was empty
    # for all 42 mixed_* NC points and the gate printed an empty table with
    # rel err = inf.
    las_poly_spatial = [int(m) for m in (oo.get("las_masks")
                                         or oo.get("accumulated_masks") or [])]
    if not las_poly_spatial:
        from src.exact_parity import masks_from_parity_matrix
        for cand in (oo.get("parity_output"), oo.get("parity"),
                     oo_path.parent / "oo_parity.txt"):
            if cand and Path(cand).is_file():
                mat = np.atleast_2d(np.loadtxt(str(cand), dtype=int))
                las_poly_spatial = [int(m) for m in masks_from_parity_matrix(mat)]
                print(f"[exh] LAS rows recovered from {cand}")
                break
    if not las_poly_spatial:
        raise SystemExit("[exh] could not resolve the polynomial LAS rows")

    cost_function = str(oo.get("cost_function") or "NC")
    M = int(oo.get("M") or oo.get("n_sym") or len(las_poly_spatial))

    print(f"[exh] point   : {oo_path}")
    print(f"[exh] molpath : {molpath}")
    print(f"[exh] norb={norb}  M={M}  scope={args.scope}  "
          f"cost={cost_function}")

    if args.scope == "spatial":
        n_bits = norb
        exact = exact_spatial
        las_poly = las_poly_spatial
    else:
        from src.sto3g_exact_symmetries import (
            expand_spatial_mask_to_interleaved_qubits,
            interleaved_spin_number_masks)
        n_bits = 2 * norb
        pa, pb = interleaved_spin_number_masks(norb)
        exact = [pa, pb] + [expand_spatial_mask_to_interleaved_qubits(m, norb)
                            for m in exact_spatial]
        las_poly = [expand_spatial_mask_to_interleaved_qubits(m, norb)
                    for m in las_poly_spatial]

    exact_rref, _ = gf2_int_rref([m for m in exact if m], n_bits)
    r = len(exact_rref)
    target = n_bits - r
    reps = coset_representatives(n_bits, exact_rref)
    print(f"[exh] r={r}  N-r={target}  |C_full|={len(reps)}  "
          f"(expected {2 ** target - 1})")
    if len(reps) != 2 ** target - 1:
        print("[exh][warn] pool size differs from 2^(N-r)-1; check the exact rows")

    # ---- pick the rotation convention that reproduces the recorded NC ------
    poly_recorded = [float(c) for c in (oo.get("selected_costs") or [])]

    def gate_error(prob):
        if not las_poly or not poly_recorded:
            return float("inf"), np.array([])
        mine = nc_scores(prob, las_poly, args.scope, cost_function)
        errs = []
        for i, m_ in enumerate(las_poly):
            if i >= len(poly_recorded):
                break
            rec = poly_recorded[i]
            if rec > 1e-12:
                errs.append(abs(float(mine[i]) - rec) / rec)
        return (max(errs) if errs else float("inf")), mine

    # The greedy/mixed path selects its quota ONCE at the initial frame and only
    # then hands off to the orbital optimiser, so its selected_costs are
    # SELECTION-TIME values at the UNROTATED orbitals. The iterative path
    # reselects every macroiteration, so its final costs sit at the final U.
    # The gate therefore has to try the identity frame too -- otherwise every
    # mixed_* point fails for a bookkeeping reason, not a physical one.
    trials = {"mo @ U.T": True, "mo @ U": False, "unrotated (selection-time)": "identity"}
    if args.u_convention == "direct":
        trials = {"mo @ U": False}
    elif args.u_convention == "transpose":
        trials = {"mo @ U.T": True}

    best = None
    for label, tr in trials.items():
        prob = build_rotated_problem(molpath, oo.get("rotation"),
                                     oo.get("orbital_rotation"), norb,
                                     transpose=tr, reference=args.reference,
                                     root=args.reference_root)
        err, mine = gate_error(prob)
        print(f"[exh] frame {label:30s}: max rel {cost_function} error "
              f"vs recorded = {err:.3e}")
        if best is None or err < best[0]:
            best = (err, tr, prob, mine, label)
    gate_err, use_T, gate_problem, poly_here, gate_label = best
    gate_ok = gate_err < 1e-2
    print(f"[exh] recorded costs reproduced in frame: {gate_label}")
    if use_T == "identity":
        print("[exh] -> confirms selected_costs are PRE-OO for this method; "
              "the exhaustive ranking below still uses the FINAL orbitals, "
              "which is what Sec.4.3 compares.")

    # The Sec.4.3 comparison must always be at the converged orbitals.
    problem = build_rotated_problem(molpath, oo.get("rotation"),
                                    oo.get("orbital_rotation"), norb,
                                    transpose=(True if use_T == "identity"
                                               else use_T),
                                    reference=args.reference,
                                    root=args.reference_root)
    print(f"[exh] scoring pool at FINAL U   "
          f"E_FCI = {problem['e_fci']:.10f}   "
          f"E1-E0 gap = {problem.get('gap', float('nan')):.6f} Ha")
    if problem.get("gap", 1.0) < 1e-3:
        print("[exh][warn] near-degenerate ground state: the reference is not "
              "uniquely defined by energy alone; cost comparisons at this "
              "geometry are sensitive to which vector the solver returned.")
    print("[exh] (E_FCI is rotation-invariant and cannot validate U; "
          "the cost gate can)")

    # free diagnostic for the pre-OO hypothesis
    tot = sum(poly_recorded) if poly_recorded else float("nan")
    print(f"[exh] sum(selected_costs)={tot:.8g}  "
          f"cost_before={oo.get('cost_before')}  cost_after={oo.get('cost_after')}")

    # The gate may have matched a different frame (mixed_* record pre-OO costs).
    # For the Sec.4.3 cost comparison both pools must be scored at the SAME
    # orbitals, so rescore the polynomial rows at the final U as well.
    poly_final = nc_scores(problem, las_poly, args.scope, cost_function)

    t0 = time.time()
    scores = nc_scores(problem, reps, args.scope, cost_function)
    print(f"[exh] scored {len(reps)} candidates in {time.time()-t0:.1f}s")

    # ---- SELF-CONSISTENCY GATE -------------------------------------------
    # The exhaustive pool is a SUPERSET of every mask the polynomial frame-local
    # scan can reach, so a minimum-NC scan over it can never return higher costs
    # than the polynomial did. If it does, this script is scoring a different
    # state or Hamiltonian -- not a Sec.4.3 candidate-space error.
    print("\n[exh] scoring gate: polynomial LAS re-scored with THIS code")
    print(f"{'mask':>8} {'recorded NC':>16} {'this code':>16} {'rel err':>10}")
    for i, m_ in enumerate(las_poly):
        rec_c = poly_recorded[i] if i < len(poly_recorded) else float("nan")
        mine = float(poly_here[i]) if len(poly_here) > i else float("nan")
        rel = abs(mine - rec_c) / rec_c if rec_c and rec_c > 1e-12 else float("nan")
        print(f"{m_:>8} {rec_c:16.10g} {mine:16.10g} {rel:10.3e}")
    print(f"[exh] scoring gate: {'PASS' if gate_ok else 'FAIL'} "
          f"(max rel err {gate_err:.3e})")
    if not gate_ok:
        print("[exh] *** Re-scored polynomial rows do not reproduce the "
              "recorded NC. The comparison below is NOT meaningful yet.")
        print("[exh]     Neither rotation convention matched, so the remaining "
              "suspects are the reference state (the run uses fci_rotate: FCI "
              "in the ORIGINAL basis then rotated) or the pair ordering used "
              "by params_to_U for this packing.")

    exh = ranked_scan(reps, scores, exact_rref, n_bits, M, target)

    # ---- Sec.4.3 comparison ------------------------------------------------
    # MUST be done modulo E. Sec.2(iii): if g' = g + e with e in E then g and g'
    # define the SAME approximate partition inside a fixed exact sector, and
    # Eq.(7) tests rank of (E u G), not of G alone. The exhaustive scan returns
    # canonical coset representatives while the polynomial pool stores raw
    # masks, so comparing span(G) directly reports a spurious disagreement.
    span_exh = span_signature([*exact, *exh["las"]], n_bits)
    span_poly = span_signature([*exact, *las_poly], n_bits)
    agree = span_exh == span_poly

    def reduce_mod_exact(m):
        v = int(m)
        for pr in exact_rref:
            v = min(v, v ^ pr)
        return v

    classes_exh = sorted(reduce_mod_exact(m) for m in exh["las"])
    classes_poly = sorted(reduce_mod_exact(m) for m in las_poly)
    same_classes = classes_exh == classes_poly

    poly_costs = [float(c) for c in (oo.get("selected_costs") or [])]
    # reps are canonical coset representatives; the polynomial masks are raw.
    # Look up ranks on the REDUCED masks or every lookup misses (this is the
    # same raw-vs-canonical mistake as the earlier span comparison).
    rank_of = {int(reps[i]): int(k)
               for k, i in enumerate(sorted(range(len(reps)),
                                            key=lambda j: (float(scores[j]),
                                                           int(reps[j]))))}
    poly_ranks = [rank_of.get(reduce_mod_exact(int(m))) for m in las_poly]
    exh_ranks = [rank_of.get(int(m)) for m in exh["las"]]

    out = {
        "oo_json": str(oo_path),
        "scope": args.scope,
        "n_bits": n_bits,
        "r": r,
        "M": M,
        "pool_size": len(reps),
        "pool_size_expected": 2 ** target - 1,
        "E_FCI": problem["e_fci"],
        "fci_gap_E1_minus_E0": problem.get("gap"),
        "reference_route": args.reference,
        "reference_root": args.reference_root,
        "exhaustive": {
            "las_masks": exh["las"],
            "las_costs": exh["las_costs"],
            "auxiliary_masks": exh["auxiliary"],
            "ranked_basis": exh["accepted"],
            "ranked_basis_costs": exh["accepted_costs"],
            "first_rejected_competitor": exh["first_rejected_competitor"],
            "span_rref_with_exact": span_exh,
            "las_classes_mod_exact": classes_exh,
        },
        "polynomial": {
            "las_masks": las_poly,
            "selected_costs": poly_costs,
            "span_rref_with_exact": span_poly,
            "las_classes_mod_exact": classes_poly,
            "ranks_in_exhaustive_order": poly_ranks,
            "exhaustive_own_ranks": exh_ranks,
            "cost_polynomial_at_final_U": [float(c) for c in poly_final],
            "sum_cost_polynomial_at_final_U": float(sum(poly_final)),
            "sum_cost_exhaustive": float(sum(exh["las_costs"])),
            "cost_ratio_exhaustive_over_polynomial": (
                float(sum(exh["las_costs"]) / sum(poly_final))
                if sum(poly_final) > 0 else None),
        },
        "scoring_gate": {
            "passed": bool(gate_ok),
            "max_relative_error": float(gate_err),
            "frame_reproducing_recorded_costs": gate_label,
            "recorded_costs_are_pre_oo": bool(use_T == "identity"),
            "cost_function": cost_function,
            "sum_selected_costs": (sum(poly_recorded) if poly_recorded else None),
            "cost_before": oo.get("cost_before"),
            "cost_after": oo.get("cost_after"),
            "polynomial_nc_recorded": poly_recorded,
            "polynomial_nc_rescored_here": [float(c) for c in poly_here],
            "note": ("The exhaustive pool contains every polynomial-reachable "
                     "mask, so exhaustive costs cannot exceed polynomial ones. "
                     "If the gate fails, any disagreement below is an artifact "
                     "of this script."),
        },
        "agreement": {
            "las_span_identical": bool(agree),
            "las_classes_identical_mod_exact": bool(same_classes),
            "verdict": (
                "SCORING GATE FAILED -- comparison not meaningful"
                if not gate_ok else
                "polynomial pool validated at this U (span(E u G) matches; "
                "differing generator sets spanning the same quotient subspace "
                "are equivalent by Sec.2(iii))" if agree else
                "CANDIDATE-SPACE ERROR (Sec.4.3): the polynomial search "
                "converged to a pool that the exhaustive ranking does not select"),
        },
        "selection_trace": exh["trace"],
        "not_covered": (
            "Sec.4.2 self-consistent exhaustive fixed point: orbitals were NOT "
            "re-optimised against the exhaustive pool. This is the Sec.4.3 "
            "comparison at the polynomial search's converged U."),
    }
    path = Path(args.output or (oo_path.parent /
                                f"exhaustive_{args.scope}.json"))
    path.write_text(json.dumps(out, indent=2))
    if M >= n_bits - r:
        print(f"[exh] NOTE: M={M} equals N-r={n_bits - r}. Any M independent rows "
              "span the whole quotient, so the span test below cannot fail and "
              "carries no information. The cost comparison is the informative "
              "quantity at this budget.")
    _rp, _re = float(sum(poly_final)), float(sum(exh["las_costs"]))
    print(f"[exh] total {cost_function}: polynomial={_rp:.10g}  exhaustive={_re:.10g}  "
          f"ratio={_re / _rp if _rp > 0 else float('nan'):.6f}")
    print(f"[exh] span(E u G) identical      : {agree}")
    print(f"[exh] LAS classes identical mod E: {same_classes}  "
          f"(informational; Eq.(7) tests the span, not the generator list)")
    if not agree:
        print(f"[exh] *** {out['agreement']['verdict']}")
        print(f"[exh] exhaustive LAS costs : {exh['las_costs']}")
        print(f"[exh] polynomial LAS costs : {poly_costs}")
    print(f"[exh] wrote {path}")


if __name__ == "__main__":
    main()
