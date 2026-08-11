#!/usr/bin/env python
"""Sec.4.2 self-consistent exhaustive-pool fixed point (route A).

Implements steps (1)-(6) of Sec.4.2 of
``plans/Iterative_NC_Ranked_LAS_Search_Procedure_poly+exhaustive.pdf``:

  (1) initial orbitals U(0); build the route-A reference state;
  (2) score EVERY member of C_full = {gbar != 0} (2^(N-r)-1 classes), sort
      ascending, greedily accept rows that raise the quotient rank; first M
      accepted are G(t), the scan continues to N-r for the Clifford frame;
  (3) with G(t) fixed, re-optimise the orbitals against
          C_A(U; G) = sum_i c_{Psi_A(U),U}(g_i)                       Eq.(21)
      recomputing the approximate state at every trial U (route A, Eq.(20));
  (4) at the optimised U(t), rescore ALL candidates and repeat the full greedy
      scan -- not just the neighbourhood of the previous top M;
  (5) if the new first-M rows do not span the same LAS subspace, replace and
      return to (3); if the span is unchanged and the orbital objective and
      gradient satisfy their tolerances for 2 consecutive macroiterations,
      declare a self-consistent exhaustive fixed point;
  (6) detect cycles of LAS spans, retain the best-so-far under the declared
      final criterion.

WHAT THIS IS NOT (Sec.4.4)
--------------------------
Not a global optimum. It removes the candidate-omission approximation only.
Local orbital minima, basis dependence of the generator-sum objective,
approximate-state misranking and cycles all remain. Different orbital starts
can converge to different fixed points -- use --start to test that.

BUDGET CAVEAT
-------------
When M == N - r the LAS span is the entire quotient for ANY independent pool,
so the partition (and hence K, sectors, D_max) cannot change and step (5)'s
span test is trivially satisfied at the first macroiteration. The loop then
only minimises a basis-dependent generator sum. Run with --M below N-r for a
test with physical content.

Output
------
  exhaustive_selfconsistent[_M<k>].json  full trajectory + fixed point
  oo_selfconsistent[_M<k>].json          metrics.py-compatible OO record, so K
                                         at the fixed point can be obtained with
                                         a normal metrics run

Usage
-----
    python scripts/exhaustive_selfconsistent.py \
        --oo results/n2_endpoint_grid/bond_1p8000/U_irrep/iterative/NC/oo.json
    python scripts/exhaustive_selfconsistent.py --oo ... --M 6 --start identity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "audit"))
sys.path.insert(0, str(REPO / "scripts"))

from _qs_json import load_oo                                        # noqa: E402
from src.gf2_utils import gf2_int_rref, gf2_int_try_add_to_span     # noqa: E402
from exhaustive_quotient_check import (                             # noqa: E402
    coset_representatives, in_span, nc_scores, parity_of, sign_tables,
    span_signature, _z_signs,
)


# --------------------------------------------------------------------------
def build_problem_from_params(ctx, x, *, root=0):
    """Route A, Eq.(20): rebuild H(U) and RE-SOLVE the reference at this U."""
    from pyscf import ao2mo, fci

    from src.orbital_rotation import params_to_U

    norb, nelec = ctx["norb"], ctx["nelec"]
    U = params_to_U(np.asarray(x, dtype=float), norb, ctx["pairs"]) \
        if len(x) else np.eye(norb)
    # convention fixed empirically: ffsim's rotated(U) == mo @ U.T
    mo_rot = ctx["mo"][:, :norb] @ U.T
    h1e = mo_rot.T @ ctx["hcore"] @ mo_rot
    eri = ao2mo.kernel(ctx["mol"], mo_rot)
    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    e_list, ci_list = fci.direct_spin1.kernel(
        h1e, eri, norb, nelec, ecore=ctx["ecore"], nroots=max(2, root + 1))
    roots = ci_list if isinstance(ci_list, (list, tuple)) else [ci_list]
    ci = np.asarray(roots[min(root, len(roots) - 1)])
    ci = ci / np.linalg.norm(ci)
    gap = float(np.atleast_1d(e_list)[1] - np.atleast_1d(e_list)[0])
    return dict(norb=norb, nelec=nelec, h2e=h2e, ci=ci,
                e_fci=float(np.atleast_1d(e_list)[0]), gap=gap)


def objective(ctx, x, generators, cost_function, root=0):
    """Eq.(21): C_A(U; G) = sum_i c(g_i) with the state recomputed at U."""
    prob = build_problem_from_params(ctx, x, root=root)
    vals = nc_scores(prob, generators, ctx["scope"], cost_function)
    return float(np.sum(vals)), prob, vals


def ranked_scan(reps, scores, exact_rref, n_bits, M, target):
    """Sec.4.2 step (2): full greedy independence scan on the sorted pool."""
    order = sorted(range(len(reps)),
                   key=lambda i: (float(scores[i]), int(reps[i])))
    rref = list(exact_rref)
    accepted, acc_cost, trace = [], [], []
    boundary = None
    for pos in order:
        g = int(reps[pos])
        new = gf2_int_try_add_to_span(g, rref, n_bits)
        if new is None:
            trace.append({"event": "reject", "mask": g,
                          "reason": ("exact_parity" if in_span(g, exact_rref)
                                     else "gf2_dependent_or_exact_dressed"),
                          "cost": float(scores[pos])})
            continue
        if len(accepted) == M and boundary is None:
            boundary = {"mask": g, "cost": float(scores[pos])}
        rref = new
        accepted.append(g)
        acc_cost.append(float(scores[pos]))
        trace.append({"event": "accept_las" if len(accepted) <= M
                      else "accept_auxiliary",
                      "mask": g, "cost": float(scores[pos])})
        if len(accepted) >= target:
            break
    return dict(ranked_basis=accepted, ranked_costs=acc_cost,
                las=accepted[:M], las_costs=acc_cost[:M],
                auxiliary=accepted[M:],
                first_rejected_competitor=boundary, trace=trace)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oo", required=True,
                    help="polynomial run to take the problem definition from")
    ap.add_argument("--scope", default="spatial", choices=("spatial", "qubit"))
    ap.add_argument("--M", type=int, default=None,
                    help="LAS budget (default: the run's M). M < N-r is the "
                         "only regime where the span test has content")
    ap.add_argument("--start", default="run",
                    choices=("run", "identity", "random"),
                    help="initial orbitals: the run's converged U, the "
                         "canonical MOs, or a random point (Sec.3.2/7.1 "
                         "initialisation robustness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-macro", type=int, default=20)
    ap.add_argument("--obj-tol", type=float, default=1e-8,
                    help="Sec.3.3(3): orbital objective change tolerance")
    ap.add_argument("--grad-tol", type=float, default=1e-6,
                    help="Sec.3.3(4): projected gradient tolerance")
    ap.add_argument("--stable", type=int, default=2,
                    help="consecutive macroiterations required (Sec.3.3)")
    ap.add_argument("--maxiter", type=int, default=200,
                    help="L-BFGS-B iterations per orbital optimisation")
    ap.add_argument("--reference-root", type=int, default=0)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    from pyscf import ao2mo, scf
    from pyscf.scf import chkfile
    from scipy.optimize import minimize

    from src.orbital_rotation import resolve_orbital_rotation

    oo_path = Path(args.oo)
    oo = load_oo(oo_path)
    molpath = oo["molpath"]
    norb = 7 if "H2O" in str(molpath) else 10
    exact_spatial = [int(m) for m in (oo.get("exact_masks") or [])]
    cost_function = str(oo.get("cost_function") or "NC")
    packing = str(oo.get("orbital_rotation") or "irrep")

    if args.scope == "spatial":
        n_bits, exact = norb, exact_spatial
    else:
        from src.sto3g_exact_symmetries import (
            expand_spatial_mask_to_interleaved_qubits,
            interleaved_spin_number_masks)
        n_bits = 2 * norb
        pa, pb = interleaved_spin_number_masks(norb)
        exact = [pa, pb] + [expand_spatial_mask_to_interleaved_qubits(m, norb)
                            for m in exact_spatial]

    exact_rref, _ = gf2_int_rref([m for m in exact if m], n_bits)
    r = len(exact_rref)
    target = n_bits - r
    M = int(args.M) if args.M is not None else int(
        oo.get("M") or oo.get("n_sym") or target)
    reps = coset_representatives(n_bits, exact_rref)

    mol = chkfile.load_mol(str(molpath))
    ctx = dict(
        mol=mol, mo=np.asarray(chkfile.load(str(molpath), "scf")["mo_coeff"]),
        hcore=scf.hf.get_hcore(mol), ecore=float(mol.energy_nuc()),
        norb=norb, nelec=(mol.nelectron // 2, mol.nelectron // 2),
        pairs=resolve_orbital_rotation(packing, str(molpath), norb)[0],
        scope=args.scope)

    x_run = np.asarray(oo.get("rotation") or [], dtype=float)
    n_par = len(x_run)
    if args.start == "run":
        x = x_run.copy()
    elif args.start == "identity":
        x = np.zeros(n_par)
    else:
        x = np.random.default_rng(args.seed).normal(0.0, 0.25, n_par)

    print(f"[sc] {oo_path}")
    print(f"[sc] norb={norb} r={r} N-r={target} M={M} pool={len(reps)} "
          f"cost={cost_function} packing={packing} start={args.start}")
    if M >= target:
        print(f"[sc] note: M={M} = N-r={target}. Every independent pool spans "
              "the whole quotient, so the partition is identical for all pools "
              "and self-consistency is tested on the ordered generator set and "
              "the objective, not on the span.")

    history, seen_spans, best = [], {}, None
    prev_span, stable_count = None, 0
    stop_reason = "max_macro"
    t_start = time.time()

    for macro in range(args.max_macro):
        # ---- steps (2)/(4): rescore the FULL pool at the current orbitals ---
        prob = build_problem_from_params(ctx, x, root=args.reference_root)
        scores = nc_scores(prob, reps, args.scope, cost_function)
        scan = ranked_scan(reps, scores, exact_rref, n_bits, M, target)
        span = tuple(span_signature([*exact, *scan["las"]], n_bits))
        # At M = N - r the span is the whole quotient for ANY independent pool,
        # so a span-only stopping test fires at macroiteration 0 and the loop
        # never re-optimises. Track the ordered generator multiset instead; it
        # still varies when the span does not.
        pool_key = tuple(sorted(int(m) for m in scan["las"]))
        state_key = (span, pool_key)

        # ---- step (6): cycle detection --------------------------------------
        cycled = state_key in seen_spans
        seen_spans.setdefault(state_key, macro)

        # ---- step (3): re-optimise the orbitals against this pool -----------
        gens = list(scan["las"])
        f0, _, _ = objective(ctx, x, gens, cost_function, args.reference_root)
        res = minimize(
            lambda xx: objective(ctx, xx, gens, cost_function,
                                 args.reference_root)[0],
            x, method="L-BFGS-B",
            options=dict(maxiter=args.maxiter, ftol=args.obj_tol,
                         gtol=args.grad_tol))
        x_new, f1 = np.asarray(res.x), float(res.fun)
        gnorm = float(np.max(np.abs(res.jac))) if res.jac is not None else float("nan")

        rec = dict(macro=macro, span=list(span), objective_before=f0,
                   objective_after=f1, delta=f0 - f1, grad_inf_norm=gnorm,
                   nit=int(res.nit), nfev=int(res.nfev),
                   las_masks=[int(m) for m in scan["las"]],
                   las_costs=scan["las_costs"],
                   ranked_basis=[int(m) for m in scan["ranked_basis"]],
                   first_rejected_competitor=scan["first_rejected_competitor"],
                   e_fci=prob["e_fci"], fci_gap=prob["gap"],
                   pool=list(pool_key),
                   span_repeat_of=seen_spans.get(state_key) if cycled else None,
                   x=[float(v) for v in x_new])
        history.append(rec)
        print(f"[sc] macro {macro:2d}  obj {f0:.10g} -> {f1:.10g}  "
              f"|g|={gnorm:.2e}  span_rank={len(span)}  "
              f"{'CYCLE' if cycled else ''}")

        if best is None or f1 < best["objective_after"]:
            best = rec

        # ---- step (5): self-consistency test -------------------------------
        # Self-consistency requires the LAS pool itself to stop moving, not
        # merely its span (identical for every pool when M = N - r).
        pool_unchanged = (prev_span is not None and state_key == prev_span)
        tol_ok = (abs(f0 - f1) < args.obj_tol) and (
            not np.isfinite(gnorm) or gnorm < args.grad_tol)
        stable_count = stable_count + 1 if (pool_unchanged and tol_ok) else 0
        prev_span, x = state_key, x_new

        if stable_count >= args.stable:
            stop_reason = "self_consistent_fixed_point"
            break
        if cycled and macro > 0:
            stop_reason = "span_cycle_detected"
            break

    # ---- final pool at the converged orbitals ------------------------------
    prob = build_problem_from_params(ctx, x, root=args.reference_root)
    scores = nc_scores(prob, reps, args.scope, cost_function)
    final = ranked_scan(reps, scores, exact_rref, n_bits, M, target)
    final_span = span_signature([*exact, *final["las"]], n_bits)

    poly_las = [int(m) for m in (oo.get("las_masks")
                                 or oo.get("accumulated_masks") or [])]
    if not poly_las:
        from src.exact_parity import masks_from_parity_matrix
        for cand in (oo.get("parity_output"), oo_path.parent / "oo_parity.txt"):
            if cand and Path(cand).is_file():
                poly_las = [int(m) for m in masks_from_parity_matrix(
                    np.atleast_2d(np.loadtxt(str(cand), dtype=int)))]
                break
    poly_at_fixed = nc_scores(prob, poly_las, args.scope, cost_function) \
        if poly_las else np.array([])

    tag = f"_M{M}" if args.M is not None else ""
    out = dict(
        oo_json=str(oo_path), scope=args.scope, n_bits=n_bits, r=r, M=M,
        target=target, pool_size=len(reps), cost_function=cost_function,
        packing=packing, start=args.start, seed=args.seed,
        M_equals_N_minus_r=bool(M >= target),
        stop_reason=stop_reason, n_macroiterations=len(history),
        elapsed_s=time.time() - t_start,
        fixed_point=dict(
            x=[float(v) for v in x], las_masks=[int(m) for m in final["las"]],
            las_costs=final["las_costs"],
            ranked_basis=[int(m) for m in final["ranked_basis"]],
            objective=float(sum(final["las_costs"])),
            first_rejected_competitor=final["first_rejected_competitor"],
            span_rref_with_exact=final_span,
            e_fci=prob["e_fci"], fci_gap=prob["gap"]),
        polynomial=dict(
            las_masks=poly_las,
            cost_at_fixed_point_orbitals=[float(c) for c in poly_at_fixed],
            objective_at_fixed_point_orbitals=float(sum(poly_at_fixed))
            if len(poly_at_fixed) else None,
            span_rref_with_exact=span_signature([*exact, *poly_las], n_bits)
            if poly_las else None),
        history=history,
        caveats=[
            "Sec.4.4: not a global optimum; removes candidate omission only.",
            "Local orbital minima remain -- compare --start run/identity/random.",
            "Eq.(21) is a generator-SUM and is basis dependent (Sec.5), so the "
            "objective value is not an invariant of the LAS span.",
        ])
    path = Path(args.output or
                oo_path.parent / f"exhaustive_selfconsistent{tag}.json")
    path.write_text(json.dumps(out, indent=2))

    # metrics.py-compatible record so K at the fixed point is one command away
    parity_txt = oo_path.parent / f"sc_parity{tag}.txt"
    rows = np.zeros((len(final["las"]), norb), dtype=int)
    for i, m in enumerate(final["las"]):
        for b in range(norb):
            if (m >> b) & 1:
                rows[i, b] = 1
    np.savetxt(parity_txt, rows, fmt="%d")
    oo_sc = dict(oo)
    oo_sc.update(dict(
        rotation=[float(v) for v in x], las_masks=[int(m) for m in final["las"]],
        accumulated_masks=[int(m) for m in final["las"]],
        parity=str(parity_txt), parity_output=str(parity_txt),
        selection="exhaustive_selfconsistent",
        selection_rule="exhaustive_quotient_fixed_point",
        candidates=f"exhaustive_{args.scope}", M=M, n_sym=M,
        selected_costs=final["las_costs"],
        cost_before=history[0]["objective_before"] if history else None,
        cost_after=float(sum(final["las_costs"]))))
    oo_out = oo_path.parent / f"oo_selfconsistent{tag}.json"
    oo_out.write_text(json.dumps(oo_sc, indent=2, default=str))

    print(f"\n[sc] stop: {stop_reason} after {len(history)} macroiteration(s)")
    print(f"[sc] fixed-point objective  : {sum(final['las_costs']):.10g}")
    if len(poly_at_fixed):
        print(f"[sc] polynomial pool at same U: {sum(poly_at_fixed):.10g}")
    print(f"[sc] span identical to polynomial: "
          f"{final_span == (span_signature([*exact, *poly_las], n_bits) if poly_las else None)}")
    print(f"[sc] wrote {path}")
    print(f"[sc] wrote {oo_out}  (run metrics.py on it for K at the fixed point)")


if __name__ == "__main__":
    main()
