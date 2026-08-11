#!/usr/bin/env bash
# AUDIT 7 — build the Sec.10 "Minimum reproducibility record" from the existing
# artifacts and report, item by item, what is still missing.
#
# Sec.10 of Iterative_NC_Ranked_LAS_Search_Procedure_poly+exhaustive.pdf:
#   1 qubit ordering, rotation convention, allowed group, r exact rows + eigenvalues
#   2 N, r, M, candidate mode, pool size, initial frame, initial LAS pool,
#     completion rule, deterministic NC tie-break rule
#   3 per iteration: full candidate list, NC scores, sorted order, rejected rows
#     + reasons, all N-r accepted quotient rows, first M LAS vs auxiliary
#   4 every intermediate and final F2 span
#   5 orbital matrix, objective, optimizer convergence, state route, solver
#     params, energy, residual/variance, exact-symmetry leakage
#   6 route B only (DMRG): sector set, bond dims, sweeps, discarded weights
#   7 exact-control decoupled energy + the full K-dependent energy sequence
#
# Writes results/<...>/repro_record.json per point and a coverage table.
# Cheap: reads JSON only. Run:  bash scripts/audit/audit_07_repro_record.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
mkdir -p tables/audit
export U="${U:-irrep}"

python3 - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo

U = os.environ.get("U", "irrep")
cov = {i: {"have": 0, "miss": 0} for i in range(1, 8)}
missing_detail = {}
n = 0

for md in sorted(Path("results").glob(f"*_endpoint_grid/bond_*/U_{U}/*/*/metrics.json")):
    d = load_oo(md)
    oo_path = md.parent / "oo.json"
    oo = load_oo(oo_path) if oo_path.is_file() else {}
    if "reference_weight_ok" not in d and "exact_sector_source" not in d:
        continue
    n += 1
    mol = md.parts[1].split("_")[0]
    method = md.parts[-3]
    rec, gaps = {}, []

    # ---- item 1 -----------------------------------------------------------
    rec["item1_frame_and_exacts"] = {
        "qubit_ordering": "interleaved Jordan-Wigner (2p = alpha_p, 2p+1 = beta_p)",
        "orbital_rotation_convention": "params_to_U, givens pairs; see src/orbital_rotation.py",
        "allowed_rotation_group": oo.get("orbital_rotation") or U,
        "exact_masks": d.get("exact_masks"),
        "n_exact": d.get("n_exact"),
        "exact_sector_eigenvalues": d.get("exact_sector"),
    }
    if not d.get("exact_masks") or d.get("exact_sector") is None:
        gaps.append(1)

    # ---- item 2 -----------------------------------------------------------
    norb = 7 if mol == "h2o" else 10
    rec["item2_problem_spec"] = {
        "N_spatial_register": norb,
        "N_qubits": 2 * norb,
        "r": d.get("n_exact"),
        "M": oo.get("M") or oo.get("n_sym") or d.get("n_las"),
        "candidate_mode": oo.get("candidates", "senquart (polynomial current-frame)"),
        "pool_size": norb * (norb + 1) // 2,
        "initial_full_frame": "identity F(0) = I_N (Gf2ParityFrame.identity)",
        "initial_LAS_pool": oo.get("initial_pool"),
        "completion_rule": "continued ranked scan to N-r (no separate completion)",
        "nc_tie_break": "ascending (cost, frame_indices) -- iterative_pool._frame_candidates",
    }
    if rec["item2_problem_spec"]["M"] is None:
        gaps.append(2)

    # ---- item 3 -----------------------------------------------------------
    rounds = oo.get("rounds") or []
    # iterative: per-macroiteration trace lives in rounds[i]["selection_trace"]
    # mixed/greedy: single top-level trace + (new) full ranked_candidates
    per_round = [len(r.get("selection_trace") or []) for r in rounds
                 if isinstance(r, dict)]
    top_trace = oo.get("selection_trace") or []
    ranked = oo.get("ranked_candidates") or []
    rec["item3_per_iteration_scan"] = {
        "n_macroiterations": len(rounds),
        "trace_entries_per_round": per_round,
        "top_level_trace_entries": len(top_trace),
        "ranked_candidate_library": len(ranked),
        "pool_size_expected": rec["item2_problem_spec"]["pool_size"],
        "ranked_basis_masks": oo.get("ranked_basis_masks"),
        "las_masks": oo.get("las_masks"),
        "auxiliary_masks": oo.get("auxiliary_masks"),
    }
    have_full_scan = bool(per_round and all(per_round)) or (
        len(ranked) >= rec["item2_problem_spec"]["pool_size"])
    if not have_full_scan:
        gaps.append(3)

    # ---- item 4 -----------------------------------------------------------
    spans = [r.get("span_key") for r in rounds if isinstance(r, dict)]
    span_trace = oo.get("span_trace") or []
    rec["item4_spans"] = {
        "intermediate_spans_recorded": sum(1 for s in spans if s),
        "span_B_keys": sum(1 for r in rounds
                           if isinstance(r, dict) and r.get("span_B_key")),
        "greedy_span_trace": len(span_trace),
        "final_span_rref": (spans[-1] if spans else
                            (span_trace[-1]["span_rref"] if span_trace else None)),
    }
    if not any(spans) and not span_trace:
        gaps.append(4)

    # ---- item 5 -----------------------------------------------------------
    # The path recorded in oo.json is only present if that oo.json was written
    # by the post-fix driver. The artifact itself is a sibling file, so check
    # disk too -- otherwise a tree synced without oo.json reports a false gap.
    oo_trace = oo.get("oo_trace_json")
    oo_trace_file = md.parent / "oo_trace.json"
    if oo_trace is None and oo_trace_file.is_file():
        oo_trace = str(oo_trace_file)
    n_oo_steps = 0
    if oo_trace_file.is_file():
        try:
            n_oo_steps = len(json.loads(oo_trace_file.read_text()))
        except Exception:  # noqa: BLE001
            n_oo_steps = 0
    rec["item5_orbital_and_state"] = {
        # the optimised rotation parameters live under "rotation"
        "orbital_parameters": oo.get("rotation"),
        "objective_before": oo.get("cost_before"),
        "objective_after": oo.get("cost_after"),
        "optimizer_message": oo.get("message"),
        "optimizer_converged": oo.get("converged"),
        "optimizer_nit": oo.get("nit"),
        "optimizer_nfev": oo.get("nfev"),
        "oo_step_tol": oo.get("oo_step_tol"),
        "oo_stop_tol": oo.get("oo_stop_tol"),
        "optimizer_maxiter": oo.get("optimizer_maxiter"),
        "oo_trace_json": oo_trace,
        "oo_trace_steps": n_oo_steps,
        "state_route": "A (recompute rotated FCI; iterative_reference=fci_rotate)",
        "states_per_sector": d.get("states_per_sector"),
        "E_FCI": d.get("E_FCI"),
        "exact_symmetry_leakage": (None if d.get("reference_weight_sum") is None
                                   else 1.0 - float(d["reference_weight_sum"])),
    }
    if rec["item5_orbital_and_state"]["orbital_parameters"] is None:
        gaps.append(5)
    elif oo_trace is None:
        # nit/nfev/message are present; the per-iteration objective trajectory
        # is not. optimize_symmetries.py --oo_trace_json writes it.
        gaps.append(5)

    # ---- item 6 (route B only; route A used here) --------------------------
    rec["item6_route_B"] = "n/a -- route A used (no DMRG in the search)"

    # ---- item 7 -----------------------------------------------------------
    curve = d.get("coupled_curve") or {}
    rec["item7_K_sequence"] = {
        "E_decoupled": d.get("E_decoupled"),
        "K": d.get("K"),
        "converged": d.get("converged"),
        "K_energies": curve.get("energies"),
        "K_order": curve.get("order"),
        "epsilon_E": 1.6e-3,
    }
    if not curve.get("energies"):
        gaps.append(7)

    for i in range(1, 8):
        if i == 6:
            continue
        (cov[i]["miss" if i in gaps else "have"]) and None
        cov[i]["miss" if i in gaps else "have"] += 1
    if gaps:
        missing_detail.setdefault(tuple(sorted(set(gaps))), []).append(
            f"{mol}/{md.parts[2]}/{method}/{md.parts[-2]}")

    (md.parent / "repro_record.json").write_text(json.dumps(rec, indent=2, default=str))

print(f"\nSec.10 coverage over {n} post-fix points (U_{U})")
print(f"{'item':>5} {'present':>9} {'missing':>9}   what")
LABEL = {
    1: "frame + exact rows/eigenvalues",
    2: "N, r, M, mode, pool, tie-break",
    3: "per-iteration candidate scan + reasons",
    4: "intermediate/final F2 spans",
    5: "orbital matrix, optimizer data, leakage",
    6: "route B / DMRG (n/a here)",
    7: "decoupled energy + K-energy sequence",
}
for i in range(1, 8):
    if i == 6:
        print(f"{i:>5} {'n/a':>9} {'n/a':>9}   {LABEL[i]}")
        continue
    print(f"{i:>5} {cov[i]['have']:>9} {cov[i]['miss']:>9}   {LABEL[i]}")

if missing_detail:
    print("\ngap groups:")
    for k, v in sorted(missing_detail.items()):
        print(f"  items {list(k)}: {len(v)} point(s), e.g. {v[0]}")

print("""
Not covered by any existing artifact -- these need new work, not new parsing:

  A. Sec.4 exhaustive quotient-pool check. Rank ALL 2^(N-r)-1 quotient rows
     (1023 for H2O, 32767 for N2) after every orbital optimisation and compare
     the fixed point against the polynomial search (Sec.4.3: LAS span, NC of
     selected vs first rejected competitor, decoupled energy, K, macroiteration
     count). NOT IMPLEMENTED -- the submitters say so explicitly. This is the
     single largest hole: without it the polynomial pool is an unvalidated
     restricted-neighbourhood heuristic (Sec.7.1).

  B. (RESOLVED in code) greedy_selection.py now emits ranked_candidates (the
     full scored library) and span_trace. Points computed before that change
     still lack it and must be recomputed to close items 3 and 4.

  C. Sec.7.3(4) boundary reranking: recompute the accepted-candidate boundary
     and near-ties with a more accurate state, and confirm the ranking is
     resolved rather than within solver noise.

  D. Sec.3.2 / 7.1 initialisation robustness: repeat from several initial binary
     frames and orbital starts. One start per point at present.
""")
PY
