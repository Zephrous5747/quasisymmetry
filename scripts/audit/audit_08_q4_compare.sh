#!/usr/bin/env bash
# AUDIT 8 — the Q4 comparison. Reads results/{mol}_q4_grid/ only.
#
# Two stages, in this order:
#   (1) PRECONDITIONS. Every point must have W = 1, n_exact = r, Eq.(K)
#       satisfied, and M at the recorded budget. Any point failing these is
#       excluded from stage 2 and listed. A comparison built on points that
#       failed a precondition is worthless.
#   (2) COMPARISON. Paired K per geometry across arms, per cost function.
#       Reported as observations; with a single orbital start there is no
#       scatter estimate, so no significance is claimed.
#
# Cheap: JSON only. Run:  bash scripts/audit/audit_08_q4_compare.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
mkdir -p tables/audit
export U="${U:-irrep}"

python3 - <<'PY'
import glob, json, os, sys, statistics
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo

U = os.environ.get("U", "irrep")
TOL = float(os.environ.get("REFERENCE_WEIGHT_TOL", "1e-4"))
# Budget is campaign-specific: M = n - r_sp (4/6) for the full-budget run,
# M = 3/5 for the reduced-budget run. r (qubit exact rank) is a property of the
# molecule and does NOT change with the budget.
BUDGET = {
    "h2o": int(os.environ.get("BUDGET_H2O", "4")),
    "n2": int(os.environ.get("BUDGET_N2", "6")),
}
R_QUBIT = {"h2o": 4, "n2": 5}
# dim(Q) = n - r_sp. M == dim(Q) forces every arm to span the whole quotient,
# so the partition is identical by construction and the arms cannot differ
# through it. Flagged so the comparison is never read as more than it is.
QUOTIENT_DIM = {"h2o": 4, "n2": 6}

rows, bad = {}, []
SUFFIX = os.environ.get("RESULTS_SUFFIX", "q4_grid")
print(f"reading results/*_{SUFFIX}/ (U_{U})")
for f in sorted(glob.glob(f"results/*_{SUFFIX}/bond_*/U_{U}/*/*/metrics.json")):
    p = Path(f).parts
    mol = p[1].split("_")[0]
    bond = round(float(p[2].replace("bond_", "").replace("p", ".")), 4)
    method, cost = p[-3], p[-2]
    d = load_oo(f)
    oo = load_oo(Path(f).parent / "oo.json") if (Path(f).parent / "oo.json").is_file() else {}
    W = d.get("reference_weight_sum")
    W = float(W) if W is not None else 0.0
    M = d.get("n_las") or oo.get("M") or oo.get("n_sym")
    fails = []
    if W <= 1.0 - TOL:                       fails.append(f"W={W:.6f}")
    if int(d.get("n_exact", -1)) != R_QUBIT[mol]: fails.append(f"n_exact={d.get('n_exact')}")
    if not d.get("converged"):               fails.append("K not converged")
    if M is not None and int(M) != BUDGET[mol]:   fails.append(f"M={M}")
    # Partition quantities must be K-INDEPENDENT to serve as cross-arm
    # invariants. relevant_sectors_* are summed over the sectors the coupled
    # curve used, so they scale with K and DIFFER whenever K differs -- which
    # is the comparison's dependent variable, not a defect. Use the
    # exact-filtered partition instead.
    rec = dict(mol=mol, bond=bond, method=method, cost=cost,
               K=d.get("K"),
               dim=d.get("exact_sector_total_dim"),
               sectors=d.get("n_sectors_retained"),
               Dmax=d.get("D_max"),
               used_dim=d.get("relevant_sectors_total_dim"),
               used_sectors=d.get("relevant_sectors_count"),
               W=W, M=M, dE=d.get("dE"), fails=fails)
    rows[(mol, bond, method, cost)] = rec
    if fails:
        bad.append(rec)

print(f"points found: {len(rows)}")
print(f"failing a precondition: {len(bad)}")
for mol in sorted({k[0] for k in rows}):
    m, q = BUDGET[mol], QUOTIENT_DIM[mol]
    if m == q:
        print(f"  [{mol}] M={m} = dim(Q)={q}: span is the WHOLE quotient for every "
              "arm, so the partition is identical by construction and arms can "
              "differ only through the optimised orbitals.")
    else:
        print(f"  [{mol}] M={m} < dim(Q)={q}: arms select different subspaces, "
              "so the partition itself can differ -- comparison has resolving power.")
for r in bad[:12]:
    print(f"   {r['mol']} R={r['bond']:.4f} {r['method']}/{r['cost']}: "
          f"{', '.join(r['fails'])}")
if bad:
    print("   -> excluded from the comparison below")

good = {k: v for k, v in rows.items() if not v["fails"]}
methods = sorted({k[2] for k in good})
if not good:
    raise SystemExit("\nno points passed the preconditions; nothing to compare")

# ---- invariants that must hold across arms at M = n - r_sp ---------------
print("\n=== partition invariants (must be identical across arms) ===")
for mol in sorted({k[0] for k in good}):
    for cost in sorted({k[3] for k in good if k[0] == mol}):
        bonds = sorted({k[1] for k in good if k[0] == mol and k[3] == cost})
        bad_inv = []
        for b in bonds:
            vals = {m: (good[(mol, b, m, cost)]["dim"],
                        good[(mol, b, m, cost)]["sectors"],
                        good[(mol, b, m, cost)]["Dmax"])
                    for m in methods if (mol, b, m, cost) in good}
            if len(set(vals.values())) > 1:
                bad_inv.append((b, vals))
        status = "OK (identical)" if not bad_inv else f"DIFFER at {len(bad_inv)} geometries"
        print(f"  {mol:4s} {cost:9s} exact_dim/retained_sectors/D_max across arms: {status}")
        for b, v in bad_inv[:3]:
            print(f"      R={b}: {v}")
        if any(good[k]["dim"] is None for k in good if k[0] == mol and k[3] == cost):
            print("      (note: some points predate exact_sector_total_dim; re-metricise)")

# ---- the comparison -------------------------------------------------------
print("\n=== K by arm ===")
for mol in sorted({k[0] for k in good}):
    for cost in sorted({k[3] for k in good if k[0] == mol}):
        bonds = sorted({k[1] for k in good if k[0] == mol and k[3] == cost})
        hdr = "  ".join(f"{m[:12]:>12}" for m in methods)
        print(f"\n  {mol.upper()}  {cost}")
        print(f"  {'R':>7}  {hdr}")
        series = defaultdict(list)
        for b in bonds:
            cells = []
            for m in methods:
                r = good.get((mol, b, m, cost))
                cells.append(f"{r['K']:>12}" if r else f"{'-':>12}")
                if r:
                    series[m].append((b, r["K"]))
            print(f"  {b:7.4f}  " + "  ".join(cells))
        common = [b for b in bonds
                  if all((mol, b, m, cost) in good for m in methods)]
        if len(common) >= 2 and len(methods) >= 2:
            print(f"  {'-' * (9 + 14 * len(methods))}")
            base = methods[0]
            for m in methods[1:]:
                d = [good[(mol, b, m, cost)]["K"] - good[(mol, b, base, cost)]["K"]
                     for b in common]
                wins = sum(1 for x in d if x < 0)
                ties = sum(1 for x in d if x == 0)
                loss = sum(1 for x in d if x > 0)
                print(f"  {m} vs {base} over {len(common)} geometries: "
                      f"lower K {wins}, equal {ties}, higher K {loss}; "
                      f"mean dK = {statistics.mean(d):+.2f}, "
                      f"range [{min(d):+d}, {max(d):+d}]")

print("""
Interpretation
--------------
At M = n - r_sp the LAS span is the whole quotient for every arm, so the
partition -- dim, sectors, D_max -- is identical by construction; the invariant
check above verifies that this holds in the data. Any difference in K therefore
arises through the optimised orbitals, i.e. through the objective landscape that
each pool induces, not through the partition.

Only one orbital start was run, so no scatter estimate exists. Differences in K
are observations, not statistically significant results. A difference of one or
two units should be read as "no detectable difference at one start".
""")
PY
