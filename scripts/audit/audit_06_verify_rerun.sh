#!/usr/bin/env bash
# AUDIT 6 — morning check on the fixed rerun. No pyscf, no FCI: reads the
# metrics.json files only. Seconds on a login node.
#
# Pass criteria, per point:
#   reference_weight_sum == 1   the retained sector holds the reference
#   n_exact == r_spatial + 2    no point-group generator left unpinned
#   converged == true           K actually reached chemical accuracy
#
# Run:  bash scripts/audit/audit_06_verify_rerun.sh
#       U=irrep bash scripts/audit/audit_06_verify_rerun.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
mkdir -p tables/audit
export U="${U:-full}"

python3 - <<'PY'
import json, os, sys
from pathlib import Path

sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo

U = os.environ.get("U", "full")
# Match metrics.py: 1e-6 flags numerical noise, real symmetry breaking is 1e-2+.
TOL = float(os.environ.get("REFERENCE_WEIGHT_TOL", "1e-4"))
BONDS = {
    "h2o": ["0.9580", "1.1293", "1.3007", "1.4720", "1.6433",
            "1.8147", "1.9860", "2.1573", "2.3287", "2.5000"],
    "n2":  ["1.2000", "1.3000", "1.4000", "1.5000", "1.6000", "1.7000",
            "1.8000", "1.9000", "2.0000", "2.1000", "2.2000"],
}
METHODS = ["mixed_disjoint", "mixed_overlap", "iterative"]
COSTS = ["NC", "variance"]
# Fully pinned document exact-sector dims, from independent determinant counts
# (audit 1); they match the PDF.
DOC_SECTOR_DIM = {"h2o": 133, "n2": 1824}

bad, missing, stale_files, total = [], [], [], 0
for mol, bonds in BONDS.items():
    r_spatial = 2 if mol == "h2o" else 3
    print(f"\n{'=' * 104}")
    print(f"{mol.upper()}  U_{U}   document exact-sector dim = {DOC_SECTOR_DIM[mol]}")
    print(f"{'=' * 104}")
    print(f"{'bond':>8} {'method':>15} {'cost':>9} {'K':>6} {'dim':>6} "
          f"{'conv':>5} {'ref_w':>9} {'n_ex':>5} {'sector':>8} {'verdict':>9}")
    for bond in bonds:
        for method in METHODS:
            for cost in COSTS:
                total += 1
                p = Path(f"results/{mol}_endpoint_grid/bond_{bond.replace('.','p')}"
                         f"/U_{U}/{method}/{cost}/metrics.json")
                if not p.is_file():
                    missing.append(str(p))
                    print(f"{bond:>8} {method:>15} {cost:>9} "
                          f"{'-':>6} {'-':>6} {'-':>5} {'-':>9} {'-':>5} "
                          f"{'-':>8} {'MISSING':>9}")
                    continue
                try:
                    d = load_oo(p)
                except Exception as exc:                   # noqa: BLE001
                    missing.append(f"{p}: {exc}")
                    continue
                # A metrics.json written BEFORE the 2026-08-07 fix has neither
                # of these keys. If the job failed, the pre-fix file is still on
                # disk and would otherwise be scored as a fresh bad result.
                stale = ("reference_weight_ok" not in d
                         and "exact_sector_source" not in d)
                w = d.get("reference_weight_sum")
                w = float(w) if w is not None else 0.0
                nex = int(d.get("n_exact", -1))
                if stale:
                    stale_files.append((mol, bond, method, cost))
                    print(f"{bond:>8} {method:>15} {cost:>9} "
                          f"{str(d.get('K')):>6} "
                          f"{str(d.get('relevant_sectors_total_dim')):>6} "
                          f"{'-':>5} {w:9.6f} {nex:>5} {'-':>8} "
                          f"{'STALE':>9}")
                    continue
                conv = bool(d.get("converged"))
                dim = d.get("relevant_sectors_total_dim")
                sec = d.get("exact_sector")
                sec = "".join(str(int(b)) for b in sec) if sec else "-"
                fails = []
                if w <= 1.0 - TOL:
                    fails.append("ref_w")
                if nex != r_spatial + 2:
                    fails.append("n_exact")
                if not conv:
                    fails.append("K")
                verdict = "ok" if not fails else "FAIL:" + "+".join(fails)
                if fails:
                    bad.append((mol, bond, method, cost, verdict))
                print(f"{bond:>8} {method:>15} {cost:>9} {str(d.get('K')):>6} "
                      f"{str(dim):>6} {('yes' if conv else 'NO'):>5} {w:9.6f} "
                      f"{nex:>5} {sec:>8} {verdict:>9}")

print(f"\n{'=' * 104}")
print(f"points checked : {total}")
print(f"missing        : {len(missing)}")
print(f"stale (pre-fix): {len(stale_files)}")
print(f"failing        : {len(bad)}")
if stale_files:
    print("""
STALE means the metrics.json on disk predates the fix -- the job for that point
did not produce a new one, so it FAILED. These are not bad science, they are
crashed jobs. Get the real error:

  for p in $(printf '%s\\n' """ + " ".join(
        f"results/{m}_endpoint_grid/bond_{b.replace('.','p')}/U_${{U:-full}}/{me}/{c}"
        for m, b, me, c in stale_files[:12]) + """); do
    echo "== $p"; tail -5 "$p"/oo.log "$p"/metrics.log 2>/dev/null
  done""")
    for row in stale_files:
        print("   stale:", row)
if bad:
    print("\nfirst failures:")
    for row in bad[:15]:
        print("   ", row)
    print("""
ref_w   the retained sector still does not hold the reference. Check the
        [endpoint] orbsym / supports lines in the job .out -- if the derived
        supports look right, the exact_sector label is the suspect.
n_exact an exact generator is still being dropped at the qubit level.
K       symmetries are fine but the pool genuinely cannot reach chemical
        accuracy at that geometry. This one is a RESULT, not a bug.""")
if not bad and not stale_files and not missing:
    print("\nall points clean: reference retained, all generators pinned, K converged.")
    print("Next: the n_sym < N - r discrimination wave (see")
    print("scripts/trillium_rerun_fixed_grid.sh footer).")
elif not bad:
    print(f"\nno bad results, but {len(stale_files) + len(missing)} point(s) have "
          "no post-fix metrics. Diagnose the crashes, then rerun -- "
          "RESUME=1 (the default) will pick exactly those points.")
PY
