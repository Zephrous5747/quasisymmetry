#!/usr/bin/env bash
# Overnight rerun of the endpoint grid with the 2026-08-07 fixes applied.
#
# What changed since the campaign this replaces:
#   1. Exact generator supports are derived PER GEOMETRY from the MO irrep
#      labels in each .chk, not from hardcoded orbital indices.
#      (run_endpoint_point.py now builds the chk BEFORE the exact matrix.)
#   2. metrics.py counts n_exact from the rows that actually survived the
#      qubit-level GF(2) filter, so no point-group generator is left unpinned.
#   3. The exact sector defaults to the sector holding the REFERENCE
#      determinant, not the densest sector.
#   4. reference_weight_sum is checked; with STRICT_REF_WEIGHT=1 a point that
#      cannot possibly converge fails immediately instead of burning hours.
#
# Usage (from the repo root on Trillium):
#   bash scripts/trillium_rerun_fixed_grid.sh                 # preflight + submit
#   DRY_RUN=1 bash scripts/trillium_rerun_fixed_grid.sh       # show what would go
#   RESUME=0 bash scripts/trillium_rerun_fixed_grid.sh        # redo every point
#   ONLY=h2o bash scripts/trillium_rerun_fixed_grid.sh
#   N_SYM=4  bash scripts/trillium_rerun_fixed_grid.sh        # discrimination wave
#
# Morning check:
#   bash scripts/audit/audit_06_verify_rerun.sh
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"                 # all | n2 | h2o
RESUME="${RESUME:-1}"               # 1 = skip points that are already clean
ORBITAL_ROTATIONS="${ORBITAL_ROTATIONS:-full,irrep}"
BACKUP="${BACKUP:-1}"

# Protocol: back to the PDF values, not the truncated resubmit defaults.
export MAXITER="${MAXITER:-100}"
export MAX_MACRO="${MAX_MACRO:-20}"
export STABLE_SPAN="${STABLE_SPAN:-2}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"
export STRICT_REF_WEIGHT="${STRICT_REF_WEIGHT:-1}"
export CAMPAIGN="${CAMPAIGN:-orbsym_exact_$(date +%Y%m%d)}"
[[ -n "${N_SYM:-}" ]] && export N_SYM

# IMPORTANT: do NOT export EXACT_PARITY. A single shared exact matrix is exactly
# the bug being fixed -- run_endpoint_point.py now writes one per geometry from
# that geometry's chk.
unset EXACT_PARITY EXACT_PARITY_EXTRA EXACT_SECTOR || true

echo "=== fixed-grid rerun ==="
echo "campaign=$CAMPAIGN  U={$ORBITAL_ROTATIONS}  resume=$RESUME  strict=$STRICT_REF_WEIGHT"
echo "MAXITER=$MAXITER MAX_MACRO=$MAX_MACRO STABLE_SPAN=$STABLE_SPAN SPS=$STATES_PER_SECTOR"
echo "n_sym=${N_SYM:-document default (5 h2o / 7 n2)}"

# ---------------------------------------------------------------- preflight ---
echo
echo "[preflight] script version fingerprint:"
grep -m1 ENDPOINT_SCRIPT_VERSION scripts/run_endpoint_point.py

echo "[preflight] the fixed code must be importable and the supports derivable"
python3 - <<'PY'
import sys
sys.path.insert(0, "scripts")
from src.sto3g_exact_symmetries import (
    exact_spatial_sets_from_orbsym, hardcoded_supports_valid)
from src.clifford_sectors import reference_sector_label   # new helper must exist
import inspect
from src import clifford_sectors
src = inspect.getsource(clifford_sectors.clifford_symmetries_from_spatial)
assert "spatial_kept_indices" in src, "clifford_sectors.py is STALE -- re-sync"
import metrics
assert hasattr(metrics, "_check_reference_weight"), "metrics.py is STALE -- re-sync"
o = ['A1','A1','B1','B2','A1','A1','B2']
d = {n: sorted(s) for n, s in exact_spatial_sets_from_orbsym(o, 'C2v')}
assert d == {'Q_B1': [2], 'Q_B2': [3, 6]}, d
assert hardcoded_supports_valid(o, 'h2o') is False
print("[preflight] OK: fixed modules present and support derivation correct")
PY

# ------------------------------------------------------------------ backup ---
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$BACKUP" == "1" && "$DRY_RUN" != "1" ]]; then
  mkdir -p archive
  for mol in h2o n2; do
    if [[ -d "results/${mol}_endpoint_grid" ]]; then
      echo "[backup] results/${mol}_endpoint_grid -> archive/${mol}_endpoint_grid_${STAMP}"
      cp -a "results/${mol}_endpoint_grid" "archive/${mol}_endpoint_grid_${STAMP}"
    fi
    if [[ -f "tables/${mol}/endpoint_grid.csv" ]]; then
      cp -a "tables/${mol}/endpoint_grid.csv" \
            "archive/${mol}_endpoint_grid_${STAMP}.csv"
    fi
  done
  echo "[backup] the superseded campaign is the evidence for the retraction --"
  echo "[backup] archive/ keeps it. Do not delete before the note is finalised."
fi

# ---------------------------------------------------------------- prebuild ---
# Serial, BEFORE any sbatch. Two shared resources would otherwise be built
# concurrently by up to 12 array tasks per geometry:
#   * the PySCF .chk  -- HDF5 has no safe concurrent-create story; the previous
#     submitter had prebuild_pg_chks() for exactly this reason
#   * exact/<mol>_norb<n>_bond<tag>_exact.txt -- now per geometry, and every
#     task for that geometry rewrites it
PREBUILD="${PREBUILD:-1}"
prebuild_all() {
  echo
  echo "[prebuild] serial chk + per-geometry exact matrices (no array races)"
  python3 - "$ONLY" <<'PY'
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.pyscf_chk import ensure_hamiltonian_chk

only = sys.argv[1]
REPO = Path(".").resolve()
GRID = {
    "h2o": dict(
        norb=7, pg="C2v", angle=104.5,
        bonds=[0.958, 1.1293333333, 1.3006666667, 1.472, 1.6433333333,
               1.8146666667, 1.986, 2.1573333333, 2.3286666667, 2.5]),
    "n2": dict(
        norb=10, pg="D2h", angle=None,
        bonds=[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]),
}
mols = ["h2o", "n2"] if only == "all" else [only]
fail = 0
for mol in mols:
    g = GRID[mol]
    for bond in g["bonds"]:
        tag = f"{bond:.4f}".replace(".", "p")
        try:
            chk = ensure_hamiltonian_chk(
                REPO, mol, bond, "sto-3g", g["angle"], g["pg"],
                require_irreps=True, log_dir="results/_endpoint_ham")
        except Exception as exc:                            # noqa: BLE001
            print(f"[prebuild] FAIL chk {mol} {bond}: {exc}")
            fail += 1
            continue
        out = Path("exact") / f"{mol}_norb{g['norb']}_bond{tag}_exact.txt"
        r = subprocess.run(
            [sys.executable, "-u", "scripts/write_default_exact_parity.py",
             "--molecule", mol, "--norb", str(g["norb"]),
             "--chk", str(chk), "-o", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[prebuild] FAIL exact {mol} {bond}:\n{r.stdout}\n{r.stderr}")
            fail += 1
            continue
        supports = [ln for ln in r.stdout.splitlines() if "=" in ln and "[exact]   " in ln]
        stale = "STALE" if any("WRONG" in ln for ln in r.stdout.splitlines()) else "ok"
        print(f"[prebuild] {mol} R={bond:.4f} -> {out.name}  "
              f"hardcoded={stale}  {' '.join(s.split('[exact]   ')[1] for s in supports)}")
if fail:
    raise SystemExit(f"[prebuild] {fail} failure(s) -- not submitting")
print("[prebuild] all geometries ready")
PY
  sync || true
  sleep 2
}

if [[ "$PREBUILD" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would prebuild chks + per-geometry exact matrices"
  else
    prebuild_all
  fi
fi

# ------------------------------------------------------- which points to run --
# Emits the SLURM array index list for one molecule+packing.
needs_rerun_array() {
  local mol="$1" u="$2"
  python3 - "$mol" "$u" "$RESUME" <<'PY'
import json, sys
from pathlib import Path

mol, u, resume = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
BONDS = {
    "h2o": ["0.9580", "1.1293", "1.3007", "1.4720", "1.6433",
            "1.8147", "1.9860", "2.1573", "2.3287", "2.5000"],
    "n2":  ["1.2000", "1.3000", "1.4000", "1.5000", "1.6000", "1.7000",
            "1.8000", "1.9000", "2.0000", "2.1000", "2.2000"],
}[mol]
METHODS = ["mixed_disjoint", "mixed_overlap", "iterative"]
COSTS = ["NC", "variance"]
R_SPATIAL = 2 if mol == "h2o" else 3

def clean(mol, bond, method, cost):
    p = Path(f"results/{mol}_endpoint_grid/bond_{bond.replace('.', 'p')}"
             f"/U_{u}/{method}/{cost}/metrics.json")
    if not p.is_file():
        return False
    try:
        sys.path.insert(0, "scripts/audit")
        from _qs_json import load_oo
        d = load_oo(p)
    except Exception:
        return False
    w = d.get("reference_weight_sum")
    if w is None or float(w) <= 1.0 - 1e-6:
        return False
    if not d.get("converged"):
        return False
    if int(d.get("n_exact", -1)) != R_SPATIAL + 2:
        return False
    return True

idx = []
for g, bond in enumerate(BONDS):
    for mc in range(6):
        method, cost = METHODS[mc // 2], COSTS[mc % 2]
        if resume and clean(mol, bond, method, cost):
            continue
        idx.append(g * 6 + mc)
print(",".join(map(str, idx)))
PY
}

IFS=',' read -r -a U_MODES <<< "$ORBITAL_ROTATIONS"

do_h2o=0; do_n2=0
case "$ONLY" in
  all) do_h2o=1; do_n2=1 ;;
  h2o) do_h2o=1 ;;
  n2)  do_n2=1 ;;
  *) echo "unknown ONLY=$ONLY" >&2; exit 1 ;;
esac

submit() {
  local script="$1" array="$2" label="$3" u="$4"
  if [[ -z "$array" ]]; then
    echo "[skip] nothing to do for $label (all points already clean)"
    return 0
  fi
  local n
  n="$(awk -F, '{print NF}' <<< "$array")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $label U=$u  $n task(s)  --array=$array"
  else
    echo "[submit] $label U=$u  $n task(s)"
    ORBITAL_ROTATION="$u" sbatch --export=ALL --array="$array" "$script"
  fi
}

for u in "${U_MODES[@]}"; do
  u="$(tr -d '[:space:]' <<< "$u")"
  [[ -z "$u" ]] && continue
  [[ "$u" != "full" && "$u" != "irrep" ]] && { echo "bad U=$u" >&2; exit 1; }
  if [[ "$do_n2" == "1" ]]; then
    submit scripts/trillium_n2_endpoint_grid.sh \
      "$(needs_rerun_array n2 "$u")" "n2 endpoint" "$u"
  fi
  if [[ "$do_h2o" == "1" ]]; then
    submit scripts/trillium_h2o_endpoint_grid.sh \
      "$(needs_rerun_array h2o "$u")" "h2o endpoint" "$u"
  fi
done

echo
echo "[ok] submitted. Monitor:  squeue -u \$USER"
echo "[ok] In the morning:      bash scripts/audit/audit_06_verify_rerun.sh"
echo
echo "Second wave (only after this one is clean) -- lets the selection rules"
echo "actually differ by keeping span(E u LAS) below full rank:"
echo "  N_SYM=4 CAMPAIGN=nsym4_\$(date +%Y%m%d) bash scripts/trillium_rerun_fixed_grid.sh   # h2o"
echo "  N_SYM=6 CAMPAIGN=nsym6_\$(date +%Y%m%d) ONLY=n2 bash scripts/trillium_rerun_fixed_grid.sh"
