#!/usr/bin/env bash
# Q4 campaign: compare three LAS selection rules at the corrected budget.
#
#   arm 1  mixed_overlap   greedy quota, one-shot at the initial frame
#   arm 2  iterative       NC-ranked frame-local scan, reselected each macro
#   arm 3  exhaustive      self-consistent fixed point over the complete
#                          quotient pool
#
# All three with orbital optimisation, irrep-block packing, identity start.
# Budget M = n - r_sp = 4 (H2O) / 6 (N2) with total particle parity included in
# the exact set; greedy quota (2,2) / (3,3). These are recorded decisions, set
# as driver defaults -- do not override without recording why.
#
# WRITES TO results/{mol}_q4_grid/ so the M=5/7 campaign under
# results/{mol}_endpoint_grid/ is untouched.
#
#SBATCH --job-name=qs_q4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-125
#SBATCH --output=qs_q4_%A_%a.out
#SBATCH --error=qs_q4_%A_%a.err
#
#   DRY_RUN=1 bash scripts/trillium_q4_grid.sh      # show the plan + prebuild
#   bash scripts/trillium_q4_grid.sh                # prebuild then submit
#
# The greedy arm defaults to mixed_overlap (plain quota). To use the
# orbital-disjoint variant instead, or to run both:
#   METHODS="mixed_disjoint iterative exhaustive" bash scripts/trillium_q4_grid.sh
#   METHODS="mixed_overlap mixed_disjoint iterative exhaustive" ...   (168 tasks)
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

read -r -a METHODS <<< "${METHODS:-mixed_overlap iterative exhaustive}"
read -r -a COSTS   <<< "${COSTS:-NC variance}"
U="${U:-irrep}"
RESULTS_ROOT_SUFFIX="${RESULTS_ROOT_SUFFIX:-q4_grid}"
CAMPAIGN="${CAMPAIGN:-q4_$(date +%Y%m%d)}"
MAXITER="${MAXITER:-100}"
MAX_MACRO="${MAX_MACRO:-20}"
STABLE_SPAN="${STABLE_SPAN:-2}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
STRICT="${STRICT_REF_WEIGHT:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

H2O_BONDS=(0.958 1.1293333333 1.3006666667 1.472 1.6433333333
           1.8146666667 1.986 2.1573333333 2.3286666667 2.5)
N2_BONDS=(1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2)

# ---- flat task list: (molecule, bond, method, cost) ----------------------
TASKS=()
for b in "${H2O_BONDS[@]}"; do
  for m in "${METHODS[@]}"; do
    for c in "${COSTS[@]}"; do TASKS+=("h2o:$b:$m:$c"); done
  done
done
for b in "${N2_BONDS[@]}"; do
  for m in "${METHODS[@]}"; do
    for c in "${COSTS[@]}"; do TASKS+=("n2:$b:$m:$c"); done
  done
done
NTASK=${#TASKS[@]}

# ---- serial prebuild: chks and per-geometry exact matrices ---------------
# The exact matrices CHANGE in this campaign (total particle parity is now
# included), and up to 6 array tasks share each geometry, so they must be
# rebuilt once, serially, before any job starts.
prebuild() {
  echo "[prebuild] chk + exact matrices (now including total particle parity)"
  python3 - <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.pyscf_chk import ensure_hamiltonian_chk

GRID = {
  "h2o": dict(norb=7, pg="C2v", angle=104.5,
              bonds=[0.958,1.1293333333,1.3006666667,1.472,1.6433333333,
                     1.8146666667,1.986,2.1573333333,2.3286666667,2.5]),
  "n2":  dict(norb=10, pg="D2h", angle=None,
              bonds=[1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]),
}
REPO = Path(".").resolve()
fail = 0
for mol, g in GRID.items():
    for bond in g["bonds"]:
        tag = f"{bond:.4f}".replace(".", "p")
        try:
            chk = ensure_hamiltonian_chk(REPO, mol, bond, "sto-3g", g["angle"],
                                         g["pg"], require_irreps=True,
                                         log_dir="results/_endpoint_ham")
        except Exception as exc:                                # noqa: BLE001
            print(f"[prebuild] FAIL chk {mol} {bond}: {exc}"); fail += 1; continue
        out = Path("exact") / f"{mol}_norb{g['norb']}_bond{tag}_exact.txt"
        r = subprocess.run(
            [sys.executable, "-u", "scripts/write_default_exact_parity.py",
             "--molecule", mol, "--norb", str(g["norb"]),
             "--chk", str(chk), "-o", str(out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[prebuild] FAIL exact {mol} {bond}:\n{r.stdout}\n{r.stderr}")
            fail += 1; continue
        note = next((l for l in r.stdout.splitlines()
                     if "total particle parity" in l), "")
        print(f"[prebuild] {mol} R={bond:.4f} -> {out.name}  {note.strip()}")
if fail:
    raise SystemExit(f"[prebuild] {fail} failure(s) -- not submitting")
print("[prebuild] all geometries ready")
PY
  sync || true; sleep 2
}

# ---- dispatch ------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "=== Q4 campaign ==="
  echo "arms   : ${METHODS[*]}"
  echo "costs  : ${COSTS[*]}"
  echo "U      : $U   campaign=$CAMPAIGN   tasks=$NTASK"
  echo "budget : M = n - r_sp (4 H2O / 6 N2), quota (2,2)/(3,3), identity start"
  echo "output : results/{mol}_${RESULTS_ROOT_SUFFIX}/"
  echo
  printf '  %s\n' "${TASKS[@]:0:4}"; echo "  ..."
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo; echo "[dry-run] would prebuild, then:"
    echo "  sbatch --export=ALL --array=0-$((NTASK-1)) $0"
    exit 0
  fi
  prebuild
  sbatch --export=ALL --array=0-$((NTASK-1)) "$0"
  echo
  echo "when finished:  bash scripts/audit/audit_08_q4_compare.sh"
  exit 0
fi

T="${SLURM_ARRAY_TASK_ID}"
(( T >= NTASK )) && { echo "[skip] task $T of $NTASK"; exit 0; }
IFS=':' read -r MOL BOND METHOD COST <<< "${TASKS[$T]}"
PG=$([[ "$MOL" == "h2o" ]] && echo C2v || echo D2h)

EXTRA=()
[[ "$STRICT" == "1" ]] && EXTRA+=(--strict_reference_weight)

echo "[q4] task=$T $MOL R=$BOND $METHOD/$COST U=$U"
python -u scripts/run_endpoint_point.py \
  --molecule "$MOL" --bond "$BOND" --method "$METHOD" --cost_function "$COST" \
  --basis sto-3g --point_group "$PG" --orbital_rotation "$U" \
  --maxiter "$MAXITER" --m_round 1 \
  --max_macroiterations "$MAX_MACRO" --stable_span_iters "$STABLE_SPAN" \
  --iterative_reference fci_rotate \
  --states_per_sector "$STATES_PER_SECTOR" \
  --results_root "${MOL}_${RESULTS_ROOT_SUFFIX}" \
  --campaign "$CAMPAIGN" \
  --results_csv "tables/${MOL}/q4_grid.csv" \
  "${EXTRA[@]}"
echo "[q4] finished task=$T at $(date -Is)"
