#!/usr/bin/env bash
#SBATCH --job-name=qs_m35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-125
#SBATCH --output=qs_m35_%A_%a.out
#SBATCH --error=qs_m35_%A_%a.err
#
# REDUCED-BUDGET Q4 CAMPAIGN:  M = 3 (H2O) / 5 (N2)
#
# WHY THIS EXISTS
#   The M = 4/6 campaign returned K identical for iterative and exhaustive at
#   42/42 points (mean dK = 0.00, range [+0,+0]). audit_09 showed that is NOT a
#   bug and NOT a property of the selection rules:
#
#     dim(Q) = n - r_sp  =  7-3 = 4 (H2O),  10-4 = 6 (N2)
#
#   so at M = n - r_sp ANY M independent rows span the WHOLE quotient. Every
#   arm is forced to the same group, hence the same sector partition -- the
#   invariant check cannot fail, and the only channel left for a rule to affect
#   K is the orbital-optimisation landscape. K is integer-valued and does not
#   resolve the resulting 1e-4..1e-2 differences in U.
#
#   At M < n - r_sp the rules select genuinely DIFFERENT subspaces of Q, so the
#   partition itself differs and the comparison acquires resolving power. That
#   is the regime this campaign probes.
#
# WRITES TO results/{mol}_m35_grid/ -- does not touch the M=4/6 campaign.
#
#   DRY_RUN=1 bash scripts/trillium_m35_grid.sh     # show plan, verify inputs
#   bash scripts/trillium_m35_grid.sh               # verify then submit
#   PREBUILD=1 bash scripts/trillium_m35_grid.sh    # force chk/exact rebuild
#
# QUOTA -- THE ONE RECORDED CHOICE HERE.
#   run_endpoint_point.py refuses to derive the greedy quota split when
#   M != n - r_sp, because it is a decision, not a consequence. At M = 4/6 you
#   fixed (2,2)/(3,3). Reducing by one, the default below drops ONE QUARTET and
#   keeps the singles count, so the singles arm stays comparable across budgets:
#       H2O  M=3 -> (n_singles, n_quartets) = (2, 1)
#       N2   M=5 -> (3, 2)
#   Override if you want the single dropped instead:
#       H2O_QUOTA="1 2" N2_QUOTA="2 3" bash scripts/trillium_m35_grid.sh
#   The quota constrains only the greedy arm; `iterative` ignores it.

set -euo pipefail
export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

read -r -a METHODS <<< "${METHODS:-mixed_overlap iterative exhaustive}"
read -r -a COSTS   <<< "${COSTS:-NC variance}"
read -r -a H2O_Q   <<< "${H2O_QUOTA:-2 1}"
read -r -a N2_Q    <<< "${N2_QUOTA:-3 2}"
U="${U:-irrep}"
RESULTS_ROOT_SUFFIX="${RESULTS_ROOT_SUFFIX:-m35_grid}"
CAMPAIGN="${CAMPAIGN:-m35_$(date +%Y%m%d)}"
MAXITER="${MAXITER:-100}"
MAX_MACRO="${MAX_MACRO:-20}"
STABLE_SPAN="${STABLE_SPAN:-2}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
STRICT="${STRICT_REF_WEIGHT:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

H2O_M="${H2O_M:-3}"
N2_M="${N2_M:-5}"

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

# ---- inputs: reuse the chks + exact matrices the M=4/6 run already built --
check_inputs() {
  python3 - "$PREBUILD_FORCE" <<'PY'
import subprocess, sys
from pathlib import Path
sys.path.insert(0, ".")

force = sys.argv[1] == "1"
GRID = {
  "h2o": dict(norb=7, pg="C2v", bonds=[0.958,1.1293333333,1.3006666667,1.472,
              1.6433333333,1.8146666667,1.986,2.1573333333,2.3286666667,2.5]),
  "n2":  dict(norb=10, pg="D2h", bonds=[1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]),
}
missing, ok = [], 0
for mol, g in GRID.items():
    for bond in g["bonds"]:
        tag = f"{bond:.4f}".replace(".", "p")
        out = Path("exact") / f"{mol}_norb{g['norb']}_bond{tag}_exact.txt"
        if out.exists() and not force:
            ok += 1
            continue
        missing.append((mol, bond, g, out))

print(f"[inputs] exact matrices present: {ok}/{sum(len(g['bonds']) for g in GRID.values())}")
if not missing:
    print("[inputs] reusing the M=4/6 campaign inputs -- nothing to build")
    raise SystemExit(0)

from src.pyscf_chk import ensure_hamiltonian_chk
fail = 0
for mol, bond, g, out in missing:
    tag = f"{bond:.4f}".replace(".", "p")
    try:
        chk = ensure_hamiltonian_chk(mol, bond, "sto-3g", g["pg"],
                                     require_irreps=True,
                                     log_dir="results/_endpoint_ham")
    except Exception as exc:                                    # noqa: BLE001
        print(f"[inputs] FAIL chk {mol} {bond}: {exc}"); fail += 1; continue
    r = subprocess.run(
        [sys.executable, "-u", "scripts/write_default_exact_parity.py",
         "--molecule", mol, "--norb", str(g["norb"]),
         "--chk", str(chk), "-o", str(out)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[inputs] FAIL exact {mol} {bond}:\n{r.stdout}\n{r.stderr}")
        fail += 1; continue
    print(f"[inputs] built {out.name}")
if fail:
    raise SystemExit(f"[inputs] {fail} failure(s) -- not submitting")
PY
}

# ---- dispatch ------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "=== reduced-budget campaign (M < n - r_sp) ==="
  echo "arms   : ${METHODS[*]}"
  echo "costs  : ${COSTS[*]}"
  echo "budget : M = ${H2O_M} (H2O, quotient dim 4) / ${N2_M} (N2, quotient dim 6)"
  echo "quota  : H2O (${H2O_Q[0]},${H2O_Q[1]})   N2 (${N2_Q[0]},${N2_Q[1]})   <-- recorded choice"
  echo "U      : $U   campaign=$CAMPAIGN   tasks=$NTASK"
  echo "output : results/{mol}_${RESULTS_ROOT_SUFFIX}/"
  echo
  echo "At these M the LAS span is a PROPER subspace of the quotient, so the"
  echo "partition differs between arms -- unlike the M=4/6 run."
  echo
  printf '  %s\n' "${TASKS[@]:0:4}"; echo "  ..."
  PREBUILD_FORCE="${PREBUILD:-0}"
  export PREBUILD_FORCE
  check_inputs
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo; echo "[dry-run] would submit:"
    echo "  sbatch --export=ALL --array=0-$((NTASK-1)) $0"
    exit 0
  fi
  sbatch --export=ALL --array=0-$((NTASK-1)) "$0"
  echo
  echo "when finished:"
  echo "  RESULTS_SUFFIX=m35_grid bash scripts/audit/audit_08_q4_compare.sh"
  echo "  RESULTS_SUFFIX=m35_grid bash scripts/audit/audit_09_arm_equivalence.sh"
  exit 0
fi

T="${SLURM_ARRAY_TASK_ID}"
(( T >= NTASK )) && { echo "[skip] task $T of $NTASK"; exit 0; }
IFS=':' read -r MOL BOND METHOD COST <<< "${TASKS[$T]}"

if [[ "$MOL" == "h2o" ]]; then
  PG=C2v; NSYM="$H2O_M"; NS="${H2O_Q[0]}"; NQ="${H2O_Q[1]}"
else
  PG=D2h; NSYM="$N2_M";  NS="${N2_Q[0]}";  NQ="${N2_Q[1]}"
fi

EXTRA=()
[[ "$STRICT" == "1" ]] && EXTRA+=(--strict_reference_weight)

echo "[m35] task=$T $MOL R=$BOND $METHOD/$COST M=$NSYM quota=($NS,$NQ) U=$U"
python -u scripts/run_endpoint_point.py \
  --molecule "$MOL" --bond "$BOND" --method "$METHOD" --cost_function "$COST" \
  --basis sto-3g --point_group "$PG" --orbital_rotation "$U" \
  --n_sym "$NSYM" --n_singles "$NS" --n_quartets "$NQ" \
  --maxiter "$MAXITER" --m_round 1 \
  --max_macroiterations "$MAX_MACRO" --stable_span_iters "$STABLE_SPAN" \
  --iterative_reference fci_rotate \
  --states_per_sector "$STATES_PER_SECTOR" \
  --results_root "${MOL}_${RESULTS_ROOT_SUFFIX}" \
  --campaign "$CAMPAIGN" \
  --results_csv "tables/${MOL}/m35_grid.csv" \
  "${EXTRA[@]}"
echo "[m35] finished task=$T at $(date -Is)"
