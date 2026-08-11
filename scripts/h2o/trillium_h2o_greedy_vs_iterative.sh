#!/usr/bin/env bash
#SBATCH --job-name=h2o_sel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=h2o_sel_%A_%a.out
#SBATCH --error=h2o_sel_%A_%a.err
#
# Trillium: H2O greedy (quota 3 singles + 2 quartets) vs iterative (n_sym=5)
# for NC and variance, using FCI reference (ffsim / PySCF), not DMRG.
#
#   0 = greedy    + NC
#   1 = greedy    + variance
#   2 = iterative + NC
#   3 = iterative + variance
#
# Do NOT pass --mem-per-cpu on Trillium. Max wall time is 24h.
#
# Submit from repo root:
#   sbatch scripts/h2o/trillium_h2o_greedy_vs_iterative.sh
#
# Optional env overrides:
#   MOL, N_SINGLES, N_QUARTETS, N_SYM, M_ROUND, MAXITER, RESULTS_CSV

set -euo pipefail

export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

MOL="${MOL:-hamiltonians/water/H2O_OH0.9580_104.5000.FCIDUMP}"
N_SINGLES="${N_SINGLES:-3}"
N_QUARTETS="${N_QUARTETS:-2}"
N_SYM="${N_SYM:-$((N_SINGLES + N_QUARTETS))}"
M_ROUND="${M_ROUND:-1}"
MAXITER="${MAXITER:-100}"
RESULTS_CSV="${RESULTS_CSV:-tables/h2o/select_greedy_vs_iterative.csv}"

if [[ ! -f "$MOL" ]]; then
  echo "[error] missing MOL=$MOL" >&2
  exit 1
fi

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
case "$TASK" in
  0) SELECT=greedy;    COST=NC ;;
  1) SELECT=greedy;    COST=variance ;;
  2) SELECT=iterative; COST=NC ;;
  3) SELECT=iterative; COST=variance ;;
  *)
    echo "[error] unexpected array task id=$TASK (expected 0..3)" >&2
    exit 1
    ;;
esac

STEM="$(basename "$MOL")"
STEM="${STEM%.FCIDUMP}"
STEM="${STEM%.chk}"
OUT_DIR="results/h2o_select/${STEM}"
mkdir -p "$OUT_DIR" "$(dirname "$RESULTS_CSV")"

TAG="${SELECT}_${COST}_fci"
if [[ "$SELECT" == "greedy" ]]; then
  TAG="${TAG}_s${N_SINGLES}q${N_QUARTETS}"
else
  TAG="${TAG}_n${N_SYM}"
fi
OUTNAME="${OUT_DIR}/${TAG}"

echo "[job] $(date -Is)"
echo "[job] host=$(hostname) task=$TASK/${SLURM_ARRAY_TASK_COUNT:-4}"
echo "[job] MOL=$MOL SELECT=$SELECT COST=$COST reference=fci"
echo "[job] N_SINGLES=$N_SINGLES N_QUARTETS=$N_QUARTETS N_SYM=$N_SYM M_ROUND=$M_ROUND"
echo "[job] MAXITER=$MAXITER OUTNAME=$OUTNAME RESULTS_CSV=$RESULTS_CSV"

# FCI / ffsim path (not optimize_dmrg). Requires pyscf+ffsim in the venv.
# Quota flags must be on the argv for greedy — do not rely on a later CMD+=.
case "$SELECT" in
  greedy)
    CMD=(
      python -u optimize_symmetries.py "$MOL"
      --reference fci
      --select greedy
      --cost_function "$COST"
      --candidates senquart
      --n_singles "$N_SINGLES"
      --n_quartets "$N_QUARTETS"
      --maxiter "$MAXITER"
      --outname "$OUTNAME"
    )
    ;;
  iterative)
    CMD=(
      python -u optimize_symmetries.py "$MOL"
      --reference fci
      --select iterative
      --cost_function "$COST"
      --candidates senquart
      --n_sym "$N_SYM"
      --m_round "$M_ROUND"
      --maxiter "$MAXITER"
      --outname "$OUTNAME"
    )
    ;;
  *)
    echo "[error] SELECT=$SELECT (expected greedy|iterative)" >&2
    exit 1
    ;;
esac

printf '[run]'
printf ' %q' "${CMD[@]}"
printf '\n'

STARTED=$(date +%s)
STATUS=failed
MESSAGE=""
set +e
"${CMD[@]}"
RC=$?
set -e
ELAPSED=$(( $(date +%s) - STARTED ))
if [[ $RC -eq 0 ]]; then
  STATUS=ok
else
  MESSAGE="optimize_symmetries exited with code ${RC}"
fi

export RESULTS_CSV MOL SELECT COST N_SINGLES N_QUARTETS N_SYM M_ROUND OUTNAME STATUS MESSAGE ELAPSED
python -u -c '
from src.results_table import append_result_row
import os
select = os.environ["SELECT"]
append_result_row(
    os.environ["RESULTS_CSV"],
    {
        "molecule": os.environ["MOL"],
        "select": select,
        "cost_function": os.environ["COST"],
        "n_singles": os.environ["N_SINGLES"] if select == "greedy" else "",
        "n_quartets": os.environ["N_QUARTETS"] if select == "greedy" else "",
        "n_sym": os.environ["N_SYM"],
        "m_round": os.environ["M_ROUND"] if select == "iterative" else "",
        "final_cost": "",
        "selected_costs": "",
        "parity_output": "",
        "outname": os.environ["OUTNAME"],
        "status": os.environ["STATUS"],
        "elapsed_s": os.environ["ELAPSED"],
        "message": os.environ.get("MESSAGE", ""),
        "reference": "fci",
    },
)
'

if [[ $RC -ne 0 ]]; then
  echo "[error] finished SELECT=$SELECT COST=$COST status=$STATUS at $(date -Is)" >&2
  exit "$RC"
fi

echo "[ok] finished SELECT=$SELECT COST=$COST at $(date -Is)"
