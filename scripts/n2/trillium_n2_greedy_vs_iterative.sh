#!/usr/bin/env bash
#SBATCH --job-name=n2_sel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=n2_sel_%A_%a.out
#SBATCH --error=n2_sel_%A_%a.err
#
# Trillium: N2 greedy (quota 4 singles + 3 quartets) vs iterative (n_sym=7)
# for NC and variance. Four array tasks write stepwise rows to RESULTS_CSV.
#
#   0 = greedy    + NC
#   1 = greedy    + variance
#   2 = iterative + NC
#   3 = iterative + variance
#
# Do NOT pass --mem-per-cpu on Trillium. Max wall time is 24h.
# N2 is memory-heavy; lower N_THREADS / keep one array task occupancy if OOM.
#
# Submit from repo root:
#   sbatch scripts/n2/trillium_n2_greedy_vs_iterative.sh
#
# Optional env overrides:
#   MOL, N_SINGLES, N_QUARTETS, N_SYM, M_ROUND, BOND_DIM, MAXITER,
#   N_THREADS, RESULTS_CSV

set -euo pipefail

export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

MOL="${MOL:-hamiltonians/n2_1.2_ccpvdz_8o8e.FCIDUMP}"
N_SINGLES="${N_SINGLES:-4}"
N_QUARTETS="${N_QUARTETS:-3}"
N_SYM="${N_SYM:-$((N_SINGLES + N_QUARTETS))}"
M_ROUND="${M_ROUND:-1}"
BOND_DIM="${BOND_DIM:-250}"
MAXITER="${MAXITER:-100}"
N_THREADS="${N_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
RESULTS_CSV="${RESULTS_CSV:-tables/n2/select_greedy_vs_iterative.csv}"

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
OUT_DIR="results/n2_select/${STEM}"
mkdir -p "$OUT_DIR" "wavefunctions" "$(dirname "$RESULTS_CSV")"

TAG="${SELECT}_${COST}"
if [[ "$SELECT" == "greedy" ]]; then
  TAG="${TAG}_s${N_SINGLES}q${N_QUARTETS}"
else
  TAG="${TAG}_n${N_SYM}"
fi
OUTNAME="${OUT_DIR}/${TAG}"
WF_DIR="wavefunctions/${STEM}_${TAG}"
mkdir -p "$WF_DIR"

echo "[job] $(date -Is)"
echo "[job] host=$(hostname) task=$TASK/${SLURM_ARRAY_TASK_COUNT:-4}"
echo "[job] MOL=$MOL SELECT=$SELECT COST=$COST"
echo "[job] N_SINGLES=$N_SINGLES N_QUARTETS=$N_QUARTETS N_SYM=$N_SYM M_ROUND=$M_ROUND"
echo "[job] BOND_DIM=$BOND_DIM MAXITER=$MAXITER N_THREADS=$N_THREADS"
echo "[job] OUTNAME=$OUTNAME WF_DIR=$WF_DIR RESULTS_CSV=$RESULTS_CSV"

CMD=(
  python -u optimize_dmrg.py "$MOL"
  --select "$SELECT"
  --cost_function "$COST"
  --candidates senquart
  --bond_dim "$BOND_DIM"
  --maxiter "$MAXITER"
  --n_threads "$N_THREADS"
  --wavefunction_dir "$WF_DIR"
  --outname "$OUTNAME"
  --results_csv "$RESULTS_CSV"
  --verbose
)

if [[ "$SELECT" == "greedy" ]]; then
  CMD+=(--n_singles "$N_SINGLES" --n_quartets "$N_QUARTETS")
else
  CMD+=(--n_sym "$N_SYM" --m_round "$M_ROUND")
fi

printf '[run]'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[ok] finished SELECT=$SELECT COST=$COST at $(date -Is)"
