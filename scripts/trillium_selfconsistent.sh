#!/usr/bin/env bash
# Sec.4.2 self-consistent exhaustive fixed point over the endpoint grid.
#
#SBATCH --job-name=qs_scfp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-125
#SBATCH --output=qs_scfp_%A_%a.out
#SBATCH --error=qs_scfp_%A_%a.err
#
# Cost per point: every objective evaluation re-solves FCI (route A, Eq.20) and
# applies H once per generator, and L-BFGS-B uses finite differences over
# n_params (6-21). Budget minutes (H2O) to ~1 h (N2) per point per macroiteration
# set, hence the 12 h wall.
#
# Env:
#   M_BUDGET   LAS budget. UNSET = the run's M (= N-r), where the span test is
#              trivially satisfied and only a basis-dependent sum is minimised.
#              Set 4 (H2O) / 6 (N2) for the regime with physical content.
#   START      run | identity | random   (Sec.3.2/7.1 initialisation robustness)
#   U          irrep (default) or full
#
#   DRY_RUN=1 bash scripts/trillium_selfconsistent.sh
#   M_BUDGET=6 START=run sbatch --export=ALL --array=0-65 scripts/trillium_selfconsistent.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

U="${U:-irrep}"
START="${START:-run}"
SCOPE="${SCOPE:-spatial}"
MAX_MACRO="${MAX_MACRO:-20}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA=()
[[ -n "${M_BUDGET:-}" ]] && EXTRA+=(--M "$M_BUDGET")
[[ -n "${SEED:-}" ]] && EXTRA+=(--seed "$SEED")

mapfile -t POINTS < <(ls -1d \
  results/n2_endpoint_grid/bond_*/U_${U}/*/*/oo.json \
  results/h2o_endpoint_grid/bond_*/U_${U}/*/*/oo.json 2>/dev/null | sort)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] ${#POINTS[@]} point(s)  U=$U start=$START scope=$SCOPE "
  echo "[dry-run] M_BUDGET=${M_BUDGET:-<run default = N-r>} max_macro=$MAX_MACRO"
  printf '  %s\n' "${POINTS[@]:0:3}"; echo "  ..."
  echo "[dry-run] sbatch --export=ALL --array=0-$(( ${#POINTS[@]} - 1 )) $0"
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[serial] running all ${#POINTS[@]} points in this shell"
  for p in "${POINTS[@]}"; do
    python -u scripts/exhaustive_selfconsistent.py --oo "$p" --scope "$SCOPE" \
      --start "$START" --max-macro "$MAX_MACRO" "${EXTRA[@]}"
  done
  exit 0
fi

T="${SLURM_ARRAY_TASK_ID}"
(( T >= ${#POINTS[@]} )) && { echo "[skip] task $T"; exit 0; }
echo "[scfp] task=$T ${POINTS[$T]}"
python -u scripts/exhaustive_selfconsistent.py --oo "${POINTS[$T]}" \
  --scope "$SCOPE" --start "$START" --max-macro "$MAX_MACRO" "${EXTRA[@]}"
