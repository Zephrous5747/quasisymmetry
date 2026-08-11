#!/usr/bin/env bash
# Sec.4 exhaustive quotient-pool check over the endpoint grid.
#
#SBATCH --job-name=qs_exh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-125
#SBATCH --output=qs_exh_%A_%a.out
#SBATCH --error=qs_exh_%A_%a.err
#
# One array task per endpoint point. Each task rescores every nontrivial parity
# class modulo the exact symmetries at that point's converged orbitals, redoes
# the greedy independence scan, and compares against the polynomial pool
# (Sec.4.3). Writes exhaustive_<scope>.json next to each oo.json.
#
# Pool sizes:
#   spatial scope (what the implementation can actually select):  31 / 127
#   qubit   scope (the literal Sec.4.1 pool):                   1023 / 32767
#
# Cost: one Hamiltonian application per candidate. Spatial scope is seconds.
# Qubit scope is ~32767 contractions for N2 -- minutes, hence the 6h wall.
#
#   SCOPE=spatial sbatch --export=ALL scripts/trillium_exhaustive_check.sh
#   SCOPE=qubit   sbatch --export=ALL scripts/trillium_exhaustive_check.sh
#   DRY_RUN=1 bash scripts/trillium_exhaustive_check.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

SCOPE="${SCOPE:-spatial}"
U="${U:-irrep}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t POINTS < <(ls -1d \
  results/n2_endpoint_grid/bond_*/U_${U}/*/*/oo.json \
  results/h2o_endpoint_grid/bond_*/U_${U}/*/*/oo.json 2>/dev/null | sort)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] scope=$SCOPE  U=$U  ${#POINTS[@]} point(s)"
  printf '  %s\n' "${POINTS[@]:0:4}"
  echo "  ..."
  echo "[dry-run] submit with: SCOPE=$SCOPE sbatch --export=ALL --array=0-$(( ${#POINTS[@]} - 1 )) $0"
  exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[serial] no SLURM_ARRAY_TASK_ID -- running all ${#POINTS[@]} points here"
  for p in "${POINTS[@]}"; do
    python -u scripts/exhaustive_quotient_check.py --oo "$p" --scope "$SCOPE"
  done
  exit 0
fi

T="${SLURM_ARRAY_TASK_ID}"
if (( T >= ${#POINTS[@]} )); then
  echo "[skip] task $T beyond ${#POINTS[@]} points"
  exit 0
fi
echo "[exh] task=$T scope=$SCOPE ${POINTS[$T]}"
python -u scripts/exhaustive_quotient_check.py --oo "${POINTS[$T]}" --scope "$SCOPE"
