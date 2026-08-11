#!/usr/bin/env bash
# Re-run only the metrics stage for points whose selection + orbital
# optimisation already completed.
#
# The Q4 array's OO stage succeeded; only the metrics stage failed, because
# metrics.py had been overwritten. Every affected point already has a valid
# metrics.in.json, so nothing needs re-optimising -- this recovers the campaign
# without repeating the expensive part.
#
# METRICS_ENTRY selects the metrics program. While metrics.py source is being
# recovered, point it at the verified compiled snapshot:
#   METRICS_ENTRY=metrics_good.pyc bash scripts/remetricise.sh
#
#SBATCH --job-name=qs_remetric
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=qs_remetric_%j.out
#SBATCH --error=qs_remetric_%j.err
#
#   DRY_RUN=1 bash scripts/remetricise.sh          # list what would be redone
#   METRICS_ENTRY=metrics_good.pyc bash scripts/remetricise.sh
#   sbatch --export=ALL scripts/remetricise.sh     # as a batch job
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

METRICS_ENTRY="${METRICS_ENTRY:-metrics.py}"
GLOB="${GLOB:-results/*_q4_grid/bond_*/U_*/*/*}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
STRICT="${STRICT_REF_WEIGHT:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -f "$METRICS_ENTRY" ]]; then
  echo "[remetric] metrics entry not found: $METRICS_ENTRY" >&2
  exit 1
fi
echo "[remetric] entry: $METRICS_ENTRY ($(stat -c %s "$METRICS_ENTRY") bytes)"

# Points with a completed OO stage but no usable metrics.json.
mapfile -t TODO < <(
  for d in $GLOB; do
    [[ -f "$d/metrics.in.json" ]] || continue
    if [[ -f "$d/metrics.json" ]] && grep -q '"K"' "$d/metrics.json" 2>/dev/null; then
      continue
    fi
    echo "$d"
  done
)
echo "[remetric] ${#TODO[@]} point(s) need the metrics stage"

if (( ${#TODO[@]} == 0 )); then echo "[remetric] nothing to do"; exit 0; fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '  %s\n' "${TODO[@]:0:10}"; (( ${#TODO[@]} > 10 )) && echo "  ..."
  exit 0
fi

EXTRA=()
[[ "$STRICT" == "1" ]] && EXTRA+=(--strict_reference_weight)

ok=0; bad=0
for d in "${TODO[@]}"; do
  # --sector_backend clifford is REQUIRED: the determinant path does not build
  # the exact/LAS split, so it emits no n_exact / exact_sector / r_sp / r_qubit
  # and audit_08's preconditions can never be met.
  if python -u "$METRICS_ENTRY" "$d/metrics.in.json" \
        --sector_backend clifford \
        --backend fci --coupled_energy_method reference --overlap_reference fci \
        --states_per_sector "$STATES_PER_SECTOR" \
        --outname "$d/metrics.json" "${EXTRA[@]}" > "$d/metrics.log" 2>&1; then
    K=$(python3 -c "import json,sys; print(json.load(open('$d/metrics.json')).get('K'))" 2>/dev/null || echo "?")
    echo "  ok   K=$K  $d"; ok=$((ok+1))
  else
    echo "  FAIL $d  (see $d/metrics.log)"; bad=$((bad+1))
  fi
done
echo "[remetric] done: $ok ok, $bad failed"
echo "[remetric] next: bash scripts/audit/audit_08_q4_compare.sh"
