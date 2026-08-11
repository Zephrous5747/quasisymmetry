#!/usr/bin/env bash
# Submit the full Aug-2026 campaign on Trillium (max array parallelization).
#
# 1) H2O endpoint grid: 60 tasks (10 geom × 3 methods × 2 costs)
# 2) N2  endpoint grid: 66 tasks (11 geom × 3 methods × 2 costs)
# 3) Checklist supplement: 12 tasks (6 representative geom × 2 costs)
#
# Usage (from repo root on Trillium):
#   bash scripts/trillium_submit_all.sh
#
# Optional env:
#   EXACT_PARITY_EXTRA=exact/h2o_pg.txt   # append PG Z-rows into E
#   POINT_GROUP=none                     # skip PG-adapted ham (not recommended)
#   DRY_RUN=1                            # print sbatch lines only

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
submit() {
  local script="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] sbatch $* $script"
  else
    echo "[submit] sbatch $* $script"
    sbatch "$@" "$script"
  fi
}

EXPORT_ARGS=(--export=ALL)
if [[ -n "${EXACT_PARITY_EXTRA:-}" ]]; then
  EXPORT_ARGS=(--export=ALL,EXACT_PARITY_EXTRA="${EXACT_PARITY_EXTRA}")
fi

echo "=== endpoint grids (full OO, exact E for iterative + metrics) ==="
submit scripts/trillium_h2o_endpoint_grid.sh "${EXPORT_ARGS[@]}"
submit scripts/trillium_n2_endpoint_grid.sh "${EXPORT_ARGS[@]}"

echo "=== checklist supplement (representative geometries) ==="
submit scripts/trillium_checklist_supplement.sh "${EXPORT_ARGS[@]}"

echo "[ok] submitted. Track with: squeue -u \$USER"
echo "Results:"
echo "  tables/h2o/endpoint_grid.csv"
echo "  tables/n2/endpoint_grid.csv"
echo "  results/h2o_endpoint_grid/  results/n2_endpoint_grid/"
echo "  results/h2o_checklist_supplement/  results/n2_checklist_supplement/"
