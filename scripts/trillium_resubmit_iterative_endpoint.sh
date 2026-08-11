#!/usr/bin/env bash
# Resubmit endpoint-grid tasks that hit the iterative post-OO KeyError
# (rounds[0]["optimization"] missing on macro 0).
#
# The bug is in optimize_symmetries.py AFTER select_iterative_pool finishes —
# not in src/iterative_pool.py. Mixed (disjoint/overlap) tasks are unaffected.
#
# Prerequisites:
#   1. Sync the fixed optimize_symmetries.py to the cluster.
#   2. Run from the repo root on Trillium.
#
# Usage:
#   bash scripts/trillium_resubmit_iterative_endpoint.sh
#   DRY_RUN=1 bash scripts/trillium_resubmit_iterative_endpoint.sh
#   ONLY=h2o bash scripts/trillium_resubmit_iterative_endpoint.sh
#   ONLY=n2  bash scripts/trillium_resubmit_iterative_endpoint.sh
#
# Array layout (same as endpoint grids):
#   task = geom_idx * 6 + method_cost_idx
#   method_cost: 0/1 = mixed_disjoint NC/var
#                2/3 = mixed_overlap  NC/var
#                4/5 = iterative      NC/var   ← these only

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-both}"  # h2o | n2 | both

# H2O: 10 geoms → iterative tasks 4,5,10,11,...,58,59
H2O_ITERATIVE=$(python - <<'PY'
print(",".join(str(g * 6 + mc) for g in range(10) for mc in (4, 5)))
PY
)

# N2: 11 geoms → iterative tasks 4,5,...,64,65
N2_ITERATIVE=$(python - <<'PY'
print(",".join(str(g * 6 + mc) for g in range(11) for mc in (4, 5)))
PY
)

submit() {
  local script="$1"
  local array="$2"
  local label="$3"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] sbatch --export=ALL --array=${array} ${script}  # ${label}"
  else
    echo "[submit] ${label}: --array=${array}"
    sbatch --export=ALL --array="${array}" "${script}"
  fi
}

echo "Resubmitting iterative endpoint tasks only (KeyError fix)."
echo "Ensure optimize_symmetries.py with _iterative_cost_before is synced."

if [[ "$ONLY" == "h2o" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_h2o_endpoint_grid.sh "$H2O_ITERATIVE" "h2o iterative (20 tasks)"
fi
if [[ "$ONLY" == "n2" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_n2_endpoint_grid.sh "$N2_ITERATIVE" "n2 iterative (22 tasks)"
fi

echo "[ok] done. Mixed tasks need not be resubmitted for this bug."
