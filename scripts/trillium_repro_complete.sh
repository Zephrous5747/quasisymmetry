#!/usr/bin/env bash
# Close the remaining Sec.10 reproducibility-record gaps.
#
# After the extractor fix (audit 7), items 1, 2, 7 are complete on existing data
# and 6 is n/a (route A). Three gaps remain, and all three need a RECOMPUTE
# because the artifacts were never written, not because they were mis-parsed:
#
#   item 3  full ranked candidate library + rejection reasons
#           - iterative: already in rounds[i].selection_trace  -> complete
#           - mixed:     src/greedy_selection.py now emits ranked_candidates
#                        (whole scored library, accepted flag) -> needs rerun
#   item 4  intermediate/final F2 spans
#           - iterative: rounds[i].span_key / span_B_key       -> complete
#           - mixed:     new span_trace                        -> needs rerun
#   item 5  optimizer convergence trajectory
#           - rotation / nit / nfev / message / tolerances were already stored;
#             the per-iteration objective trace was not. run_endpoint_point.py
#             now passes --oo_trace_json.                      -> needs rerun
#
# So: mixed_* points need a rerun for 3+4+5, iterative points for 5 only.
# The physics is unchanged; only the record grows. Results are written to a
# separate campaign tag so the validated numbers are not overwritten.
#
#SBATCH --job-name=qs_repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-125
#SBATCH --output=qs_repro_%A_%a.out
#SBATCH --error=qs_repro_%A_%a.err
#
#   DRY_RUN=1 bash scripts/trillium_repro_complete.sh     # list the work
#   bash scripts/trillium_repro_complete.sh               # submit both arrays
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

export ORBITAL_ROTATION="${ORBITAL_ROTATION:-irrep}"   # the admissible packing
export MAXITER="${MAXITER:-100}"
export MAX_MACRO="${MAX_MACRO:-20}"
export STABLE_SPAN="${STABLE_SPAN:-2}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"
export STRICT_REF_WEIGHT="${STRICT_REF_WEIGHT:-1}"
export CAMPAIGN="${CAMPAIGN:-repro_$(date +%Y%m%d)}"
unset EXACT_PARITY EXACT_PARITY_EXTRA EXACT_SECTOR || true

echo "[preflight] the record-emitting code must be present"
python3 - <<'PY'
import inspect
from src import greedy_selection
src = inspect.getsource(greedy_selection)
assert "ranked_candidates" in src, "greedy_selection.py STALE -- re-sync"
assert "span_trace" in src, "greedy_selection.py STALE -- re-sync"
rep = open("scripts/run_endpoint_point.py").read()
assert "--oo_trace_json" in rep, "run_endpoint_point.py STALE -- re-sync"
print("[preflight] OK")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] would resubmit all 126 points under U=$ORBITAL_ROTATION"
  echo "[dry-run] campaign=$CAMPAIGN"
  echo "[dry-run]   n2  : sbatch --array=0-65 scripts/trillium_n2_endpoint_grid.sh"
  echo "[dry-run]   h2o : sbatch --array=0-59 scripts/trillium_h2o_endpoint_grid.sh"
  exit 0
fi

sbatch --export=ALL --array=0-65 scripts/trillium_n2_endpoint_grid.sh
sbatch --export=ALL --array=0-59 scripts/trillium_h2o_endpoint_grid.sh

cat <<'EOF'

[ok] submitted. When it finishes:

  U=irrep bash scripts/audit/audit_06_verify_rerun.sh | tail -20   # physics unchanged
  U=irrep bash scripts/audit/audit_07_repro_record.sh | head -20   # items 1-5,7 complete

Expect audit 6 to reproduce the same K values (nothing about the search changed)
and audit 7 to show 126/126 on items 1,2,3,4,5,7.

Still NOT closed by this run -- these need implementation, not a rerun:

  * Sec.4 exhaustive quotient-pool check (1023 / 32767 candidates rescored after
    every orbital optimisation, compared against the polynomial fixed point per
    Sec.4.3). Without it Sec.7.1 applies: the polynomial pool is an
    initialisation-dependent restricted-neighbourhood heuristic, so the
    iterative-vs-mixed ranking is a statement about this neighbourhood only.
  * Sec.7.3(4) boundary reranking with a more accurate state.
  * Sec.3.2/7.1 multiple initial frames and orbital starts.
EOF
