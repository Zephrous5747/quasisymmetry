#!/usr/bin/env bash
# Resubmit N2 + H2O endpoint (all methods) and checklist with mid OO caps
# and STO-3G report exact spatial parities (PG rows, Mixed independent of E).
#
# Defaults:
#   MAXITER=50
#   MAX_MACRO=6
#   STABLE_SPAN=1
#   ITERATIVE_REFERENCE=fci_rotate
#   STATES_PER_SECTOR=100
#
# Wall estimate (from prior N2 iterative NC @ 30×4 ≈ 0.8–0.9 h OO):
#   scale (50/30)×(6/4)=2.5 → NC OO ~2–2.3 h, full job ~5–8 h (<10 h).
#   Mixed / variance much faster. Slurm wall remains 24 h.
#
# Sync before submit (repo root on Trillium):
#   rsync -avz \
#     optimize_symmetries.py metrics.py \
#     src/greedy_selection.py src/sto3g_exact_symmetries.py src/exact_parity.py \
#     src/workflow_cli.py \
#     scripts/write_default_exact_parity.py \
#     scripts/run_endpoint_point.py \
#     scripts/run_checklist_supplement_point.py \
#     scripts/trillium_n2_endpoint_grid.sh \
#     scripts/trillium_h2o_endpoint_grid.sh \
#     scripts/trillium_checklist_supplement.sh \
#     scripts/trillium_resubmit_sto3g_exact_mid.sh \
#     USER@trillium:/scratch/zephrous/quasisymmetry/
#
# Usage:
#   bash scripts/trillium_resubmit_sto3g_exact_mid.sh
#   DRY_RUN=1 bash scripts/trillium_resubmit_sto3g_exact_mid.sh
#   ONLY=n2|h2o|checklist|endpoint bash ...
#   METHODS=iterative|mixed|all  (default all)

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"          # all | n2 | h2o | endpoint | checklist
METHODS="${METHODS:-all}"    # all | iterative | mixed

export MAXITER="${MAXITER:-50}"
export MAX_MACRO="${MAX_MACRO:-6}"
export STABLE_SPAN="${STABLE_SPAN:-1}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-100}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"

# Checklist: run all three protocols unless ONLY_ITERATIVE=1
export ONLY_ITERATIVE="${ONLY_ITERATIVE:-0}"
export SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-1}"
export SKIP_CROSS="${SKIP_CROSS:-1}"
export SKIP_CLIFFORD="${SKIP_CLIFFORD:-1}"

# Pre-write exact matrices (idempotent)
python -u scripts/write_default_exact_parity.py --molecule h2o \
  -o exact/h2o_norb7_sto3g_exact.txt
python -u scripts/write_default_exact_parity.py --molecule n2 \
  -o exact/n2_norb10_sto3g_exact.txt
export EXACT_PARITY_H2O="${EXACT_PARITY_H2O:-exact/h2o_norb7_sto3g_exact.txt}"
export EXACT_PARITY_N2="${EXACT_PARITY_N2:-exact/n2_norb10_sto3g_exact.txt}"

# Endpoint arrays: task = geom*6 + method_cost
# method_cost: 0 md+NC, 1 md+var, 2 mo+NC, 3 mo+var, 4 it+NC, 5 it+var
n2_endpt_array() {
  python - <<'PY'
import os
methods = os.environ.get("METHODS", "all")
idxs = []
for g in range(11):  # N2 11 bonds
    for mc in range(6):
        kind = mc // 2  # 0 md, 1 mo, 2 it
        if methods == "iterative" and kind != 2:
            continue
        if methods == "mixed" and kind == 2:
            continue
        idxs.append(g * 6 + mc)
print(",".join(map(str, idxs)))
PY
}

h2o_endpt_array() {
  python - <<'PY'
import os
methods = os.environ.get("METHODS", "all")
idxs = []
for g in range(10):  # H2O 10 bonds
    for mc in range(6):
        kind = mc // 2
        if methods == "iterative" and kind != 2:
            continue
        if methods == "mixed" and kind == 2:
            continue
        idxs.append(g * 6 + mc)
print(",".join(map(str, idxs)))
PY
}

# Checklist: H2O geoms 0..2 → tasks 0..5; N2 geoms 3..5 → tasks 6..11
checklist_array() {
  if [[ "${ONLY_ITERATIVE}" == "1" ]]; then
    echo "0,1,2,3,4,5,6,7,8,9,10,11"
  else
    echo "0,1,2,3,4,5,6,7,8,9,10,11"
  fi
}

submit() {
  local script="$1"
  local array="$2"
  local label="$3"
  local exact="${4:-}"
  if [[ -z "$array" ]]; then
    echo "[skip] empty array for ${label}"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] EXACT_PARITY=${exact:-"(script default)"} sbatch --export=ALL --array=${array} ${script}  # ${label}"
  else
    echo "[submit] ${label}: --array=${array}"
    if [[ -n "$exact" ]]; then
      EXACT_PARITY="${exact}" sbatch --export=ALL --array="${array}" "${script}"
    else
      sbatch --export=ALL --array="${array}" "${script}"
    fi
  fi
}

echo "=== sto3g-exact mid resubmit ==="
echo "MAXITER=$MAXITER MAX_MACRO=$MAX_MACRO STABLE_SPAN=$STABLE_SPAN"
echo "METHODS=$METHODS ONLY=$ONLY ITERATIVE_REFERENCE=$ITERATIVE_REFERENCE"
echo "Expect worst NC iterative ~5–8 h (<10 h); mixed/variance much faster."

do_n2_end=0
do_h2o_end=0
do_chk=0
case "$ONLY" in
  all) do_n2_end=1; do_h2o_end=1; do_chk=1 ;;
  n2) do_n2_end=1; do_chk=1 ;;
  h2o) do_h2o_end=1; do_chk=1 ;;
  endpoint) do_n2_end=1; do_h2o_end=1 ;;
  checklist) do_chk=1 ;;
  *) echo "UNKNOWN ONLY=$ONLY"; exit 1 ;;
esac

if [[ "$do_n2_end" == "1" ]]; then
  submit scripts/trillium_n2_endpoint_grid.sh \
    "$(n2_endpt_array)" \
    "n2 endpoint (${METHODS})" \
    "$EXACT_PARITY_N2"
fi

if [[ "$do_h2o_end" == "1" ]]; then
  submit scripts/trillium_h2o_endpoint_grid.sh \
    "$(h2o_endpt_array)" \
    "h2o endpoint (${METHODS})" \
    "$EXACT_PARITY_H2O"
fi

if [[ "$do_chk" == "1" ]]; then
  # checklist script picks molecule from task id; pass both via per-job EXACT_PARITY is hard.
  # Rely on run_checklist_supplement_point default sto3g path (force-written above).
  submit scripts/trillium_checklist_supplement.sh \
    "$(checklist_array)" \
    "checklist H2O+N2" \
    ""
fi

echo "[ok] done. Monitor: squeue -u \$USER"
