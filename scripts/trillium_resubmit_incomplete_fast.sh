#!/usr/bin/env bash
# Resubmit unfinished N2 iterative endpoint + N2 checklist-supplement tasks
# with FCI-rotate OO (same cost path as mixed) and caps for <<24h wall.
#
# Incomplete today (synced local tables/logs):
#   - N2 endpoint iterative: all 22 tasks (geom×{NC,variance}), no oo.json
#   - N2 checklist: tasks 6-11 (R=1.2/1.8/2.2 × NC/variance) — mixed ok,
#     iterative_* rows missing from tables/n2/checklist_supplement_manifest.csv
#   - H2O checklist: complete in manifest (skip by default)
#
# Caps / knobs (override via env):
#   MAXITER=30                 L-BFGS iters per OO (was 100)
#   MAX_MACRO=4                discrete frame updates (was max(2*norb,8)=20 for N2)
#   STABLE_SPAN=1              stop after one all-five success (was 2)
#   ITERATIVE_REFERENCE=fci_rotate   ranking ref (OO always fci_rotate now)
#   STATES_PER_SECTOR=100
#   SKIP_CHECKPOINTS=1 SKIP_CROSS=1 SKIP_CLIFFORD=1 ONLY_ITERATIVE=1  (checklist)
#
# Prerequisites — cancel old stuck jobs first, then sync code to Trillium:
#   scancel -u $USER   # or selective: scancel <jobid>
#   rsync -avz \
#     optimize_symmetries.py \
#     src/workflow_cli.py \
#     scripts/run_endpoint_point.py \
#     scripts/run_checklist_supplement_point.py \
#     scripts/trillium_n2_endpoint_grid.sh \
#     scripts/trillium_checklist_supplement.sh \
#     scripts/trillium_resubmit_incomplete_fast.sh \
#     USER@trillium.scinet.utoronto.ca:/scratch/zephrous/quasisymmetry/
#   # note: put .py at repo root / src/, scripts under scripts/
#
# Clean old Slurm logs (repo root on Trillium) before resubmit:
#   rm -f n2_endpt_*.{out,err} checklist_supp_*.{out,err}
#
# Usage (on Trillium, repo root):
#   bash scripts/trillium_resubmit_incomplete_fast.sh
#   DRY_RUN=1 bash scripts/trillium_resubmit_incomplete_fast.sh
#   ONLY=endpoint bash scripts/trillium_resubmit_incomplete_fast.sh
#   ONLY=checklist bash scripts/trillium_resubmit_incomplete_fast.sh
#
# Do NOT pass --mem-per-cpu. Max wall is still 24h (jobs should finish much sooner).

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-both}"  # endpoint | checklist | both

# Fast defaults for <<24h N2 FCI iterative (fci_rotate OO like mixed)
export MAXITER="${MAXITER:-30}"
export MAX_MACRO="${MAX_MACRO:-4}"
export STABLE_SPAN="${STABLE_SPAN:-1}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-100}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"

# Checklist: only missing iterative protocol; skip extra OO/metrics work
export ONLY_ITERATIVE="${ONLY_ITERATIVE:-1}"
export SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-1}"
export SKIP_CROSS="${SKIP_CROSS:-1}"
export SKIP_CLIFFORD="${SKIP_CLIFFORD:-1}"

# N2 endpoint: task = geom*6 + {4=NC iterative, 5=variance iterative}, geom 0..10
N2_ENDPT_ITERATIVE=$(python - <<'PY'
print(",".join(str(g * 6 + mc) for g in range(11) for mc in (4, 5)))
PY
)

# Checklist array: geom_idx*2 + cost; N2 geoms are 3,4,5 → tasks 6..11
N2_CHECKLIST_INCOMPLETE="6,7,8,9,10,11"

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

echo "=== fast resubmit of incomplete iterative work ==="
echo "MAXITER=$MAXITER MAX_MACRO=$MAX_MACRO STABLE_SPAN=$STABLE_SPAN"
echo "STATES_PER_SECTOR=$STATES_PER_SECTOR ITERATIVE_REFERENCE=$ITERATIVE_REFERENCE"
echo "checklist: ONLY_ITERATIVE=$ONLY_ITERATIVE SKIP_CHECKPOINTS=$SKIP_CHECKPOINTS SKIP_CROSS=$SKIP_CROSS SKIP_CLIFFORD=$SKIP_CLIFFORD"
echo "Ensure optimize_symmetries.py (TrackingObjective + fci_rotate OO) is synced."

if [[ "$ONLY" == "endpoint" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_n2_endpoint_grid.sh \
    "$N2_ENDPT_ITERATIVE" \
    "n2 endpoint iterative NC/var (22 tasks)"
fi

if [[ "$ONLY" == "checklist" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_checklist_supplement.sh \
    "$N2_CHECKLIST_INCOMPLETE" \
    "n2 checklist iterative-only (tasks 6-11)"
fi

echo "[ok] submitted. Monitor with: squeue -u \$USER"
echo "Expect each OO step ~tens of seconds (mixed-like), not hours (exact_taper)."
