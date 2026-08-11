#!/usr/bin/env bash
# Resubmit N2 iterative endpoint + checklist with mid-range OO caps.
#
# Defaults (override via env):
#   MAXITER=50          L-BFGS iters per OO call  (was 30 on fast run)
#   MAX_MACRO=6         discrete frame updates   (was 4)
#   STABLE_SPAN=1
#   ITERATIVE_REFERENCE=fci_rotate
#   STATES_PER_SECTOR=100
#
# Wall-time estimate (from last fast run, fci_rotate OO):
#   NC OO alone was ~0.8–0.9 h/job at (30, 4); variance << 0.1 h.
#   Scale ≈ (50/30)×(6/4) = 2.5 → NC OO ~2.0–2.3 h, full job ~5–8 h
#   if most rounds still hit maxiter. Should finish within ~10 h; wall is 24 h.
#   Variance jobs stay well under 1 h.
#
# Usage (Trillium, repo root):
#   bash scripts/trillium_resubmit_n2_iterative_mid.sh
#   DRY_RUN=1 bash scripts/trillium_resubmit_n2_iterative_mid.sh
#   ONLY=endpoint bash scripts/trillium_resubmit_n2_iterative_mid.sh
#   ONLY=checklist bash scripts/trillium_resubmit_n2_iterative_mid.sh
#   ONLY_NC=1 bash scripts/trillium_resubmit_n2_iterative_mid.sh   # NC tasks only
#
# Sync this script (and trillium_*.sh / run_*_point.py if not already) first.
# Optional cleanup of prior Slurm logs:
#   rm -f n2_endpt_*.{out,err} checklist_supp_*.{out,err}

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-both}"       # endpoint | checklist | both
ONLY_NC="${ONLY_NC:-0}"    # 1 = NC iterative only (skip variance)

export MAXITER="${MAXITER:-50}"
export MAX_MACRO="${MAX_MACRO:-6}"
export STABLE_SPAN="${STABLE_SPAN:-1}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-100}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"

export ONLY_ITERATIVE="${ONLY_ITERATIVE:-1}"
export SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-1}"
export SKIP_CROSS="${SKIP_CROSS:-1}"
export SKIP_CLIFFORD="${SKIP_CLIFFORD:-1}"

if [[ "$ONLY_NC" == "1" ]]; then
  # geom*6 + 4 = NC iterative only
  N2_ENDPT_ITERATIVE=$(python - <<'PY'
print(",".join(str(g * 6 + 4) for g in range(11)))
PY
)
  # checklist: geom_idx*2 + cost; N2 geoms 3,4,5 → tasks 6,8,10 are NC
  N2_CHECKLIST_ITERATIVE="6,8,10"
  LABEL_EXTRA=" NC-only"
else
  N2_ENDPT_ITERATIVE=$(python - <<'PY'
print(",".join(str(g * 6 + mc) for g in range(11) for mc in (4, 5)))
PY
)
  N2_CHECKLIST_ITERATIVE="6,7,8,9,10,11"
  LABEL_EXTRA=""
fi

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

echo "=== N2 iterative mid resubmit (MAXITER=$MAXITER MAX_MACRO=$MAX_MACRO) ==="
echo "STABLE_SPAN=$STABLE_SPAN STATES_PER_SECTOR=$STATES_PER_SECTOR"
echo "ITERATIVE_REFERENCE=$ITERATIVE_REFERENCE ONLY_NC=$ONLY_NC"
echo "checklist: ONLY_ITERATIVE=$ONLY_ITERATIVE SKIP_CHECKPOINTS=$SKIP_CHECKPOINTS"
echo "Expect worst NC jobs ~5–8 h (under 10 h); variance << 1 h. Slurm wall=24 h."

if [[ "$ONLY" == "endpoint" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_n2_endpoint_grid.sh \
    "$N2_ENDPT_ITERATIVE" \
    "n2 endpoint iterative${LABEL_EXTRA}"
fi

if [[ "$ONLY" == "checklist" || "$ONLY" == "both" ]]; then
  submit scripts/trillium_checklist_supplement.sh \
    "$N2_CHECKLIST_ITERATIVE" \
    "n2 checklist iterative-only${LABEL_EXTRA}"
fi

echo "[ok] submitted. Monitor: squeue -u \$USER"
