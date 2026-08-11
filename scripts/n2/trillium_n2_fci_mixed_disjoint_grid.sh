#!/usr/bin/env bash
#SBATCH --job-name=n2_mixed_dj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-21
#SBATCH --output=n2_mixed_dj_%A_%a.out
#SBATCH --error=n2_mixed_dj_%A_%a.err
#
# Trillium: N2 Mixed (greedy quota) only — orbital-disjoint sen+quartets.
# FCI-ref OO (NC/variance) then FCI-overlap chemical-accuracy K.
#
# Grid: N–N = 1.2 … 2.2 step 0.1 (11 points)
# Select: greedy only (4 singles + 3 quartets, disjoint supports)
# Costs: NC, variance
#   → 11 × 2 = 22 tasks (array 0-21)
#
# Requires synced src/greedy_selection.py (senquart_quota_disjoint).
# Do NOT pass --mem-per-cpu. Max wall time is 24h.
#
# Submit from repo root:
#   sbatch scripts/n2/trillium_n2_fci_mixed_disjoint_grid.sh
#
# Plot:
#   sbatch --dependency=afterok:<ARRAY_JOBID> \
#     --export=ALL,PLOT_ONLY=1 \
#     scripts/n2/trillium_n2_fci_mixed_disjoint_grid.sh
#
# Optional: BONDS_CSV=1.1,1.5,2.0,2.5 with matching --array

set -euo pipefail

export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

BASIS="${BASIS:-sto-3g}"
MAXITER="${MAXITER:-100}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-200}"
DIAG_CSV="${DIAG_CSV:-tables/n2/fci_oo_metrics_mixed_disjoint.csv}"
PLOT_PNG="${PLOT_PNG:-tables/n2/fci_oo_metrics_mixed_disjoint_k.png}"
GRID_NAME="${GRID_NAME:-n2_fci_mixed_disjoint}"

if [[ -n "${BONDS_CSV:-}" ]]; then
  IFS=',' read -r -a BONDS <<< "$BONDS_CSV"
else
  mapfile -t BONDS < <(python - <<'PY'
import numpy as np
for x in np.round(np.arange(1.2, 2.21, 0.1), 10):
    print(f"{x:.10g}")
PY
)
fi

SELECT=greedy
COSTS=(NC variance)

N_BONDS=${#BONDS[@]}
N_COSTS=${#COSTS[@]}
N_TOTAL=$((N_BONDS * N_COSTS))

mkdir -p "$(dirname "$DIAG_CSV")" "$(dirname "$PLOT_PNG")"

if [[ "${PLOT_ONLY:-0}" == "1" ]]; then
  echo "[plot] $(date -Is) CSV=$DIAG_CSV -> $PLOT_PNG"
  python -u - <<PY
import ast, csv
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def f(row, k):
    raw = (row.get(k) or "").strip()
    if not raw:
        return None
    try:
        return float(ast.literal_eval(raw))
    except Exception:
        return float(raw)

rows = [r for r in csv.DictReader(Path("${DIAG_CSV}").open(newline="", encoding="utf-8")) if r.get("status") == "ok"]
groups = defaultdict(list)
for r in rows:
    groups[r["cost_function"]].append(r)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
styles = {"NC": ("#009E73", "s", "--"), "variance": ("#D55E00", "s", ":")}
for cost, group in sorted(groups.items()):
    group = sorted(group, key=lambda r: f(r, "bond"))
    color, marker, ls = styles.get(cost, ("#333", "o", "-"))
    ax.plot([f(r, "bond") for r in group], [f(r, "K") for r in group],
            marker=marker, color=color, linestyle=ls, linewidth=1.9, markersize=6.5,
            label=f"Mixed {cost} (disjoint)")
ax.set_xlabel("N--N bond length (A)")
ax.set_ylabel("Chemical-Accuracy K")
ax.set_title("N2/STO-3G, Mixed Pool (orbital-disjoint)")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("${PLOT_PNG}", dpi=220, bbox_inches="tight")
print("[ok] wrote ${PLOT_PNG}")
PY
  exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if (( TASK < 0 || TASK >= N_TOTAL )); then
  echo "[error] task=$TASK out of range 0..$((N_TOTAL - 1)) (N_BONDS=$N_BONDS)" >&2
  exit 1
fi

BOND_IDX=$((TASK % N_BONDS))
COST_IDX=$((TASK / N_BONDS))
BOND="${BONDS[$BOND_IDX]}"
COST="${COSTS[$COST_IDX]}"

echo "[job] $(date -Is) task=$TASK/$N_TOTAL bond=$BOND select=$SELECT cost=$COST"

python -u scripts/run_fci_oo_metrics_point.py \
  --molecule n2 \
  --bond "$BOND" \
  --select "$SELECT" \
  --cost_function "$COST" \
  --basis "$BASIS" \
  --n_singles 4 \
  --n_quartets 3 \
  --n_sym 7 \
  --maxiter "$MAXITER" \
  --states_per_sector "$STATES_PER_SECTOR" \
  --csv "$DIAG_CSV" \
  --out_root results \
  --grid_name "$GRID_NAME"

echo "[ok] finished bond=$BOND cost=$COST at $(date -Is)"
