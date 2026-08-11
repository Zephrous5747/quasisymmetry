#!/usr/bin/env bash
#SBATCH --job-name=h2o_fci_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=h2o_fci_grid_%A_%a.out
#SBATCH --error=h2o_fci_grid_%A_%a.err
#
# Trillium: H2O bond scan — FCI-referenced select+OO (var/NC) then FCI-referenced K.
# Parallelism = one SLURM array task per (bond × select × cost).
#
# Default grid: OH = linspace(0.958, 2.5, 10)  [Quasi_Symmetries H2O grid]
# Selects: greedy (3+2) and iterative (n_sym=5, m_round=1)
# Costs: NC, variance
#   → 10 × 2 × 2 = 40 tasks (array 0-39)
#
# Each task:
#   1) make_pyscf_hamiltonian.py h2o <bond> --basis sto-3g
#   2) optimize_symmetries.py --reference fci --select … --cost_function …
#   3) metrics.py --backend fci --coupled_energy_method reference --overlap_reference fci
#   4) append row → tables/h2o/fci_oo_metrics_grid.csv
#
# Do NOT pass --mem-per-cpu. Max wall time is 24h.
#
# Submit from repo root:
#   sbatch scripts/h2o/trillium_h2o_fci_oo_metrics_grid.sh
#
# Plot when the array finishes:
#   sbatch --dependency=afterok:<ARRAY_JOBID> \
#     --export=ALL,PLOT_ONLY=1 \
#     scripts/h2o/trillium_h2o_fci_oo_metrics_grid.sh
#
# Optional env:
#   BASIS, HOH_ANGLE, MAXITER, M_ROUND, STATES_PER_SECTOR, DIAG_CSV, PLOT_PNG

set -euo pipefail

export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

BASIS="${BASIS:-sto-3g}"
HOH_ANGLE="${HOH_ANGLE:-104.5}"
MAXITER="${MAXITER:-100}"
M_ROUND="${M_ROUND:-1}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-200}"
DIAG_CSV="${DIAG_CSV:-tables/h2o/fci_oo_metrics_grid.csv}"
PLOT_PNG="${PLOT_PNG:-tables/h2o/fci_oo_metrics_grid_k_sectors_dim.png}"

# 10 OH bond lengths (Å)
mapfile -t BONDS < <(python - <<'PY'
import numpy as np
for x in np.linspace(0.958, 2.5, 10):
    print(f"{x:.10g}")
PY
)

SELECTS=(greedy iterative)
COSTS=(NC variance)

N_BONDS=${#BONDS[@]}
N_SELECTS=${#SELECTS[@]}
N_COSTS=${#COSTS[@]}
N_TOTAL=$((N_BONDS * N_SELECTS * N_COSTS))

mkdir -p "$(dirname "$DIAG_CSV")" "$(dirname "$PLOT_PNG")"

if [[ "${PLOT_ONLY:-0}" == "1" ]]; then
  echo "[plot] $(date -Is) CSV=$DIAG_CSV -> $PLOT_PNG"
  python -u scripts/plot/plot_k_sectors_dim.py "$DIAG_CSV" \
    --output "$PLOT_PNG" \
    --title "H2O FCI-ref OO + K (sto-3g)" \
    --xlabel "OH bond, A"
  exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if (( TASK < 0 || TASK >= N_TOTAL )); then
  echo "[error] task=$TASK out of range 0..$((N_TOTAL - 1))" >&2
  exit 1
fi

BOND_IDX=$((TASK % N_BONDS))
REST=$((TASK / N_BONDS))
COST_IDX=$((REST % N_COSTS))
SELECT_IDX=$((REST / N_COSTS))

BOND="${BONDS[$BOND_IDX]}"
SELECT="${SELECTS[$SELECT_IDX]}"
COST="${COSTS[$COST_IDX]}"

echo "[job] $(date -Is) task=$TASK/$N_TOTAL"
echo "[job] bond=$BOND select=$SELECT cost=$COST basis=$BASIS"
echo "[job] DIAG_CSV=$DIAG_CSV"

python -u scripts/run_fci_oo_metrics_point.py \
  --molecule h2o \
  --bond "$BOND" \
  --select "$SELECT" \
  --cost_function "$COST" \
  --basis "$BASIS" \
  --hoh_angle "$HOH_ANGLE" \
  --n_singles 3 \
  --n_quartets 2 \
  --n_sym 5 \
  --m_round "$M_ROUND" \
  --maxiter "$MAXITER" \
  --states_per_sector "$STATES_PER_SECTOR" \
  --csv "$DIAG_CSV" \
  --out_root results

# Opportunistic plot once enough ok rows exist.
N_OK=$(python -u - <<PY
import csv
from pathlib import Path
p = Path("${DIAG_CSV}")
print(0 if not p.is_file() else sum(1 for r in csv.DictReader(p.open(newline="", encoding="utf-8")) if r.get("status") == "ok"))
PY
)
echo "[job] ok rows so far: $N_OK"
if [[ "$N_OK" -ge 4 ]]; then
  python -u scripts/plot/plot_k_sectors_dim.py "$DIAG_CSV" \
    --output "$PLOT_PNG" \
    --title "H2O FCI-ref OO + K (sto-3g)" \
    --xlabel "OH bond, A" || true
fi

echo "[ok] finished bond=$BOND select=$SELECT cost=$COST at $(date -Is)"
