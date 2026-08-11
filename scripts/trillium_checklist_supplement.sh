#!/usr/bin/env bash
#SBATCH --job-name=qs_chklist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=checklist_supp_%A_%a.out
#SBATCH --error=checklist_supp_%A_%a.err
#
# Trillium checklist supplement — one job per (geometry × cost_function).
#
# Array layout (12 tasks):
#   task = geom_idx * 2 + cost_idx
#   geom_idx 0..5, cost_idx 0=NC 1=variance
#
#   geom 0: H2O 0.958      equilibrium
#   geom 1: H2O 1.81466667 strongly correlated
#   geom 2: H2O 2.5        dissociative
#   geom 3: N2  1.2        near-equilibrium
#   geom 4: N2  1.8        strongly correlated
#   geom 5: N2  2.2        dissociative
#
# Each task runs for that cost:
#   mixed_disjoint + iterative OO (with selection_trace + oo_trace)
#   pre/post-OO K, OO checkpoints (0/25/50/75/100%), iterative round metrics
#   Clifford sensitivity, cross-objective OO (select=this cost, OO=other cost)
#
# Do NOT pass --mem-per-cpu. Max wall 24h.
#
# ========== SYNC TO SERVER ==========
# From laptop repo root (scripts/ is often gitignored — copy explicitly):
#
#   rsync -avz \
#     src/greedy_selection.py \
#     src/iterative_pool.py \
#     src/workflow_cli.py \
#     optimize_symmetries.py \
#     metrics.py \
#     make_pyscf_hamiltonian.py \
#     external_imports.py \
#     scripts/run_checklist_supplement_point.py \
#     scripts/trillium_checklist_supplement.sh \
#     cluster_tests/_qs_env.sh \
#     USER@trillium.computecanada.ca:/scratch/zephrous/quasisymmetry/
#
# Also ensure matching package layout for:
#   src/  (exact_parity, exact_taper, iterative_pool, clifford_sectors, …)
#   chemistry.py fcidump_openfermion.py as needed by your env
#
# ========== SUBMIT ==========
#   cd /scratch/zephrous/quasisymmetry
#   sbatch scripts/trillium_checklist_supplement.sh
#   # or all campaign jobs:
#   bash scripts/trillium_submit_all.sh
#
# Optional:
#   EXACT_PARITY_EXTRA=exact/h2o_pg.txt   # append PG Z-parity rows into E
#   sbatch --export=ALL,SKIP_CHECKPOINTS=1,SKIP_CLIFFORD=1 scripts/trillium_checklist_supplement.sh
#
# ========== SYNC FROM SERVER ==========
#   results/h2o_checklist_supplement/
#   results/n2_checklist_supplement/
#   tables/h2o/checklist_supplement_manifest.csv
#   tables/n2/checklist_supplement_manifest.csv
#
#   rsync -avz \
#     USER@trillium.computecanada.ca:/scratch/zephrous/quasisymmetry/results/h2o_checklist_supplement \
#     USER@trillium.computecanada.ca:/scratch/zephrous/quasisymmetry/results/n2_checklist_supplement \
#     ./results/
#   rsync -avz \
#     USER@trillium.computecanada.ca:/scratch/zephrous/quasisymmetry/tables/h2o/checklist_supplement_manifest.csv \
#     ./tables/h2o/
#   rsync -avz \
#     USER@trillium.computecanada.ca:/scratch/zephrous/quasisymmetry/tables/n2/checklist_supplement_manifest.csv \
#     ./tables/n2/

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
MAX_MACRO="${MAX_MACRO:-}"
STABLE_SPAN="${STABLE_SPAN:-2}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"

MOLECULES=(h2o h2o h2o n2 n2 n2)
BONDS=(0.958 1.81466667 2.5 1.2 1.8 2.2)
REGIMES=(equilibrium strong dissociative equilibrium strong dissociative)
COSTS=(NC variance)

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if (( TASK < 0 || TASK > 11 )); then
  echo "[error] task=$TASK out of range 0..11" >&2
  exit 1
fi

GEOM_IDX=$((TASK / 2))
COST_IDX=$((TASK % 2))
MOL="${MOLECULES[$GEOM_IDX]}"
BOND="${BONDS[$GEOM_IDX]}"
REGIME="${REGIMES[$GEOM_IDX]}"
COST="${COSTS[$COST_IDX]}"

echo "[job] $(date -Is) task=$TASK/$SLURM_ARRAY_TASK_COUNT geom=$GEOM_IDX molecule=$MOL bond=$BOND regime=$REGIME cost=$COST"

EXTRA=()
if [[ "${SKIP_CHECKPOINTS:-0}" == "1" ]]; then
  EXTRA+=(--skip_checkpoints)
fi
if [[ "${SKIP_CROSS:-0}" == "1" ]]; then
  EXTRA+=(--skip_cross)
fi
if [[ "${SKIP_CLIFFORD:-0}" == "1" ]]; then
  EXTRA+=(--skip_clifford)
fi
if [[ "${ONLY_ITERATIVE:-0}" == "1" ]]; then
  EXTRA+=(--only_iterative)
fi
if [[ -n "${MAX_MACRO:-}" ]]; then
  EXTRA+=(--max_macroiterations "$MAX_MACRO")
fi
if [[ -n "${STABLE_SPAN:-}" ]]; then
  EXTRA+=(--stable_span_iters "$STABLE_SPAN")
fi
if [[ -n "${ITERATIVE_REFERENCE:-}" ]]; then
  EXTRA+=(--iterative_reference "$ITERATIVE_REFERENCE")
fi
if [[ -n "${EXACT_PARITY:-}" ]]; then
  EXTRA+=(--exact_parity "$EXACT_PARITY")
fi
if [[ -n "${EXACT_PARITY_EXTRA:-}" ]]; then
  EXTRA+=(--exact_parity_extra "$EXACT_PARITY_EXTRA")
fi
if [[ -n "${EXACT_SECTOR:-}" ]]; then
  EXTRA+=(--exact_sector "$EXACT_SECTOR")
fi
if [[ -n "${POINT_GROUP:-}" ]]; then
  EXTRA+=(--point_group "$POINT_GROUP")
fi
if [[ -n "${ORBITAL_ROTATION:-}" ]]; then
  EXTRA+=(--orbital_rotation "$ORBITAL_ROTATION")
fi

python -u scripts/run_checklist_supplement_point.py \
  --molecule "$MOL" \
  --bond "$BOND" \
  --cost_function "$COST" \
  --regime "$REGIME" \
  --basis "$BASIS" \
  --hoh_angle "$HOH_ANGLE" \
  --maxiter "$MAXITER" \
  --m_round "$M_ROUND" \
  --states_per_sector "$STATES_PER_SECTOR" \
  "${EXTRA[@]}"

echo "[ok] finished task=$TASK at $(date -Is)"
