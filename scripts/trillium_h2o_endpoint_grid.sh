#!/usr/bin/env bash
#SBATCH --job-name=h2o_endpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --output=h2o_endpt_%A_%a.out
#SBATCH --error=h2o_endpt_%A_%a.err
#
# H2O STO-3G endpoint grid (FCI ref).
#
# Geometries: OH stretch R in [0.958, 2.5] Å, 10 points, HOH=104.5°.
# Methods × costs (6 per geometry):
#   0 mixed_disjoint + NC
#   1 mixed_disjoint + variance
#   2 mixed_overlap  + NC
#   3 mixed_overlap  + variance
#   4 iterative      + NC
#   5 iterative      + variance
#
# task = geom_idx * 6 + method_cost_idx   (0..59)
#
# Orbital packing via ORBITAL_ROTATION={full,irrep} (default full).
# Results: results/h2o_endpoint_grid/bond_*/U_{full|irrep}/<method>/<cost>/
# Exact E: spatial PG (Q_B1, Q_B2); Clifford also includes Nα/Nβ (document r=4).
# Ham chk with --point_group C2v (required for irrep U).
#
# Submit both packings:
#   ORBITAL_ROTATION=full  sbatch --export=ALL scripts/trillium_h2o_endpoint_grid.sh
#   ORBITAL_ROTATION=irrep sbatch --export=ALL scripts/trillium_h2o_endpoint_grid.sh
# Or: bash scripts/trillium_resubmit_document_exact_U.sh

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
POINT_GROUP="${POINT_GROUP:-C2v}"
ORBITAL_ROTATION="${ORBITAL_ROTATION:-full}"
RESULTS_CSV="${RESULTS_CSV:-tables/h2o/endpoint_grid.csv}"

BONDS=(0.958 1.1293333333 1.3006666667 1.472 1.6433333333 1.8146666667 1.986 2.1573333333 2.3286666667 2.5)
METHODS=(mixed_disjoint mixed_overlap iterative)
COSTS=(NC variance)

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if (( TASK < 0 || TASK > 59 )); then
  echo "[error] task=$TASK out of range 0..59" >&2
  exit 1
fi

GEOM_IDX=$((TASK / 6))
MC_IDX=$((TASK % 6))
METHOD_IDX=$((MC_IDX / 2))
COST_IDX=$((MC_IDX % 2))
BOND="${BONDS[$GEOM_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"
COST="${COSTS[$COST_IDX]}"

EXTRA=()
if [[ -n "${EXACT_PARITY:-}" ]]; then
  EXTRA+=(--exact_parity "$EXACT_PARITY")
fi
if [[ -n "${EXACT_PARITY_EXTRA:-}" ]]; then
  EXTRA+=(--exact_parity_extra "$EXACT_PARITY_EXTRA")
fi
if [[ -n "${EXACT_SECTOR:-}" ]]; then
  EXTRA+=(--exact_sector "$EXACT_SECTOR")
fi

echo "[job] $(date -Is) task=$TASK geom=$GEOM_IDX bond=$BOND method=$METHOD cost=$COST U=$ORBITAL_ROTATION"

ENDPT_EXTRA=()
if [[ -n "${MAX_MACRO}" ]]; then
  ENDPT_EXTRA+=(--max_macroiterations "$MAX_MACRO")
fi
if [[ -n "${STABLE_SPAN}" ]]; then
  ENDPT_EXTRA+=(--stable_span_iters "$STABLE_SPAN")
fi
if [[ -n "${ITERATIVE_REFERENCE:-}" ]]; then
  ENDPT_EXTRA+=(--iterative_reference "$ITERATIVE_REFERENCE")
fi
if [[ -n "${N_SYM:-}" ]]; then
  ENDPT_EXTRA+=(--n_sym "$N_SYM")
fi
if [[ -n "${CAMPAIGN:-}" ]]; then
  ENDPT_EXTRA+=(--campaign "$CAMPAIGN")
fi
if [[ "${STRICT_REF_WEIGHT:-0}" == "1" ]]; then
  ENDPT_EXTRA+=(--strict_reference_weight)
fi

python -u scripts/run_endpoint_point.py \
  --molecule h2o \
  --bond "$BOND" \
  --method "$METHOD" \
  --cost_function "$COST" \
  --basis "$BASIS" \
  --hoh_angle "$HOH_ANGLE" \
  --point_group "$POINT_GROUP" \
  --orbital_rotation "$ORBITAL_ROTATION" \
  --maxiter "$MAXITER" \
  --m_round "$M_ROUND" \
  --states_per_sector "$STATES_PER_SECTOR" \
  --results_csv "$RESULTS_CSV" \
  "${ENDPT_EXTRA[@]}" \
  "${EXTRA[@]}"

echo "[ok] finished task=$TASK at $(date -Is)"
