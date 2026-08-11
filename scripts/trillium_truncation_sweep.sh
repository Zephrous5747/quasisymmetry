#!/usr/bin/env bash
#SBATCH --job-name=qs_trunc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=trunc_sweep_%A_%a.out
#SBATCH --error=trunc_sweep_%A_%a.err
#
# Truncation sweep on existing checklist / mixed-disjoint OO JSONs.
# Array 0-7: 4 N2 anomaly (geom x cost) x 2 protocols (mixed_disjoint, iterative)
#   geom 0: N2 1.2
#   geom 1: N2 1.8
#   cost 0: NC, cost 1: variance
#   protocol: task%2 == 0 mixed_disjoint else iterative
#
# task = ((geom*2 + cost)*2 + protocol)
#
# Requires prior OO outputs under results/n2_checklist_supplement/ or
# results/n2_fci_mixed_disjoint / results/n2_fci_oo_grid — set OO_ROOT.
#
# Do NOT pass --mem-per-cpu. Max wall 24h.
#
# Sync TO server: scripts/run_truncation_sweep_point.py, this script,
#   metrics.py, src/parity_rank.py, src/fci_rotation_checks.py, src/energy_diagnostics.py
# Sync FROM: results/n2_truncation_sweep/ tables/n2/truncation_sweep_manifest.csv

set -euo pipefail
export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?}"
if (( TASK < 0 || TASK > 7 )); then
  echo "[error] task=$TASK out of range 0..7" >&2
  exit 1
fi

PROTO_IDX=$((TASK % 2))
REST=$((TASK / 2))
COST_IDX=$((REST % 2))
GEOM_IDX=$((REST / 2))

BONDS=(1.2 1.8)
COSTS=(NC variance)
PROTOS=(mixed_disjoint iterative)

BOND="${BONDS[$GEOM_IDX]}"
COST="${COSTS[$COST_IDX]}"
PROTO="${PROTOS[$PROTO_IDX]}"
BOND_TAG=$(printf "%.4f" "$BOND" | tr '.' 'p')

# Prefer checklist supplement OO; fall back to grid paths.
CANDIDATES=(
  "results/n2_checklist_supplement/bond_${BOND_TAG}/${COST}/${PROTO}_${COST}_oo.json"
  "results/n2_fci_mixed_disjoint/bond_${BOND_TAG}/greedy_${COST}_fci.json"
  "results/n2_fci_oo_grid/bond_${BOND_TAG}/iterative_${COST}_fci.json"
)

OO_JSON=""
for c in "${CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then
    OO_JSON="$c"
    break
  fi
done

if [[ -z "$OO_JSON" ]]; then
  echo "[error] no OO JSON found for bond=$BOND cost=$COST proto=$PROTO" >&2
  printf '  tried: %s\n' "${CANDIDATES[@]}" >&2
  exit 1
fi

echo "[job] task=$TASK bond=$BOND cost=$COST proto=$PROTO oo=$OO_JSON"

python -u scripts/run_truncation_sweep_point.py \
  --oo_json "$OO_JSON" \
  --molecule n2 \
  --bond "$BOND" \
  --cost_function "$COST" \
  --protocol "$PROTO"

echo "[ok] finished task=$TASK"
