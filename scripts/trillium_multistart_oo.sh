#!/usr/bin/env bash
#SBATCH --job-name=qs_mstart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=multistart_%A_%a.out
#SBATCH --error=multistart_%A_%a.err
#
# Multi-start OO on N2 anomaly bonds (1.2, 1.8) x {NC,variance} x
# {mixed_disjoint, iterative} fixed parity from checklist supplement.
#
# task = ((geom*2 + cost)*2 + protocol)  as in trillium_truncation_sweep.sh
#
# Env: N_STARTS (default 5), MAXITER (default 200), SEED (default 0)
# Do NOT pass --mem-per-cpu.
#
# Sync TO: scripts/run_multistart_oo_point.py, this script, optimize_symmetries.py, metrics.py, ...
# Sync FROM: results/n2_multistart_oo/ tables/n2/multistart_oo_manifest.csv

set -euo pipefail
export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?}"
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
CHK="hamiltonians/N2_bond$(printf '%.4f' "$BOND")sto-3g.chk"

PARITY=""
for c in \
  "results/n2_checklist_supplement/bond_${BOND_TAG}/${COST}/${PROTO}_${COST}_parity.txt" \
  "results/n2_checklist_supplement/bond_${BOND_TAG}/${COST}/${PROTO}_${COST}_oo.json"
do
  if [[ -f "$c" ]]; then
    if [[ "$c" == *.json ]]; then
      # Extract parity_output path from OO JSON if parity txt missing.
      PARITY=$(python - <<PY
import json
from json import JSONDecoder
from pathlib import Path
raw=Path("$c").read_text()
dec=JSONDecoder(); objs=[]; i=0
while i < len(raw):
  while i < len(raw) and raw[i].isspace(): i+=1
  if i>=len(raw) or raw[i]!='{': break
  o,e=dec.raw_decode(raw,i); objs.append(o); i=e
oo=objs[-1]
print(oo.get("parity_output") or oo.get("parity") or "")
PY
)
    else
      PARITY="$c"
    fi
    break
  fi
done

if [[ -z "${PARITY}" || ! -f "$PARITY" ]]; then
  echo "[error] parity not found for bond=$BOND cost=$COST proto=$PROTO" >&2
  exit 1
fi
if [[ ! -f "$CHK" ]]; then
  echo "[warn] missing chk $CHK — make_pyscf_hamiltonian will be needed upstream" >&2
fi

N_STARTS="${N_STARTS:-5}"
MAXITER="${MAXITER:-200}"
SEED="${SEED:-0}"
STATES="${STATES_PER_SECTOR:-500}"

echo "[job] task=$TASK bond=$BOND cost=$COST proto=$PROTO parity=$PARITY n_starts=$N_STARTS"

python -u scripts/run_multistart_oo_point.py \
  --chk "$CHK" \
  --parity "$PARITY" \
  --molecule n2 \
  --bond "$BOND" \
  --cost_function "$COST" \
  --protocol "$PROTO" \
  --n_starts "$N_STARTS" \
  --maxiter "$MAXITER" \
  --seed "$SEED" \
  --states_per_sector "$STATES"

echo "[ok] finished task=$TASK"
