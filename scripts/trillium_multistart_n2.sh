#!/usr/bin/env bash
#SBATCH --job-name=qs_mstart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-32
#SBATCH --output=qs_mstart_%A_%a.out
#SBATCH --error=qs_mstart_%A_%a.err
#
# MULTI-START ROBUSTNESS — the critical path for the Q4(ii) claim.
#
# The report states that the quota rule reaches a lower K than the exhaustive
# rule more often than a higher one (mean dK = -0.27 at M=n-r_sp, -0.45 at
# M=n-r_sp-1), and immediately qualifies it: one orbital start was run, so no
# scatter estimate exists and the difference cannot be called significant.
#
# This measures that scatter. For each (geometry, arm) the POOL IS HELD FIXED
# at the one the campaign selected, and only the orbital start x0 is varied:
# identity plus 4 random starts. The spread of K at fixed pool is the noise
# floor against which dK must be read.
#
# Scope: N2 only, NC only, M = n-r_sp-1 (m35_grid) -- where the effect is
# largest. 11 geometries x 3 arms = 33 tasks, 5 starts each.
#
#   DRY_RUN=1 bash scripts/trillium_multistart_n2.sh   # list the 33 inputs
#   sbatch --export=ALL scripts/trillium_multistart_n2.sh
#
# Afterwards:  bash scripts/audit/audit_10_multistart.sh

set -euo pipefail
export TRILLIUM=1
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

SUFFIX="${SUFFIX:-m35_grid}"
COST="${COST:-NC}"
N_STARTS="${N_STARTS:-5}"
SCALE="${SCALE:-0.3}"
SEED="${SEED:-0}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-500}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t POINTS < <(ls -d results/n2_${SUFFIX}/bond_*/U_irrep/*/${COST} 2>/dev/null | sort)
NTASK=${#POINTS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "=== multi-start robustness ==="
  echo "campaign : results/n2_${SUFFIX}  cost=${COST}"
  echo "points   : $NTASK   starts each: $N_STARTS (identity + $((N_STARTS-1)) random, scale $SCALE)"
  echo "pool     : FIXED per point; only x0 varies"
  printf '  %s\n' "${POINTS[@]:0:4}"; echo "  ..."
  if (( NTASK == 0 )); then echo "[error] no points found" >&2; exit 1; fi
  [[ "${DRY_RUN:-0}" == "1" ]] && { echo; echo "[dry-run] sbatch --array=0-$((NTASK-1)) $0"; exit 0; }
  sbatch --export=ALL --array=0-$((NTASK-1)) "$0"
  echo "when finished:  bash scripts/audit/audit_10_multistart.sh"
  exit 0
fi

T="${SLURM_ARRAY_TASK_ID}"
(( T >= NTASK )) && { echo "[skip] task $T of $NTASK"; exit 0; }
D="${POINTS[$T]}"

# Pool and geometry are taken from the completed run, so the discrete
# selection is identical across starts by construction.
read -r BOND PARITY CHK <<<"$(python3 - "$D" <<'PY'
import sys, re
from pathlib import Path
sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo
d = Path(sys.argv[1])
oo = load_oo(d / "oo.json")
bond = float(re.sub(r"^bond_", "", d.parts[1]).replace("p", "."))
parity = oo.get("parity") or oo.get("parity_output")
print(bond, parity, oo["molpath"])
PY
)"

OUT="results/n2_multistart/${SUFFIX}_${COST}/$(basename $(dirname "$D"))_$(basename "$D")"
mkdir -p "$OUT"
echo "[mstart] task=$T $D  bond=$BOND  starts=$N_STARTS"

python -u scripts/run_multistart_oo_point.py \
  --chk "$CHK" \
  --parity "$PARITY" \
  --molecule n2 \
  --bond "$BOND" \
  --cost_function "$COST" \
  --n_starts "$N_STARTS" \
  --seed "$SEED" \
  --scale "$SCALE" \
  --states_per_sector "$STATES_PER_SECTOR" \
  --out_dir "$OUT"

echo "[mstart] finished task=$T at $(date -Is)"
