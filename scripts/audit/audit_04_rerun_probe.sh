#!/usr/bin/env bash
# AUDIT 4 — controlled reruns that isolate each suspected cause of K saturation.
# Submit as a small SLURM array (12 short jobs) or run one arm interactively.
#
#SBATCH --job-name=qs_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=qs_audit_%A_%a.out
#SBATCH --error=qs_audit_%A_%a.err
#
# Arms (per molecule, at one compact and one saturated geometry):
#   A  baseline        : reproduce the campaign exactly            -> expect K=261/3584
#   B  +EXACT_SECTOR   : pin the sector that holds the HF/FCI state
#   C  irrep U only    : point-group-preserving rotation
#   D  B + C           : both fixes together
#
# Interactive single arm:
#   ARM=D MOLECULE=n2 BOND=1.8 bash scripts/audit/audit_04_rerun_probe.sh
#
# Array:
#   sbatch --export=ALL scripts/audit/audit_04_rerun_probe.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

# H2O: r=4 -> sector label length 4 (Na,Nb,Q_B1,Q_B2); N2: r=5.
# All-(+1) = all zero bits; spin bits come from (Na mod 2, Nb mod 2) = (1,1).
SECTOR_H2O="${SECTOR_H2O:-1,1,0,0}"
SECTOR_N2="${SECTOR_N2:-1,1,0,0,0}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  T="$SLURM_ARRAY_TASK_ID"
  MOLS=(h2o n2); BONDS_H2O=(0.958 1.8146666667); BONDS_N2=(1.2 1.8)
  ARMS=(A B C D)
  MOLECULE="${MOLS[$(( (T / 4) % 2 ))]}"
  ARM="${ARMS[$(( T % 4 ))]}"
  IDX=$(( T / 8 ))
  if [[ "$MOLECULE" == "h2o" ]]; then BOND="${BONDS_H2O[$IDX]}"; else BOND="${BONDS_N2[$IDX]}"; fi
else
  MOLECULE="${MOLECULE:?set MOLECULE=h2o|n2}"
  BOND="${BOND:?set BOND}"
  ARM="${ARM:-A}"
fi

if [[ "$MOLECULE" == "h2o" ]]; then
  EXACT="exact/h2o_norb7_sto3g_exact.txt"; PG=C2v; SECTOR="$SECTOR_H2O"
else
  EXACT="exact/n2_norb10_sto3g_exact.txt"; PG=D2h; SECTOR="$SECTOR_N2"
fi
python -u scripts/write_default_exact_parity.py --molecule "$MOLECULE" -o "$EXACT"

U=full
EXTRA=()
case "$ARM" in
  A) ;;
  B) EXTRA+=(--exact_sector "$SECTOR") ;;
  C) U=irrep ;;
  D) U=irrep; EXTRA+=(--exact_sector "$SECTOR") ;;
  *) echo "unknown ARM=$ARM" >&2; exit 1 ;;
esac

echo "[audit4] molecule=$MOLECULE bond=$BOND arm=$ARM U=$U sector=${EXTRA[*]:-none}"

for METHOD in iterative mixed_disjoint; do
  python -u scripts/run_endpoint_point.py \
    --molecule "$MOLECULE" \
    --bond "$BOND" \
    --method "$METHOD" \
    --cost_function NC \
    --basis sto-3g \
    --point_group "$PG" \
    --orbital_rotation "$U" \
    --maxiter "${MAXITER:-100}" \
    --m_round 1 \
    --max_macroiterations "${MAX_MACRO:-20}" \
    --stable_span_iters "${STABLE_SPAN:-2}" \
    --iterative_reference fci_rotate \
    --states_per_sector "${STATES_PER_SECTOR:-500}" \
    --results_csv "tables/audit/arm_${ARM}.csv" \
    "${EXTRA[@]}"
done

echo "[audit4] done. Compare K / sectors / dim / converged across arms:"
echo "  column -s, -t tables/audit/arm_*.csv"
