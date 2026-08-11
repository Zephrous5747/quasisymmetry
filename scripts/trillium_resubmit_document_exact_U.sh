#!/usr/bin/env bash
# Resubmit N2 + H2O endpoint grids with document exact symmetries and both U packings.
#
# Exact (PDF):
#   H2O r=4 = Nα,Nβ + Q_B1,Q_B2   (spatial file + spin in Clifford)
#   N2  r=5 = Nα,Nβ + Q_pix,Q_piy,Q_u
# Orbital rotation: full SO(n) AND irrep-restricted (point-group chk).
# Exhaustive quotient search is NOT enabled (later).
#
# Defaults (mid caps, matching prior mid resubmit):
#   MAXITER=50
#   MAX_MACRO=6
#   STABLE_SPAN=1
#   ITERATIVE_REFERENCE=fci_rotate
#   STATES_PER_SECTOR=100
#   ORBITAL_ROTATIONS=full,irrep
#
# Sync before submit (repo root on Trillium):
#   rsync -avz \
#     optimize_symmetries.py metrics.py make_pyscf_hamiltonian.py \
#     src/greedy_selection.py src/iterative_pool.py \
#     src/sto3g_exact_symmetries.py src/exact_parity.py src/exact_taper.py \
#     src/clifford_sectors.py src/orbital_rotation.py src/pyscf_chk.py \
#     src/workflow_cli.py \
#     scripts/write_default_exact_parity.py \
#     scripts/run_endpoint_point.py \
#     scripts/run_checklist_supplement_point.py \
#     scripts/trillium_n2_endpoint_grid.sh \
#     scripts/trillium_h2o_endpoint_grid.sh \
#     scripts/trillium_checklist_supplement.sh \
#     scripts/trillium_resubmit_document_exact_U.sh \
#     USER@trillium:/scratch/zephrous/quasisymmetry/
#
# Usage:
#   bash scripts/trillium_resubmit_document_exact_U.sh
#   DRY_RUN=1 bash scripts/trillium_resubmit_document_exact_U.sh
#   ONLY=n2|h2o|endpoint|checklist bash ...
#   METHODS=iterative|mixed|all
#   ORBITAL_ROTATIONS=full          # or irrep, or full,irrep
#   MAXITER=100 MAX_MACRO=8 bash ...

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# Load modules + .venv before any local python (exact-matrix write, array helpers).
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"          # all | n2 | h2o | endpoint | checklist
METHODS="${METHODS:-all}"    # all | iterative | mixed
ORBITAL_ROTATIONS="${ORBITAL_ROTATIONS:-full,irrep}"

export MAXITER="${MAXITER:-50}"
export MAX_MACRO="${MAX_MACRO:-6}"
export STABLE_SPAN="${STABLE_SPAN:-1}"
export STATES_PER_SECTOR="${STATES_PER_SECTOR:-100}"
export M_ROUND="${M_ROUND:-1}"
export ITERATIVE_REFERENCE="${ITERATIVE_REFERENCE:-fci_rotate}"

export ONLY_ITERATIVE="${ONLY_ITERATIVE:-0}"
export SKIP_CHECKPOINTS="${SKIP_CHECKPOINTS:-1}"
export SKIP_CROSS="${SKIP_CROSS:-1}"
export SKIP_CLIFFORD="${SKIP_CLIFFORD:-1}"

# Pre-write spatial exact matrices (spin number added in Clifford)
python -u scripts/write_default_exact_parity.py --molecule h2o \
  -o exact/h2o_norb7_sto3g_exact.txt
python -u scripts/write_default_exact_parity.py --molecule n2 \
  -o exact/n2_norb10_sto3g_exact.txt
export EXACT_PARITY_H2O="${EXACT_PARITY_H2O:-exact/h2o_norb7_sto3g_exact.txt}"
export EXACT_PARITY_N2="${EXACT_PARITY_N2:-exact/n2_norb10_sto3g_exact.txt}"

# Serial prebuild of PG chks so irrep array tasks never race on HDF5 locks.
# Set PREBUILD_PG_CHKS=0 to skip (only if chks already exist and are good).
PREBUILD_PG_CHKS="${PREBUILD_PG_CHKS:-1}"
prebuild_pg_chks() {
  local need_irrep=0
  local u
  for u in "${U_MODES[@]}"; do
    u="$(echo "$u" | tr -d '[:space:]')"
    [[ "$u" == "irrep" ]] && need_irrep=1
  done
  if [[ "$need_irrep" != "1" || "$PREBUILD_PG_CHKS" != "1" ]]; then
    return 0
  fi
  echo "[prebuild] serial PG hamiltonians for irrep U ..."
  # Stale array tasks (old code) overwrite PG chks without sidecars — cancel first.
  if [[ "${SCANCEL_STALE_BEFORE_PREBUILD:-1}" == "1" ]]; then
    echo "[prebuild] scancel stale n2_endpt/h2o_endpt/qs_chklist for $USER"
    scancel -u "$USER" --name=n2_endpt 2>/dev/null || true
    scancel -u "$USER" --name=h2o_endpt 2>/dev/null || true
    scancel -u "$USER" --name=qs_chklist 2>/dev/null || true
    sleep 3
  fi
  mkdir -p hamiltonians results/_endpoint_ham
  # Rebuild if missing OR if irreps/txt sidecar cannot be loaded.
  _pg_chk_ok() {
    local chk="$1"
    python - <<PY
from src.orbital_rotation import load_orbital_irreps, _orbsym_txt_path, _orbsym_sidecar_path
from pathlib import Path
import sys
chk = Path(r"""$chk""")
labs = load_orbital_irreps(chk)
txt = _orbsym_txt_path(chk)
npy = _orbsym_sidecar_path(chk)
if labs is None:
    print(f"[prebuild] LOAD_FAIL {chk} txt={txt.is_file()} npy={npy.is_file()}", flush=True)
    sys.exit(1)
if not txt.is_file():
    print(f"[prebuild] LOAD_FAIL {chk.name} irreps ok but missing {txt.name}", flush=True)
    sys.exit(1)
print(f"[prebuild] ok {chk.name} irreps={labs.tolist()} txt={txt.is_file()} npy={npy.is_file()}", flush=True)
sys.exit(0)
PY
  }
  local b rc
  if [[ "${do_n2_end:-0}" == "1" || "${do_chk:-0}" == "1" ]]; then
    for b in 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2; do
      local out="hamiltonians/N2_bond$(printf '%.4f' "$b")sto-3g_D2h.chk"
      if [[ -f "$out" ]] && _pg_chk_ok "$out"; then
        continue
      fi
      echo "[prebuild] building $out"
      rm -f "$out" "${out}.lock" "${out}.tmp."* "${out}.orbsym.npy" "${out}.orbsym.txt"
      python -u make_pyscf_hamiltonian.py n2 "$b" --basis sto-3g --point_group D2h \
        >"results/_endpoint_ham/$(basename "$out" .chk).log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "[prebuild] FAIL make_pyscf exit=$rc for $out (see log)" >&2
        tail -30 "results/_endpoint_ham/$(basename "$out" .chk).log" >&2 || true
        return 1
      fi
      _pg_chk_ok "$out" || {
        echo "[prebuild] FAIL irreps in $out" >&2
        tail -30 "results/_endpoint_ham/$(basename "$out" .chk).log" >&2 || true
        return 1
      }
    done
  fi
  if [[ "${do_h2o_end:-0}" == "1" || "${do_chk:-0}" == "1" ]]; then
    for b in 0.958 1.1293333333 1.3006666667 1.472 1.6433333333 1.8146666667 1.986 2.1573333333 2.3286666667 2.5; do
      local out="hamiltonians/H2O_OH$(printf '%.4f' "$b")_104.5000sto-3g_C2v.chk"
      if [[ -f "$out" ]] && _pg_chk_ok "$out"; then
        continue
      fi
      echo "[prebuild] building $out"
      rm -f "$out" "${out}.lock" "${out}.tmp."* "${out}.orbsym.npy" "${out}.orbsym.txt"
      python -u make_pyscf_hamiltonian.py h2o "$b" --basis sto-3g \
        --mol_parameter_2 104.5 --point_group C2v \
        >"results/_endpoint_ham/$(basename "$out" .chk).log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "[prebuild] FAIL make_pyscf exit=$rc for $out (see log)" >&2
        tail -30 "results/_endpoint_ham/$(basename "$out" .chk).log" >&2 || true
        return 1
      fi
      _pg_chk_ok "$out" || {
        echo "[prebuild] FAIL irreps in $out" >&2
        tail -30 "results/_endpoint_ham/$(basename "$out" .chk).log" >&2 || true
        return 1
      }
    done
  fi
  echo "[prebuild] syncing filesystem before sbatch ..."
  sync || true
  sleep 2
  echo "[prebuild] done — all PG chks reloadable with .orbsym.txt; submitting"
}

n2_endpt_array() {
  python - <<'PY'
import os
methods = os.environ.get("METHODS", "all")
idxs = []
for g in range(11):
    for mc in range(6):
        kind = mc // 2  # 0 md, 1 mo, 2 it
        if methods == "iterative" and kind != 2:
            continue
        if methods == "mixed" and kind == 2:
            continue
        idxs.append(g * 6 + mc)
print(",".join(map(str, idxs)))
PY
}

h2o_endpt_array() {
  python - <<'PY'
import os
methods = os.environ.get("METHODS", "all")
idxs = []
for g in range(10):
    for mc in range(6):
        kind = mc // 2
        if methods == "iterative" and kind != 2:
            continue
        if methods == "mixed" and kind == 2:
            continue
        idxs.append(g * 6 + mc)
print(",".join(map(str, idxs)))
PY
}

checklist_array() {
  echo "0,1,2,3,4,5,6,7,8,9,10,11"
}

IFS=',' read -r -a U_MODES <<< "${ORBITAL_ROTATIONS}"

submit() {
  local script="$1"
  local array="$2"
  local label="$3"
  local exact="${4:-}"
  local u_mode="${5:-full}"
  if [[ -z "$array" ]]; then
    echo "[skip] empty array for ${label}"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] ORBITAL_ROTATION=${u_mode} EXACT_PARITY=${exact:-"(default)"} sbatch --export=ALL --array=${array} ${script}  # ${label}"
  else
    echo "[submit] ${label}: U=${u_mode} --array=${array}"
    if [[ -n "$exact" ]]; then
      ORBITAL_ROTATION="${u_mode}" EXACT_PARITY="${exact}" \
        sbatch --export=ALL --array="${array}" "${script}"
    else
      ORBITAL_ROTATION="${u_mode}" \
        sbatch --export=ALL --array="${array}" "${script}"
    fi
  fi
}

echo "=== document-exact + U={${ORBITAL_ROTATIONS}} resubmit ==="
echo "MAXITER=$MAXITER MAX_MACRO=$MAX_MACRO STABLE_SPAN=$STABLE_SPAN"
echo "METHODS=$METHODS ONLY=$ONLY ITERATIVE_REFERENCE=$ITERATIVE_REFERENCE"
echo "Results under results/<mol>_endpoint_grid/bond_*/U_{full|irrep}/..."

do_n2_end=0
do_h2o_end=0
do_chk=0
case "$ONLY" in
  all) do_n2_end=1; do_h2o_end=1; do_chk=1 ;;
  n2) do_n2_end=1; do_chk=1 ;;
  h2o) do_h2o_end=1; do_chk=1 ;;
  endpoint) do_n2_end=1; do_h2o_end=1 ;;
  checklist) do_chk=1 ;;
  *) echo "UNKNOWN ONLY=$ONLY"; exit 1 ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
  prebuild_pg_chks
else
  echo "[dry-run] would prebuild PG chks (PREBUILD_PG_CHKS=$PREBUILD_PG_CHKS)"
fi

for u in "${U_MODES[@]}"; do
  u="$(echo "$u" | tr -d '[:space:]')"
  [[ -z "$u" ]] && continue
  if [[ "$u" != "full" && "$u" != "irrep" ]]; then
    echo "[error] unknown ORBITAL_ROTATION=$u (want full|irrep)" >&2
    exit 1
  fi
  if [[ "$do_n2_end" == "1" ]]; then
    submit scripts/trillium_n2_endpoint_grid.sh \
      "$(n2_endpt_array)" \
      "n2 endpoint (${METHODS}, U=${u})" \
      "$EXACT_PARITY_N2" \
      "$u"
  fi
  if [[ "$do_h2o_end" == "1" ]]; then
    submit scripts/trillium_h2o_endpoint_grid.sh \
      "$(h2o_endpt_array)" \
      "h2o endpoint (${METHODS}, U=${u})" \
      "$EXACT_PARITY_H2O" \
      "$u"
  fi
  if [[ "$do_chk" == "1" ]]; then
    submit scripts/trillium_checklist_supplement.sh \
      "$(checklist_array)" \
      "checklist H2O+N2 (U=${u})" \
      "" \
      "$u"
  fi
done

echo "[ok] done. Monitor: squeue -u \$USER"
echo "Artifacts: results/{h2o,n2}_endpoint_grid/bond_*/U_{full,irrep}/..."
