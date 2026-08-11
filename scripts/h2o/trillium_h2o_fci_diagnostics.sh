#!/usr/bin/env bash
#SBATCH --job-name=h2o_fci_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=h2o_fci_diag_%A_%a.out
#SBATCH --error=h2o_fci_diag_%A_%a.err
#
# Trillium: FCI-backend sector diagnostics (E_dec, K) on H2O FCI-selection OO
# results, then a three-panel plot (cost | |E_dec-E_FCI| | K).
#
# Array tasks (override OO_* env vars as needed):
#   0 = greedy    + NC
#   1 = greedy    + variance
#   2 = iterative + NC
#   3 = iterative + variance
#
# Do NOT pass --mem-per-cpu. Max wall time is 24h (4h is enough for H2O FCI).
#
# Submit from repo root (after FCI selection OO JSONs exist):
#   sbatch scripts/h2o/trillium_h2o_fci_diagnostics.sh
#
# After the array finishes, plot with a dependency job:
#   ARRAY_JOBID=<id>
#   sbatch --dependency=afterok:${ARRAY_JOBID} \
#     --export=ALL,PLOT_ONLY=1 \
#     scripts/h2o/trillium_h2o_fci_diagnostics.sh
#
# Or locally:
#   python scripts/h2o/plot_select_diagnostics.py tables/h2o/fci_select_diagnostics.csv
#
# Optional env:
#   STEM, OO_GREEDY_NC, OO_GREEDY_VAR, OO_ITER_NC, OO_ITER_VAR
#   DIAG_CSV, PLOT_PNG, STATES_PER_SECTOR, GEOM_PARAM, PLOT_ONLY=1

set -euo pipefail

export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

STEM="${STEM:-H2O_OH0.9580_104.5000}"
BASE="results/h2o_select/${STEM}"
OO_GREEDY_NC="${OO_GREEDY_NC:-${BASE}/greedy_NC_fci_s3q2}"
OO_GREEDY_VAR="${OO_GREEDY_VAR:-${BASE}/greedy_variance_fci_s3q2}"
OO_ITER_NC="${OO_ITER_NC:-${BASE}/iterative_NC_fci_n5}"
OO_ITER_VAR="${OO_ITER_VAR:-${BASE}/iterative_variance_fci_n5}"

DIAG_CSV="${DIAG_CSV:-tables/h2o/fci_select_diagnostics.csv}"
PLOT_PNG="${PLOT_PNG:-tables/h2o/fci_select_diagnostics_three_panel.png}"
STATES_PER_SECTOR="${STATES_PER_SECTOR:-200}"
GEOM_PARAM="${GEOM_PARAM:-0.958}"

mkdir -p "$(dirname "$DIAG_CSV")" "$(dirname "$PLOT_PNG")" "results/h2o_select_metrics/${STEM}"

if [[ "${PLOT_ONLY:-0}" == "1" ]]; then
  echo "[plot] $(date -Is) CSV=$DIAG_CSV -> $PLOT_PNG"
  python -u scripts/h2o/plot_select_diagnostics.py "$DIAG_CSV" \
    --output "$PLOT_PNG" \
    --title "H2O FCI selection · ${STEM}"
  exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
case "$TASK" in
  0) SELECT=greedy;    COST=NC;       OO="$OO_GREEDY_NC" ;;
  1) SELECT=greedy;    COST=variance; OO="$OO_GREEDY_VAR" ;;
  2) SELECT=iterative; COST=NC;       OO="$OO_ITER_NC" ;;
  3) SELECT=iterative; COST=variance; OO="$OO_ITER_VAR" ;;
  *)
    echo "[error] unexpected array task id=$TASK (expected 0..3)" >&2
    exit 1
    ;;
esac

METRICS_OUT="results/h2o_select_metrics/${STEM}/${SELECT}_${COST}_fci_metrics.json"

echo "[job] $(date -Is) task=$TASK SELECT=$SELECT COST=$COST"
echo "[job] OO=$OO"
echo "[job] METRICS_OUT=$METRICS_OUT DIAG_CSV=$DIAG_CSV"

export DIAG_CSV STEM GEOM_PARAM SELECT COST OO METRICS_OUT

append_csv_row() {
  local status="$1"
  local message="${2:-}"
  export STATUS="$status" MESSAGE="$message"
  python -u - <<'PY'
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

FIELDNAMES = [
    "molecule",
    "geometry_param",
    "select",
    "cost_function",
    "n_singles",
    "n_quartets",
    "n_sym",
    "m_round",
    "cost",
    "E_FCI",
    "E_decoupled",
    "edec_error",
    "K",
    "oo_json",
    "metrics_json",
    "status",
    "message",
    "reference",
    "backend",
]


def load_oo(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    objs = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] != "{":
            break
        obj, end = dec.raw_decode(raw, i)
        if isinstance(obj, dict):
            objs.append(obj)
        i = end
    if not objs:
        raise ValueError(f"no JSON object in {path}")
    return objs[-1]


def pool_cost(oo: dict) -> float | None:
    rounds = oo.get("rounds") or []
    if rounds:
        opt = rounds[-1].get("optimization") or {}
        if opt.get("cost_after") is not None:
            return float(opt["cost_after"])
    costs = oo.get("selected_costs")
    if costs:
        return float(sum(float(c) for c in costs))
    return None


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    with path.open("a+", encoding="utf-8", newline="") as handle:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            empty = handle.read(1) == ""
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
            if empty:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


select = os.environ["SELECT"]
cost_fn = os.environ["COST"]
oo_path = os.environ["OO"]
metrics_path = os.environ["METRICS_OUT"]
status = os.environ.get("STATUS", "failed")
message = os.environ.get("MESSAGE", "")

row = {
    "molecule": os.environ.get("STEM", ""),
    "geometry_param": os.environ.get("GEOM_PARAM", ""),
    "select": select,
    "cost_function": cost_fn,
    "n_singles": "3" if select == "greedy" else "",
    "n_quartets": "2" if select == "greedy" else "",
    "n_sym": "5",
    "m_round": "1" if select == "iterative" else "",
    "cost": "",
    "E_FCI": "",
    "E_decoupled": "",
    "edec_error": "",
    "K": "",
    "oo_json": oo_path,
    "metrics_json": metrics_path if Path(metrics_path).is_file() else "",
    "status": status,
    "message": message,
    "reference": "fci",
    "backend": "fci",
}

if Path(oo_path).is_file():
    try:
        oo = load_oo(oo_path)
        c = pool_cost(oo)
        if c is not None:
            row["cost"] = repr(c)
    except Exception as exc:  # noqa: BLE001
        row["message"] = (row["message"] + f"; oo parse: {exc}").strip("; ")

if status == "ok" and Path(metrics_path).is_file():
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    e_fci = metrics.get("E_FCI")
    e_dec = metrics.get("E_decoupled")
    k = metrics.get("K")
    row["E_FCI"] = "" if e_fci is None else repr(float(e_fci))
    row["E_decoupled"] = "" if e_dec is None else repr(float(e_dec))
    if e_fci is not None and e_dec is not None:
        row["edec_error"] = repr(abs(float(e_dec) - float(e_fci)))
    if k is not None:
        kf = float(k)
        row["K"] = repr(int(kf) if kf.is_integer() else kf)

append_row(Path(os.environ["DIAG_CSV"]), row)
print(
    "[csv] appended",
    row["status"],
    select,
    cost_fn,
    "cost=",
    row["cost"],
    "edec_error=",
    row["edec_error"],
    "K=",
    row["K"],
)
PY
}

if [[ ! -f "$OO" ]]; then
  echo "[error] missing OO JSON: $OO" >&2
  echo "[error] run FCI selection first (trillium_h2o_greedy_vs_iterative.sh)" >&2
  RC=1
  append_csv_row "missing_oo" "OO JSON not found"
  exit 1
fi

echo "[run] python -u metrics.py \"$OO\" --backend fci --coupled_energy_method perturbation --states_per_sector $STATES_PER_SECTOR --outname \"$METRICS_OUT\""
set +e
python -u metrics.py "$OO" \
  --backend fci \
  --coupled_energy_method perturbation \
  --states_per_sector "$STATES_PER_SECTOR" \
  --outname "$METRICS_OUT"
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
  append_csv_row "ok" ""
else
  append_csv_row "failed" "metrics exited with code ${RC}"
  echo "[error] metrics failed SELECT=$SELECT COST=$COST rc=$RC" >&2
  exit "$RC"
fi

# Opportunistic plot when at least two ok rows exist (final plot via PLOT_ONLY).
N_OK=$(python -u - <<PY
import csv
from pathlib import Path
p = Path("${DIAG_CSV}")
if not p.is_file():
    print(0)
else:
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    print(sum(1 for r in rows if r.get("status") == "ok"))
PY
)
echo "[job] ok rows in CSV so far: $N_OK"
if [[ "$N_OK" -ge 2 ]]; then
  python -u scripts/h2o/plot_select_diagnostics.py "$DIAG_CSV" \
    --output "$PLOT_PNG" \
    --title "H2O FCI selection · ${STEM}" || true
fi

echo "[ok] finished SELECT=$SELECT COST=$COST at $(date -Is)"
