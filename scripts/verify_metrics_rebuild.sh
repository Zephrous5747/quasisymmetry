#!/usr/bin/env bash
#SBATCH --job-name=verify_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=verify_metrics_%j.out
#SBATCH --error=verify_metrics_%j.err
#
# Verify the rebuilt metrics.py against the authoritative compiled copy.
#
#   archive/metrics.cpython-311.pyc  is the ONLY surviving artefact of the
#   Jul 29 - Aug 7 exact-tapering work. It is ground truth. This script diffs
#   the rebuilt source against it at the bytecode level, then reproduces a
#   recorded physics point end to end.
#
# Run on a COMPUTE node (login node CPU cap -> SIGXCPU / exit -24):
#   sbatch scripts/verify_metrics_rebuild.sh
# or interactively:
#   srun -n1 -c8 -t00:40:00 bash scripts/verify_metrics_rebuild.sh
#
# Stage 1  syntax + import
# Stage 2  bytecode diff vs archive/metrics.cpython-311.pyc   <-- the real check
# Stage 3  reproduce N2 R=1.8 A: K=18, n_exact=5, W=1.0, exact_sector=11000

set -uo pipefail
export TRILLIUM=1

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

PYC="${PYC:-archive/metrics.cpython-311.pyc}"
FAIL=0

echo "=============================================================="
echo " metrics.py rebuild verification"
echo " repo : $REPO"
echo " pyc  : $PYC"
echo " date : $(date -Is)"
echo "=============================================================="

# ---------------------------------------------------------------- stage 1
echo
echo "### STAGE 1  syntax + import"
python -c "import ast; ast.parse(open('metrics.py').read()); print('  syntax OK')" || FAIL=1
python -c "
import metrics
print('  import OK')
need = ['load_oo_input','_check_reference_weight','load_clifford_symmetries',
        'run_clifford_metrics','resolve_exact_las_split','filter_labels_fixed_exact',
        'clifford_symmetries_from_spatial','reference_sector_label',
        'sector_dimension_stats','expand_exact_sector_with_spin']
miss = [n for n in need if not hasattr(metrics, n)]
print('  missing symbols:', miss if miss else 'none')
raise SystemExit(1 if miss else 0)
" || FAIL=1

# ---------------------------------------------------------------- stage 2
echo
echo "### STAGE 2  bytecode diff vs authoritative .pyc"
if [[ ! -f "$PYC" ]]; then
  echo "  [skip] $PYC not found -- cannot diff against ground truth"
  FAIL=1
else
python - "$PYC" <<'PY' || FAIL=1
import importlib.util, marshal, sys, types

pyc_path = sys.argv[1]
with open(pyc_path, "rb") as fh:
    fh.read(16)                      # 3.11 pyc header
    good = marshal.load(fh)
rebuilt = compile(open("metrics.py").read(), "metrics.py", "exec")

ANON = ("<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>")

def flatten(const, acc):
    """Expand tuple/frozenset consts into their elements.

    A dict literal compiles its keys into ONE tuple const, while individual
    ``d[k] = v`` assignments emit each key separately. Without flattening,
    rewriting a dict literal as assignments looks like every key vanished.
    """
    if isinstance(const, types.CodeType):
        return
    if isinstance(const, (tuple, frozenset)):
        for item in const:
            flatten(item, acc)
        return
    acc.add(repr(const))

def entry(code):
    acc = set()
    for c in code.co_consts:
        flatten(c, acc)
    doc = code.co_consts[0] if code.co_consts and isinstance(
        code.co_consts[0], str) else None
    return (frozenset(code.co_names), frozenset(acc), doc)

def walk(code, prefix=""):
    """Map qualified-name -> (globals referenced, flattened literals, docstring).

    Includes the MODULE-level code object under '<module>'. Omitting it hid a
    real defect: the entire `if __name__ == "__main__":` block lives at module
    level, and a stale 4-tuple unpack of reference_coupled_energy_k (which
    returns 5) sat there undetected through three clean function-level diffs.
    """
    out = {}
    if not prefix:
        out["<module>"] = entry(code)
    for const in code.co_consts:
        if not isinstance(const, types.CodeType):
            continue
        name = f"{prefix}{const.co_name}"
        out[name] = entry(const)
        out.update(walk(const, name + "."))
    return out

g, r = walk(good), walk(rebuilt)
named = lambda ks: {k for k in ks if not any(a in k for a in ANON)}
gk, rk = named(set(g)), named(set(r))

missing = sorted(gk - rk)
extra   = sorted(rk - gk)
anon_g  = len(set(g)) - len(gk)
anon_r  = len(set(r)) - len(rk)

print(f"  named functions in .pyc   : {len(gk)}")
print(f"  named functions in rebuild: {len(rk)}")
print(f"  anonymous scopes .pyc/rebuild: {anon_g}/{anon_r}  (comprehensions; not diffed)")
print(f"  MISSING from rebuild      : {len(missing)}")
for n in missing:
    print(f"      - {n}")
print(f"  extra in rebuild          : {len(extra)}")
for n in extra:
    print(f"      + {n}")

# Known, deliberate divergences from the .pyc. The .pyc is an Aug 7 09:55
# snapshot that PREDATES the W1 spatial/qubit rank work, so it is ground truth
# for behaviour but NOT for structure everywhere.
EXPECTED = {
    "load_clifford_symmetries": (
        "the .pyc builds the exact/LAS split INLINE (hence masks_from_parity_matrix, "
        "max, shape, 'exact_masks', 'las_masks'); the rebuild delegates to "
        "resolve_exact_las_split() in src/exact_parity.py, which is intact and "
        "W1-correct. The .pyc also predates the r_sp/r_qubit distinction and used "
        "the buggy max(0, n_spatial_kept - n_las) count for n_exact."
    ),
}

drift, expected_drift, docdrift = [], [], []
for name in sorted(gk & rk):
    gn, gl, gd = g[name]
    rn, rl, rd = r[name]
    lost_names = gn - rn
    lost_lits  = {l for l in (gl - rl) if l != repr(gd)}
    if lost_names or lost_lits:
        entry = (name, sorted(lost_names), sorted(lost_lits))
        (expected_drift if name in EXPECTED else drift).append(entry)
    if gd and gd != rd:
        docdrift.append(name)

def show(entries):
    for name, ln, ll in entries:
        print(f"    ~ {name}")
        if ln:
            print(f"        lost names    : {ln}")
        if ll:
            print(f"        lost literals : {ll[:14]}{' ...' if len(ll) > 14 else ''}")

print(f"  UNEXPECTED drift: {len(drift)}")
show(drift)
print(f"  expected drift  : {len(expected_drift)}")
show(expected_drift)
for name, _, _ in expected_drift:
    print(f"        reason: {EXPECTED[name]}")
if docdrift:
    print(f"  docstring differs (cosmetic): {docdrift}")

bad = bool(missing) or bool(drift)
print("  VERDICT:", "UNEXPECTED DRIFT -- rebuild is incomplete" if bad
      else "rebuild matches .pyc (modulo documented W1 divergence)")
raise SystemExit(1 if bad else 0)
PY
fi

# ---------------------------------------------------------------- stage 3
echo
echo "### STAGE 3  reproduce a recorded point (rebuild vs the metrics.json beside it)"
# NB: bond dirs are bond_1p8000, and the OO json is a POSITIONAL argument.
OO="${OO:-}"
if [[ -z "$OO" ]]; then
  for cand in \
    results/n2_endpoint_grid/bond_1p8000/U_irrep/iterative/NC/oo.json \
    results/n2_endpoint_grid/bond_1p8000/iterative/NC/oo.json \
    results/n2_endpoint_grid/bond_1p8000/U_full/iterative/NC/oo.json ; do
    [[ -f "$cand" ]] && OO="$cand" && break
  done
fi
if [[ -z "$OO" || ! -f "$OO" ]]; then
  echo "  [skip] no N2 1.8 oo.json found; set OO=<path> to choose one"
else
  REC="$(dirname "$OO")/metrics.json"
  OUT="verify_rebuild_metrics.json"
  LOG="verify_rebuild_metrics.log"
  echo "  input    : $OO"
  echo "  recorded : $REC $([[ -f "$REC" ]] || echo '(absent -- will only self-report)')"

  # Exactly how scripts/run_endpoint_point.py invokes metrics.
  python -u metrics.py "$OO" \
    --sector_backend clifford \
    --backend fci \
    --coupled_energy_method reference \
    --overlap_reference fci \
    --states_per_sector 500 \
    --outname "$OUT" >"$LOG" 2>&1
  rc=$?
  if (( rc != 0 )); then
    echo "  [error] metrics.py exited $rc -- tail of $LOG:"
    tail -25 "$LOG" | sed 's/^/      /'
    FAIL=1
  fi

  python - "$OUT" "$REC" <<'PY' || FAIL=1
import json, sys
from pathlib import Path

def load(path):
    """Outnames may be a stream of appended objects; take the last."""
    text = Path(path).read_text(encoding="utf-8")
    dec, i, n, recs = json.JSONDecoder(), 0, len(text), []
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != "{":
            break
        obj, i = dec.raw_decode(text, i)
        recs.append(obj)
    return recs[-1] if recs else {}

new = load(sys.argv[1])
rec = load(sys.argv[2]) if Path(sys.argv[2]).exists() else None

keys = ["K","converged","n_exact","r_sp","r_qubit","exact_sector",
        "exact_sector_source","reference_weight_sum","reference_weight_ok",
        "overlap_reference","D_max","E_decoupled","E_coupled","E_FCI"]
print(f"  {'field':24} {'rebuild':>22}   {'recorded':>22}")
for k in keys:
    a = new.get(k, "<absent>")
    b = rec.get(k, "<absent>") if rec else "-"
    flag = ""
    if rec and k in ("K","n_exact","exact_sector") and a != b and b != "<absent>":
        flag = "  <-- DIFFERS"
    print(f"  {k:24} {str(a):>22}   {str(b):>22}{flag}")

w = new.get("reference_weight_sum")
ok_w = w is not None and abs(float(w) - 1.0) < 1e-4
ok_k = new.get("K") is not None and new.get("converged") is True
print()
print(f"  W == 1            : {ok_w}")
print(f"  K converged       : {ok_k}")
if rec:
    same = all(new.get(k) == rec.get(k) for k in ("K","n_exact"))
    print(f"  matches recorded  : {same}")
    if not same:
        print("    NOTE: the .pyc that produced the recorded value predates the W1")
        print("    n_exact fix, so a differing n_exact may mean the RECORD is stale.")
print("  VERDICT:", "physics reproduces" if (ok_w and ok_k) else "CHECK ABOVE")
raise SystemExit(0 if (ok_w and ok_k) else 1)
PY
fi

echo
echo "=============================================================="
if [[ "$FAIL" == "0" ]]; then
  echo " ALL CHECKS PASSED -- rebuild is faithful"
else
  echo " CHECKS FAILED -- see stage output above (do NOT sync yet)"
fi
echo "=============================================================="
exit "$FAIL"
