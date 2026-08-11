#!/usr/bin/env bash
# AUDIT 2 (v2) — reproduce the exact/LAS bookkeeping the metrics path uses.
# Needs the repo venv only (openfermion). Cheap: safe on a login node.
#
# v2 fix: oo.json is a STREAM of concatenated JSON records, not one document.
#         Parsed via scripts/audit/_qs_json.py.
#
# Claim under test:
#   metrics.py recomputes  n_exact = n_spin + (n_spatial_kept - n_las)
#   with n_las the PRE-drop count. span(E u LAS) is full rank, so exactly one
#   combined row is qubit-dependent on P_alpha^P_beta and is dropped -> n_exact
#   lands one too small -> filter_labels_fixed_exact pins only the leading
#   n_exact generators and the LAST point-group generator stays free.
#
# Run:  bash scripts/audit/audit_02_exact_split.sh [path/to/oo.json ...]
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

if [[ $# -gt 0 ]]; then
  OO_JSONS=("$@")
else
  mapfile -t OO_JSONS < <(
    ls -1 results/h2o_endpoint_grid/bond_*/U_full/iterative/NC/oo.json 2>/dev/null | head -2
    ls -1 results/n2_endpoint_grid/bond_*/U_full/iterative/NC/oo.json  2>/dev/null | head -2
  )
fi
printf '[audit2] inspecting %d OO json(s)\n' "${#OO_JSONS[@]}"

python3 - "${OO_JSONS[@]}" <<'PY'
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo, load_json_stream          # noqa: E402

from src.exact_parity import resolve_exact_las_split
from src.clifford_sectors import clifford_symmetries_from_spatial
from src.sto3g_exact_symmetries import (
    document_exact_rank, STO3G_NORB, STO3G_EXACT_SPATIAL)


def report(path, data, norb, mol, n_records):
    print(f"\n=== {path}   ({n_records} JSON records; using the last) ===")
    pm = data.get("parity_matrix")
    pm = np.atleast_2d(np.asarray(pm, dtype=int)) if pm is not None \
        else np.zeros((0, norb), dtype=int)
    split = resolve_exact_las_split(data, pm, norb)
    combined = split["combined_matrix"]
    rank = 0
    if combined.size:
        # GF(2) rank, not the float rank numpy would give
        rows, r = [int("".join(map(str, row[::-1])), 2) for row in combined % 2], 0
        piv = []
        for v in rows:
            for p in piv:
                v = min(v, v ^ p)
            if v:
                piv.append(v)
                piv.sort(reverse=True)
                r += 1
        rank = r

    print(f"  exact_masks kept    : {split['exact_masks']}")
    print(f"  n_exact_spatial     : {split['n_exact_spatial']}")
    print(f"  n_spin_exact        : {split['n_spin_exact']}")
    print(f"  n_exact (split)     : {split['n_exact']}   "
          f"<-- document r = {document_exact_rank(mol)}")
    print(f"  n_las (pre-drop)    : {split['n_las']}")
    print(f"  las dropped in GF2  : {split['las_masks_dropped']}")
    print(f"  combined rows/rank  : {combined.shape[0]} / {rank}   "
          f"(ambient N = {norb})")
    print(f"  span(E u LAS) full rank? "
          f"{'YES  <-- finest possible sector partition' if rank >= norb else 'no'}")

    built = clifford_symmetries_from_spatial(
        combined, norb,
        include_spin_number=bool(split.get("include_spin_number_exact", True)))
    n_spin = int(built["n_spin"])
    n_spatial_kept = int(built["n_spatial"])
    n_las = int(split["n_las"])
    n_exact_metrics = n_spin + max(0, n_spatial_kept - n_las)
    print(f"  clifford: n_spin={n_spin} n_spatial_kept={n_spatial_kept} "
          f"n_total={built['n_total']}")
    print(f"  metrics.py n_exact = {n_spin} + max(0,{n_spatial_kept}-{n_las})"
          f" = {n_exact_metrics}")

    r_doc = document_exact_rank(mol)
    if n_exact_metrics != r_doc:
        gens = [lab for lab, _ in STO3G_EXACT_SPATIAL[mol]]
        n_free = r_doc - n_exact_metrics
        print(f"  *** BUG: n_exact={n_exact_metrics}, document r={r_doc}")
        print(f"  *** only the first {n_exact_metrics} Clifford generators are "
              f"pinned by filter_labels_fixed_exact")
        print(f"  *** unpinned point-group generator(s): {gens[-n_free:]}")
        print(f"  *** retained space is ~2^{n_free}x the document exact sector")
    else:
        print("  n_exact matches the document rank.")
    print(f"  exact_sector recorded: {data.get('exact_sector')!r}   "
          "(None => choose_default_exact_sector takes the DENSEST sector)")


for p in sys.argv[1:]:
    p = Path(p)
    if not p.is_file():
        print(f"[skip] {p} not found")
        continue
    try:
        recs = load_json_stream(p)
        data = load_oo(p)
    except Exception as exc:                              # noqa: BLE001
        print(f"[skip] {p}: {exc}")
        continue
    mol = "h2o" if "h2o" in str(p) else "n2"
    report(str(p), data, STO3G_NORB[mol], mol, len(recs))
PY
