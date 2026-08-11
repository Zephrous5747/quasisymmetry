#!/usr/bin/env bash
# AUDIT 1 — pure combinatorics, no pyscf/block2 needed. Safe on a login node.
#
# Question: what Hilbert-space dimension SHOULD the document exact sector have,
# and which (sub)set of exact generators reproduces the observed
# K = dim = 261 (H2O) / 3584 (N2) reported in
# tables/analysis/endpoint_document_exact_full_U_tables.tex ?
#
# Run:  bash scripts/audit/audit_01_sector_dims.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

python3 - <<'PY'
from itertools import combinations, product
from collections import Counter

CASES = {
  "h2o": dict(norb=7, na=5, gens={"Q_B1": {4}, "Q_B2": {2, 6}},
              pdf_exact_dim=133, observed=261),
  "n2":  dict(norb=10, na=7, gens={"Q_pix": {4, 7}, "Q_piy": {5, 8},
                                   "Q_u": {1, 3, 4, 5, 9}},
              pdf_exact_dim=1824, observed=3584),
}

for mol, c in CASES.items():
    norb, na, gens = c["norb"], c["na"], c["gens"]
    names = list(gens)
    strings = list(combinations(range(norb), na))
    cls = Counter(tuple(len(set(s) & gens[g]) % 2 for g in names) for s in strings)

    # sign sector of a determinant = XOR of alpha/beta class bits (0 => Q = +1)
    sec = Counter()
    for ka, va in cls.items():
        for kb, vb in cls.items():
            sec[tuple(a ^ b for a, b in zip(ka, kb))] += va * vb

    total = sum(sec.values())
    allplus = sec[tuple(0 for _ in names)]
    print(f"=== {mol.upper()}  norb={norb} nelec=({na},{na}) "
          f"det space = {len(strings)}^2 = {total}")
    print(f"    generators (hardcoded in src/sto3g_exact_symmetries.py): "
          + ", ".join(f"{n}={sorted(gens[n])}" for n in names))
    print(f"    all-(+1) exact sector dim = {allplus}   "
          f"(PDF says {c['pdf_exact_dim']}) "
          f"{'MATCH' if allplus == c['pdf_exact_dim'] else 'MISMATCH'}")
    print("    dim of every sign sector (0 = Q=+1):")
    for k in sorted(sec):
        lab = ",".join(f"{n}={'+' if b == 0 else '-'}" for n, b in zip(names, k))
        print(f"      {lab:40s} {sec[k]}")

    obs = c["observed"]
    print(f"    --- which retained space equals the observed dim {obs}? ---")
    hits = []
    # leave a subset of generators UNCONSTRAINED, pin the rest to +/-1
    for r in range(len(names)):
        for free in combinations(range(len(names)), r):
            pinned = [i for i in range(len(names)) if i not in free]
            for signs in product([0, 1], repeat=len(pinned)):
                d = sum(v for k, v in sec.items()
                        if all(k[i] == s for i, s in zip(pinned, signs)))
                if d == obs:
                    desc = ", ".join(
                        f"{names[i]}={'+' if s == 0 else '-'}"
                        for i, s in zip(pinned, signs)) or "(nothing pinned)"
                    freed = ", ".join(names[i] for i in free) or "(none)"
                    hits.append(f"      pinned: {desc:45s} FREE: {freed}")
    print("\n".join(hits) if hits else "      no exact match found")
    print()

print("READ THIS:")
print("  If the observed dim is ~2x the PDF exact-sector dim, exactly one exact")
print("  generator is NOT being pinned by the sector filter. See audit 2.")
PY
