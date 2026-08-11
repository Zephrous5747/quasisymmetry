#!/usr/bin/env bash
# AUDIT 9 — why do iterative and exhaustive report identical K everywhere?
#
# Two candidate explanations, with opposite implications:
#
#   (a) BENIGN / EXPECTED. The two arms select the SAME LAS pool at every
#       geometry. Then they pose the same orbital-optimisation problem, converge
#       to the same U, and K must agree exactly. The equality is then a positive
#       result -- the polynomial pool recovers the exhaustive optimum -- and NOT
#       an independent comparison of two selection rules.
#
#   (b) SUSPICIOUS. The arms select DIFFERENT pools yet still give identical K
#       at all 42 points. That would mean K is insensitive to the pool, or that
#       orbital optimisation collapses to the same point regardless of input,
#       or that OO is a no-op.
#
# This distinguishes them, and separately checks that OO actually ran:
# cost must decrease, nit/nfev > 0, and the rotation must be non-zero.
#
# JSON only -- safe on a login node.
#   bash scripts/audit/audit_09_arm_equivalence.sh

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
mkdir -p tables/audit

python3 - <<'PY'
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo


def gf2_rref(masks, nbits):
    """Row-reduce packed GF(2) rows; returns pivot rows."""
    rows, piv = [], []
    for m in (int(x) for x in masks):
        cur = m
        for p, r in zip(piv, rows):
            if (cur >> p) & 1:
                cur ^= r
        if cur:
            p = cur.bit_length() - 1
            rows.append(cur)
            piv.append(p)
            order = sorted(range(len(piv)), key=lambda i: -piv[i])
            rows = [rows[i] for i in order]
            piv = [piv[i] for i in order]
    return rows


def span_key(masks, exact, nbits):
    """Canonical key for span(masks) reduced modulo span(exact)."""
    e = gf2_rref(exact, nbits)
    red = []
    for m in (int(x) for x in masks):
        cur = m
        for r in e:
            p = r.bit_length() - 1
            if (cur >> p) & 1:
                cur ^= r
        if cur:
            red.append(cur)
    return tuple(sorted(gf2_rref(red, nbits)))


import os
SUFFIX = os.environ.get("RESULTS_SUFFIX", "q4_grid")
print(f"reading results/*_{SUFFIX}/")

rows = {}
for f in sorted(glob.glob(f"results/*_{SUFFIX}/bond_*/U_*/*/*/oo.json")):
    parts = Path(f).parts
    mol = parts[1].split(f"_{SUFFIX}")[0]
    bond = float(re.sub(r"^bond_", "", parts[2]).replace("p", "."))
    method, cost = parts[4], parts[5]
    try:
        d = load_oo(f)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [skip] {f}: {exc}")
        continue
    las = d.get("las_masks") or d.get("accumulated_masks") or []
    exact = d.get("exact_masks") or []
    rot = d.get("rotation") or []
    nbits = 10 if mol == "n2" else 7
    rows[(mol, bond, method, cost)] = dict(
        las=tuple(sorted(int(m) for m in las)),
        span=span_key(las, exact, nbits),
        cost_before=d.get("cost_before"),
        cost_after=d.get("cost_after"),
        nit=d.get("nit"), nfev=d.get("nfev"),
        converged=d.get("converged"),
        rot=[float(v) for v in rot],
        M=d.get("M"), M_eff=d.get("M_eff"),
    )

print(f"points: {len(rows)}")
mols = sorted({k[0] for k in rows})

# ---- 1. is orbital optimisation actually doing anything? -----------------
print("\n=== 1. orbital optimisation activity ===")
noop, zero_rot, worse = [], [], []
for k, v in rows.items():
    cb, ca = v["cost_before"], v["cost_after"]
    if cb is None or ca is None:
        continue
    if abs(cb - ca) < 1e-12:
        noop.append(k)
    if ca > cb + 1e-12:
        worse.append(k)
    if v["rot"] and max(abs(x) for x in v["rot"]) < 1e-10:
        zero_rot.append(k)
drops = [v["cost_before"] - v["cost_after"] for v in rows.values()
         if v["cost_before"] is not None and v["cost_after"] is not None]
drops.sort()
if drops:
    print(f"  cost reduction (cost_before - cost_after) over {len(drops)} points:")
    print(f"    min={drops[0]:.6g}  median={drops[len(drops)//2]:.6g}  max={drops[-1]:.6g}")
print(f"  points where OO did nothing (cost unchanged): {len(noop)}")
print(f"  points where cost got WORSE:                  {len(worse)}")
print(f"  points with an all-zero rotation vector:      {len(zero_rot)}")
for k in (noop + zero_rot + worse)[:6]:
    print(f"      {k}")
if not noop and not zero_rot and not worse:
    print("  -> OO ran and reduced the cost at every point")

# ---- 2. do iterative and exhaustive select the SAME pool? ----------------
print("\n=== 2. selected pool: iterative vs exhaustive ===")
for mol in mols:
    for cost in sorted({k[3] for k in rows if k[0] == mol}):
        bonds = sorted({k[1] for k in rows if k[0] == mol and k[3] == cost})
        same_mask, same_span, diff = 0, 0, []
        for b in bonds:
            a = rows.get((mol, b, "iterative", cost))
            e = rows.get((mol, b, "exhaustive", cost))
            if not a or not e:
                continue
            if a["las"] == e["las"]:
                same_mask += 1
            elif a["span"] == e["span"]:
                same_span += 1
            else:
                diff.append(b)
        n = same_mask + same_span + len(diff)
        print(f"  {mol:4s} {cost:9s} of {n} geometries: "
              f"identical masks {same_mask}, same span only {same_span}, "
              f"genuinely different {len(diff)}")
        for b in diff[:4]:
            a, e = rows[(mol, b, "iterative", cost)], rows[(mol, b, "exhaustive", cost)]
            print(f"      R={b}: iter={list(a['las'])}")
            print(f"           exh ={list(e['las'])}")

# ---- 3. did the arms converge to the SAME orbitals? ----------------------
print("\n=== 3. converged rotation: max |x_iter - x_exh| (and vs mixed_overlap) ===")
for mol in mols:
    for cost in sorted({k[3] for k in rows if k[0] == mol}):
        bonds = sorted({k[1] for k in rows if k[0] == mol and k[3] == cost})
        de, dm = [], []
        for b in bonds:
            a = rows.get((mol, b, "iterative", cost))
            e = rows.get((mol, b, "exhaustive", cost))
            m = rows.get((mol, b, "mixed_overlap", cost))
            if a and e and len(a["rot"]) == len(e["rot"]) and a["rot"]:
                de.append(max(abs(x - y) for x, y in zip(a["rot"], e["rot"])))
            if a and m and len(a["rot"]) == len(m["rot"]) and a["rot"]:
                dm.append(max(abs(x - y) for x, y in zip(a["rot"], m["rot"])))
        f = lambda v: f"max={max(v):.3g}" if v else "n/a"
        print(f"  {mol:4s} {cost:9s} iter vs exh: {f(de)}   iter vs mixed: {f(dm)}")

print("""
Reading this
------------
If (2) reports identical masks everywhere and (3) shows iter-vs-exh rotation
differences at solver tolerance (~1e-5), then the arms are the SAME RUN and
equal K is a tautology. Report it as "the polynomial pool attains the exhaustive
optimum", not as "two rules give equal K" -- the latter implies an independent
comparison that was not performed.

If (2) reports genuinely different pools but K is still identical everywhere,
that is the suspicious case: investigate whether K is pool-insensitive at
M = n - r_sp, or whether OO is converging to the same U regardless of input.
""")
PY
