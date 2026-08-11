#!/usr/bin/env python3
"""Per-start scatter in K at fixed pool, versus the between-rule difference.

The Q4(ii) observation is dK = K_quota - K_exhaustive, mean -0.27 and -0.45.
It is only meaningful if it exceeds the spread of K produced by the orbital
optimiser alone at a FIXED pool. This prints both side by side.

Reads results/n2_multistart/**/ produced by trillium_multistart_n2.sh.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(p):
    t = Path(p).read_text(encoding="utf-8")
    dec, i, recs = json.JSONDecoder(), 0, []
    while i < len(t):
        while i < len(t) and t[i].isspace():
            i += 1
        if i >= len(t) or t[i] != "{":
            break
        o, i = dec.raw_decode(t, i)
        recs.append(o)
    return recs


spread = defaultdict(list)
for f in glob.glob("results/n2_multistart/**/*.json", recursive=True):
    for rec in load(f):
        Ks = rec.get("K_per_start") or rec.get("K_values")
        if Ks is None and rec.get("K") is not None:
            Ks = [rec["K"]]
        if not Ks:
            continue
        m = re.search(r"bond_([0-9p.]+)", f)
        bond = float(m.group(1).replace("p", ".")) if m else None
        arm = next((a for a in ("exhaustive", "iterative", "mixed_overlap")
                    if a in f), "?")
        spread[(bond, arm)].extend(int(k) for k in Ks if k is not None)

if not spread:
    print("no multi-start results found under results/n2_multistart/")
    raise SystemExit(0)

print("=== K per start at FIXED pool (N2, NC, M=n-rsp-1) ===")
print(f"  {'R':>7} {'arm':<15} {'n':>3} {'K values':<22} {'range':>6} {'sd':>6}")
ranges = []
for (bond, arm) in sorted(spread, key=lambda t: (t[0] or 0, t[1])):
    Ks = spread[(bond, arm)]
    rng = max(Ks) - min(Ks)
    sd = st.pstdev(Ks) if len(Ks) > 1 else 0.0
    ranges.append(rng)
    print(f"  {bond:7.4f} {arm:<15} {len(Ks):>3} "
          f"{str(sorted(Ks)):<22} {rng:>6} {sd:>6.2f}")

print(f"\n  per-start range: median {st.median(ranges)}, max {max(ranges)}")

# between-rule difference from the campaign, for comparison
p = Path("tables/analysis/data/q4_summary.csv")
if p.exists():
    R = list(csv.DictReader(open(p)))
    d = []
    for r in R:
        if (r["suffix"], r["mol"], r["cost"], r["arm"]) == \
           ("m35_grid", "n2", "NC", "mixed_overlap"):
            e = next((q for q in R if (q["suffix"], q["mol"], q["cost"],
                                       q["arm"], q["bond"]) ==
                      ("m35_grid", "n2", "NC", "exhaustive", r["bond"])), None)
            if e:
                d.append(int(r["K"]) - int(e["K"]))
    if d:
        print(f"  between-rule dK  : mean {sum(d)/len(d):+.2f}, "
              f"range [{min(d):+d},{max(d):+d}]")
        print()
        if max(ranges) >= max(abs(min(d)), abs(max(d))):
            print("  VERDICT: per-start scatter is at least as large as the")
            print("           between-rule difference. dK is NOT separable")
            print("           from optimiser noise; report as observation only.")
        else:
            print("  VERDICT: the between-rule difference exceeds the per-start")
            print("           scatter at these geometries. Q4(ii) may be")
            print("           strengthened -- state the scatter alongside it.")
