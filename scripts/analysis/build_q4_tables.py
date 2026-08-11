#!/usr/bin/env python3
"""Build the Q4 data tables for report_las_sto3g.tex from both campaigns.

Every number quoted in the Q4 sections is emitted here, so nothing is
transcribed by hand. Reads results/{mol}_{suffix}/ for each budget and writes:

  tables/analysis/data/q4_summary.csv          one row per point (all 252)
  tables/analysis/gen_pool_equivalence.tex     Q4(i): cost/span/K agreement
  tables/analysis/gen_k_by_arm.tex             Q4(ii): K per arm per geometry
  tables/analysis/gen_budget_tradeoff.tex      D_max / K across budgets

Run on the cluster (JSON only, login-node safe):
  python3 scripts/analysis/build_q4_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo                                  # noqa: E402

CAMPAIGNS = [("q4_grid", r"n-\rsp"), ("m35_grid", r"n-\rsp-1")]
ARMS = ["exhaustive", "iterative", "mixed_overlap"]
ARM_TEX = {"exhaustive": "exhaustive", "iterative": "iterative",
           "mixed_overlap": "quota"}
OUT = Path("tables/analysis")
(OUT / "data").mkdir(parents=True, exist_ok=True)


def load_stream(path):
    text = Path(path).read_text(encoding="utf-8")
    dec, i, recs = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        obj, i = dec.raw_decode(text, i)
        recs.append(obj)
    return recs[-1] if recs else {}


def gf2_rref(masks):
    rows, piv = [], []
    for m in (int(x) for x in masks):
        cur = m
        for p, r in zip(piv, rows):
            if (cur >> p) & 1:
                cur ^= r
        if cur:
            rows.append(cur)
            piv.append(cur.bit_length() - 1)
            o = sorted(range(len(piv)), key=lambda i: -piv[i])
            rows, piv = [rows[i] for i in o], [piv[i] for i in o]
    return rows


def span_key(masks, exact):
    e = gf2_rref(exact)
    red = []
    for m in (int(x) for x in masks):
        cur = m
        for r in e:
            if (cur >> (r.bit_length() - 1)) & 1:
                cur ^= r
        if cur:
            red.append(cur)
    return tuple(sorted(gf2_rref(red)))


# ---------------------------------------------------------------- collect
rows = {}
for suffix, _ in CAMPAIGNS:
    for f in sorted(glob.glob(f"results/*_{suffix}/bond_*/U_irrep/*/*/metrics.json")):
        p = Path(f).parts
        mol = p[1].split(f"_{suffix}")[0]
        bond = float(re.sub(r"^bond_", "", p[2]).replace("p", "."))
        arm, cost = p[4], p[5]
        d = load_stream(f)
        oo_path = Path(f).with_name("oo.json")
        oo = load_oo(oo_path) if oo_path.exists() else {}
        las = oo.get("las_masks") or oo.get("accumulated_masks") or []
        exact = oo.get("exact_masks") or []
        rows[(suffix, mol, bond, arm, cost)] = dict(
            suffix=suffix, mol=mol, bond=bond, arm=arm, cost=cost,
            K=d.get("K"), converged=d.get("converged"),
            D_max=d.get("D_max"), D_min=d.get("D_min"),
            retained=d.get("exact_sector_total_dim"),
            n_sectors=d.get("n_sectors_retained"),
            n_exact=d.get("n_exact"), r_sp=d.get("r_sp"), r_qubit=d.get("r_qubit"),
            W=d.get("reference_weight_sum"),
            E_FCI=d.get("E_FCI"), E_dec=d.get("E_decoupled"),
            E_cpl=d.get("E_coupled"), dE=d.get("dE"),
            M=oo.get("M"), M_eff=oo.get("M_eff"),
            cost_before=oo.get("cost_before"), cost_after=oo.get("cost_after"),
            span=span_key(las, exact), n_las=len(las),
        )

print(f"points collected: {len(rows)}")

with open(OUT / "data" / "q4_summary.csv", "w", newline="") as fh:
    cols = [c for c in next(iter(rows.values())) if c != "span"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows.values():
        w.writerow(r)
print(f"wrote {OUT/'data'/'q4_summary.csv'}")


def esc(x, fmt="{}"):
    return "--" if x is None else fmt.format(x)


# ------------------------------------------------- Q4(i) pool equivalence
lines = [
    r"\begin{tabular}{llrrrr}", r"\hline",
    r"budget & cost & points & $\max|\Delta c|$ & span agree & $K$ agree\\",
    r"\hline",
]
for suffix, blabel in CAMPAIGNS:
    for cost in ("NC", "variance"):
        pts = dc = sp = ka = 0
        for (s, mol, b, arm, c), v in rows.items():
            if s != suffix or c != cost or arm != "iterative":
                continue
            e = rows.get((s, mol, b, "exhaustive", c))
            if not e:
                continue
            pts += 1
            if v["cost_before"] is not None and e["cost_before"] is not None:
                dc = max(dc, abs(v["cost_before"] - e["cost_before"]))
            sp += int(v["span"] == e["span"])
            ka += int(v["K"] == e["K"])
        mant, expo = f"{dc:.1e}".split("e")
        dtex = f"{mant}\\times10^{{{int(expo)}}}"
        lines.append(
            f"$M={blabel}$ & {cost} & {pts} & ${dtex}$ "
            f"& {sp}/{pts} & {ka}/{pts}\\\\"
        )
lines += [r"\hline", r"\end{tabular}"]
(OUT / "gen_pool_equivalence.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {OUT/'gen_pool_equivalence.tex'}")

# ------------------------------------------------------ Q4(ii) K by arm
# ONE tabular with an explicit subheading per (budget, molecule, cost) block,
# so the whole thing carries a single table number and every block is labelled.
MOLTEX = {"h2o": r"\mathrm{H_2O}", "n2": r"\mathrm{N_2}"}
t = [r"\begin{tabular}{lrrr}", r"\hline",
     r"$R$ (\AA) & exhaustive & iterative & quota\\"]
summary = []
for suffix, blabel in CAMPAIGNS:
    for mol in ("h2o", "n2"):
        for cost in ("NC", "variance"):
            bonds = sorted({b for (s, m, b, a, c) in rows
                            if s == suffix and m == mol and c == cost})
            if not bonds:
                continue
            head = (f"${MOLTEX[mol]}$, {cost} cost, $M={blabel}$")
            t += [r"\hline",
                  r"\multicolumn{4}{l}{\emph{" + head + r"}}\\", r"\hline"]
            lo = hi = eq = 0
            tot = 0.0
            for b in bonds:
                g = lambda a: rows.get((suffix, mol, b, a, cost), {}).get("K")
                e, i, q = g("exhaustive"), g("iterative"), g("mixed_overlap")
                t.append(f"{b:.4f} & {esc(e)} & {esc(i)} & {esc(q)}\\\\")
                if e is not None and q is not None:
                    d = q - e
                    tot += d
                    lo += d < 0
                    hi += d > 0
                    eq += d == 0
            n = lo + hi + eq
            if n:
                summary.append(f"% {mol} {cost} M={blabel}: quota vs exhaustive "
                               f"lower {lo}, equal {eq}, higher {hi}, "
                               f"mean dK = {tot/n:+.2f}")
t += [r"\hline", r"\end{tabular}"]
(OUT / "gen_k_by_arm.tex").write_text("\n".join(summary + t) + "\n")
print(f"wrote {OUT/'gen_k_by_arm.tex'}")

# --------------------------------------------------- budget trade-off
t = [r"\begin{tabular}{llrrrr}", r"\hline",
     r"molecule & budget & $M$ & $\Dmax$ & retained dim & $K$ range\\", r"\hline"]
for mol in ("h2o", "n2"):
    for suffix, blabel in CAMPAIGNS:
        sel = [v for (s, m, b, a, c), v in rows.items()
               if s == suffix and m == mol and a == "exhaustive" and c == "NC"]
        if not sel:
            continue
        ks = [v["K"] for v in sel if v["K"] is not None]
        dm = sorted({v["D_max"] for v in sel if v["D_max"] is not None})
        rt = sorted({v["retained"] for v in sel if v["retained"] is not None})
        ms = sorted({v["M"] for v in sel if v["M"] is not None})
        moltex = r"$\mathrm{H_2O}$" if mol == "h2o" else r"$\mathrm{N_2}$"
        t.append(f"{moltex} & ${blabel}$"
                 + f" & {','.join(map(str, ms))} & {','.join(map(str, dm))}"
                 + f" & {','.join(map(str, rt))} & {min(ks)}--{max(ks)}\\\\")
t += [r"\hline", r"\end{tabular}"]
(OUT / "gen_budget_tradeoff.tex").write_text("\n".join(t) + "\n")
print(f"wrote {OUT/'gen_budget_tradeoff.tex'}")
print("\ndone -- scp tables/analysis/gen_*.tex and tables/analysis/data/q4_summary.csv")
