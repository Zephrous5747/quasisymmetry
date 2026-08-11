#!/usr/bin/env python
"""Rebuild endpoint plots + LaTeX table snippets from synced endpoint_grid CSVs.

Writes:
  tables/analysis/Results/images/iterative_pool/iterative_{n2,h2o}.png
  tables/analysis/endpoint_tables_snippet.tex
  tables/analysis/endpoint_refresh_summary.json
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_IMG = REPO / "tables" / "analysis" / "Results" / "images" / "iterative_pool"
OUT_DIR = REPO / "tables" / "analysis"

STYLES = {
    ("iterative", "NC"): ("#0072B2", "o", "--", "iterative NC"),
    ("iterative", "variance"): ("#E69F00", "o", ":", "iterative variance"),
    ("mixed_disjoint", "NC"): ("#009E73", "s", "--", "mixed-disjoint NC"),
    ("mixed_disjoint", "variance"): ("#D55E00", "s", ":", "mixed-disjoint variance"),
    ("mixed_overlap", "NC"): ("#56B4E9", "^", "--", "mixed-overlap NC"),
    ("mixed_overlap", "variance"): ("#CC79A7", "^", ":", "mixed-overlap variance"),
}


def _f(row: dict, key: str) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def load_latest(path: Path) -> dict[tuple, dict]:
    latest: dict[tuple, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = (_f(row, "bond"), row["method"], row["cost_function"])
            latest[key] = row
    return latest


def ediff(row: dict) -> float:
    e_d, e_f = _f(row, "E_decoupled"), _f(row, "E_FCI")
    if math.isnan(e_d) or math.isnan(e_f):
        return math.nan
    return abs(e_d - e_f)


def fmt_cost(v: float) -> str:
    if math.isnan(v):
        return "---"
    if abs(v) >= 0.01:
        return f"{v:.3g}"
    return f"{v:.1e}"


def fmt_de(v: float) -> str:
    if math.isnan(v):
        return "---"
    return f"{v:.1e}".replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{+")
    # fallback simpler:
    # return f"{v:.1e}"


def fmt_de_tex(v: float) -> str:
    if math.isnan(v):
        return "---"
    exp = int(math.floor(math.log10(abs(v)))) if v != 0 else 0
    mant = v / (10**exp)
    return f"${mant:.1f}\\times10^{{{exp}}}$"


def fmt_k(v: float) -> str:
    if math.isnan(v):
        return "---"
    return str(int(round(v)))


def plot_mol(latest: dict, mol: str, xlabel: str, title: str) -> Path:
    series: dict[tuple[str, str], list] = defaultdict(list)
    for (_b, method, cost), row in latest.items():
        series[(method, cost)].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    panels = [("K", r"$K$"), ("sectors", "sectors"), ("D_max", r"$D_{\max}$")]
    order = list(STYLES.keys())
    for key in order:
        group = series.get(key)
        if not group:
            continue
        group = sorted(group, key=lambda r: _f(r, "bond"))
        color, marker, ls, label = STYLES[key]
        xs = [_f(r, "bond") for r in group]
        for ax, (field, ylabel) in zip(axes, panels):
            ys = [
                _f(r, field) if field != "D_max" else (_f(r, "D_max") if not math.isnan(_f(r, "D_max")) else _f(r, "dim"))
                for r in group
            ]
            ax.plot(
                xs,
                ys,
                marker=marker,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                markersize=5.5,
                label=label,
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[-1].legend(loc="best", frameon=False, fontsize=7.5)
    fig.suptitle(title)
    fig.tight_layout()
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    out = OUT_IMG / f"iterative_{mol}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")
    return out


def generators_from_oo(oo_path: str | Path) -> str:
    path = Path(oo_path)
    # Cluster paths won't exist locally; try remapped results path.
    local = None
    s = str(path).replace("\\", "/")
    if "results/" in s:
        local = REPO / s[s.index("results/") :]
    elif path.is_file():
        local = path
    if local is None or not local.is_file():
        return "(OO JSON not synced)"
    raw = local.read_text(encoding="utf-8", errors="replace")
    # last JSON object
    from json import JSONDecoder

    dec = JSONDecoder()
    objs = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] != "{":
            break
        obj, end = dec.raw_decode(raw, i)
        objs.append(obj)
        i = end
    if not objs:
        return "(empty OO)"
    oo = objs[-1]
    orbs = oo.get("accumulated_orbitals")
    if orbs:
        parts = []
        for support in orbs:
            support = list(map(int, support))
            if len(support) == 1:
                parts.append(f"Z_{{{support[0]}}}")
            elif len(support) == 2:
                parts.append(f"Z_{{{support[0]}}}Z_{{{support[1]}}}")
            else:
                parts.append("".join(f"Z_{{{p}}}" for p in support))
        return ", ".join(parts)
    singles = oo.get("singles") or []
    quarts = oo.get("quartets") or []
    parts = [f"Z_{{{int(s)}}}" for s in singles]
    parts += [f"Z_{{{a}}}Z_{{{b}}}" for a, b in quarts]
    return ", ".join(parts) if parts else "(none)"


def endpoint_table_tex(latest: dict, mol: str, cost: str, label: str) -> str:
    methods = ["mixed_overlap", "mixed_disjoint", "iterative"]
    bonds = sorted({b for (b, m, c) in latest if c == cost})
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{mol} {cost} endpoints (synced {{{{provisional}}}}): "
        r"$K$, post-OO cost, $|\Delta E_{\mathrm{dec}}|$.}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c ccc ccc ccc}",
        r"\toprule",
        r"$R$ & \multicolumn{3}{c}{Mixed overlap} & \multicolumn{3}{c}{Mixed disjoint}"
        r" & \multicolumn{3}{c}{Iterative} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r" & $K$ & cost & $|\Delta E|$ & $K$ & cost & $|\Delta E|$"
        r" & $K$ & cost & $|\Delta E|$ \\",
        r"\midrule",
    ]
    for b in bonds:
        cells = [f"{b:g}"]
        for m in methods:
            r = latest.get((b, m, cost))
            if r is None:
                cells.extend(["---", "---", "---"])
                continue
            cells.append(fmt_k(_f(r, "K")))
            cells.append(fmt_cost(_f(r, "cost_after")))
            cells.append(fmt_de_tex(ediff(r)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ]
    # fix caption braces from f-string mess
    text = "\n".join(lines)
    text = text.replace("{{provisional}}", "provisional")
    return text


def nc_vs_var(latest: dict, method: str) -> tuple[int, int, int]:
    bonds = sorted({b for (b, m, c) in latest if m == method})
    nc_w = var_w = ties = 0
    for b in bonds:
        a = latest.get((b, method, "NC"))
        v = latest.get((b, method, "variance"))
        if a is None or v is None:
            continue
        ka, da = _f(a, "K"), (_f(a, "D_max") if not math.isnan(_f(a, "D_max")) else _f(a, "dim"))
        kv, dv = _f(v, "K"), (_f(v, "D_max") if not math.isnan(_f(v, "D_max")) else _f(v, "dim"))
        if math.isnan(ka) or math.isnan(kv):
            continue
        if (ka, da) < (kv, dv):
            nc_w += 1
        elif (kv, dv) < (ka, da):
            var_w += 1
        else:
            ties += 1
    return nc_w, var_w, ties


def load_checklist_latest(path: Path) -> dict[tuple, dict]:
    latest = {}
    if not path.is_file():
        return latest
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = (
                _f(row, "bond"),
                row.get("cost_function", ""),
                row.get("protocol", ""),
            )
            latest[key] = row
    return latest


def meff_table(latest_h2o, latest_n2) -> str:
    rows = []
    for mol, L, reps, M in [
        ("H2O", latest_h2o, [0.958, 1.8146666667, 2.5], 5),
        ("N2", latest_n2, [1.2, 1.8, 2.2], 7),
    ]:
        for b in reps:
            for method, label in [
                ("mixed_disjoint", "disjoint Mixed"),
                ("iterative", "Iterative"),
            ]:
                r = L.get((b, method, "NC")) or L.get((b, method, "variance"))
                if not r:
                    continue
                meff = r.get("M_eff") or ""
                n_ex = r.get("n_exact") or ""
                rows.append(
                    (
                        f"{mol} {label}, $R={b:g}$",
                        str(M),
                        str(meff) if meff != "" else "?",
                        str(n_ex) if n_ex != "" else "?",
                    )
                )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Raw nominal $M$ vs $M_{\mathrm{eff}}$ and reported $n_{\mathrm{exact}}$ "
        r"(synced endpoints; $E=\mathrm{span}\{\mathbf 1\}$ in this snapshot).}",
        r"\label{tab:meff}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Pool & Nominal $M$ & $M_{\mathrm{eff}}$ & $n_{\mathrm{exact}}$ \\",
        r"\midrule",
    ]
    for name, M, meff, nex in rows:
        lines.append(f"{name} & {M} & {meff} & {nex} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    h2o = load_latest(REPO / "tables/h2o/endpoint_grid.csv")
    n2 = load_latest(REPO / "tables/n2/endpoint_grid.csv")
    plot_mol(
        h2o,
        "h2o",
        "OH bond, A",
        r"H$_2$O endpoint grid (synced): FCI-ref OO + $K$ (sto-3g)",
    )
    plot_mol(
        n2,
        "n2",
        "N--N bond, A",
        r"N$_2$ endpoint grid (synced): FCI-ref OO + $K$ (sto-3g)",
    )

    snippet = []
    snippet.append(meff_table(h2o, n2))
    snippet.append(
        endpoint_table_tex(h2o, r"$\mathrm{H}_2\mathrm{O}$", "NC", "tab:h2o-nc")
    )
    snippet.append(
        endpoint_table_tex(h2o, r"$\mathrm{H}_2\mathrm{O}$", "variance", "tab:h2o-var")
    )
    snippet.append(endpoint_table_tex(n2, r"$\mathrm{N}_2$", "NC", "tab:n2-nc"))
    snippet.append(
        endpoint_table_tex(n2, r"$\mathrm{N}_2$", "variance", "tab:n2-var")
    )
    (OUT_DIR / "endpoint_tables_snippet.tex").write_text(
        "\n".join(snippet), encoding="utf-8"
    )

    # Generators at representative bonds
    gens = {"h2o": {}, "n2": {}}
    for mol, L, bonds in [
        ("h2o", h2o, [0.958, 1.8146666667, 2.5]),
        ("n2", n2, [1.2, 1.8, 2.2]),
    ]:
        for b in bonds:
            gens[mol][f"{b:g}"] = {}
            for method in ["mixed_overlap", "mixed_disjoint", "iterative"]:
                r = L.get((b, method, "NC"))
                if r and r.get("oo_json"):
                    gens[mol][f"{b:g}"][method] = generators_from_oo(r["oo_json"])
                else:
                    gens[mol][f"{b:g}"][method] = "(missing)"

    chk_h2o = load_checklist_latest(
        REPO / "tables/h2o/checklist_supplement_manifest.csv"
    )
    chk_n2 = load_checklist_latest(
        REPO / "tables/n2/checklist_supplement_manifest.csv"
    )

    def chk_summary(chk):
        # latest by (bond, cost, protocol family)
        out = []
        for (b, cost, proto), r in sorted(chk.items()):
            out.append(
                {
                    "bond": b,
                    "cost": cost,
                    "protocol": proto,
                    "K_pre": r.get("K_pre"),
                    "K_post": r.get("K_post"),
                    "dim_pre": r.get("dim_pre"),
                    "dim_post": r.get("dim_post"),
                    "n_oo_checkpoints": r.get("n_oo_checkpoints"),
                }
            )
        return out

    summary = {
        "n_h2o_ok": len(h2o),
        "n_n2_ok": len(n2),
        "h2o_methods": Counter(m for _, m, _ in h2o),
        "n2_methods": Counter(m for _, m, _ in n2),
        "n_exact_values_h2o": sorted(
            {r.get("n_exact") for r in h2o.values() if r.get("n_exact")}
        ),
        "n_exact_values_n2": sorted(
            {r.get("n_exact") for r in n2.values() if r.get("n_exact")}
        ),
        "nc_vs_var": {
            "h2o_mixed_disjoint": nc_vs_var(h2o, "mixed_disjoint"),
            "h2o_iterative": nc_vs_var(h2o, "iterative"),
            "n2_mixed_disjoint": nc_vs_var(n2, "mixed_disjoint"),
            "n2_iterative": nc_vs_var(n2, "iterative"),
        },
        "generators_nc": gens,
        "checklist_h2o": chk_summary(chk_h2o),
        "checklist_n2": chk_summary(chk_n2),
        "note": (
            "Synced snapshot uses n_exact=1 (all-ones E). "
            "Document PG exact + irrep U rerun will supersede."
        ),
    }
    # jsonify Counters
    summary["h2o_methods"] = dict(summary["h2o_methods"])
    summary["n2_methods"] = dict(summary["n2_methods"])
    (OUT_DIR / "endpoint_refresh_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[ok] wrote summary + snippets")
    print("nc_vs_var", summary["nc_vs_var"])
    print("n_exact", summary["n_exact_values_h2o"], summary["n_exact_values_n2"])


if __name__ == "__main__":
    main()
