#!/usr/bin/env python
"""Build document-exact U_full endpoint plots + table data for the TeX update."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from json import JSONDecoder
from pathlib import Path

import matplotlib.pyplot as plt

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


def _load_last(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    dec = JSONDecoder()
    objs: list[dict] = []
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
        raise ValueError(f"no JSON in {path}")
    return objs[-1]


def _bond_tag(tag: str) -> float:
    return float(tag.replace("bond_", "", 1).replace("p", ".", 1))


def collect(mol: str, u_tag: str = "U_full") -> list[dict]:
    root = REPO / "results" / f"{mol}_endpoint_grid"
    rows: list[dict] = []
    for bond_dir in sorted(root.glob("bond_*")):
        bond = _bond_tag(bond_dir.name)
        for method in ("mixed_disjoint", "mixed_overlap", "iterative"):
            for cost in ("NC", "variance"):
                d = bond_dir / u_tag / method / cost
                if not (d / "metrics.json").is_file():
                    continue
                m = _load_last(d / "metrics.json")
                o = _load_last(d / "oo.json") if (d / "oo.json").is_file() else {}
                e_f, e_d = m.get("E_FCI"), m.get("E_decoupled")
                de = (
                    abs(float(e_d) - float(e_f))
                    if e_f is not None and e_d is not None
                    else math.nan
                )
                dim = m.get("D_max")
                if dim is None:
                    dim = m.get("relevant_sectors_total_dim")
                rows.append(
                    {
                        "bond": bond,
                        "method": method,
                        "cost_function": cost,
                        "K": m.get("K"),
                        "dim": dim,
                        "sectors": m.get(
                            "n_sectors_total", m.get("relevant_sectors_count")
                        ),
                        "sectors_rel": m.get("relevant_sectors_count"),
                        "n_exact": m.get("n_exact"),
                        "n_las": m.get("n_las"),
                        "M_eff": m.get("M_eff", o.get("M_eff")),
                        "cost_after": o.get("cost_after"),
                        "cost_before": o.get("cost_before"),
                        "E_decoupled": e_d,
                        "E_FCI": e_f,
                        "ediff": de,
                        "converged": m.get("converged"),
                        "orbital_rotation": o.get("orbital_rotation", u_tag[2:]),
                        "oo_json": str(d / "oo.json"),
                    }
                )
    return rows


def plot_mol(rows: list[dict], mol: str, xlabel: str, title: str, stem: str) -> Path:
    series: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        series[(r["method"], r["cost_function"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    # Third panel: PDF-style D_max = max sector dim (falls back to K-used dim).
    panels = [
        ("K", r"$K$"),
        ("sectors_rel", "sectors (K-used)"),
        ("dim", r"$D_{\mathrm{max}}$"),
    ]
    for key, (color, marker, ls, label) in STYLES.items():
        group = series.get(key)
        if not group:
            continue
        group = sorted(group, key=lambda r: float(r["bond"]))
        xs = [float(r["bond"]) for r in group]
        for ax, (field, ylabel) in zip(axes, panels):
            ys = []
            for r in group:
                v = r.get(field)
                ys.append(float(v) if v is not None and v != "" else math.nan)
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
    out = OUT_IMG / f"{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    # also molecule tables/
    tables_out = REPO / "tables" / mol / f"{mol}_document_exact_full_U_k.png"
    fig.savefig(tables_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    print(f"[ok] {tables_out}")
    return out


def fmt_de_tex(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    v = float(v)
    if v == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / (10**exp)
    return f"${mant:.1f}\\times10^{{{exp}}}$"


def fmt_cost(v) -> str:
    if v is None or v == "":
        return "---"
    v = float(v)
    if abs(v) >= 0.01:
        return f"{v:.3g}"
    return f"{v:.1e}"


def endpoint_table_tex(rows: list[dict], mol_tex: str, cost: str, label: str) -> str:
    by = {(r["bond"], r["method"]): r for r in rows if r["cost_function"] == cost}
    bonds = sorted({r["bond"] for r in rows})
    methods = ["mixed_overlap", "mixed_disjoint", "iterative"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{mol_tex} {cost} endpoints (document PG exact, full-$U$): "
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
            r = by.get((b, m))
            if r is None:
                cells.extend(["---", "---", "---"])
                continue
            cells.append(str(int(r["K"])) if r["K"] is not None else "---")
            cells.append(fmt_cost(r.get("cost_after")))
            cells.append(fmt_de_tex(r.get("ediff")))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    h2o = collect("h2o", "U_full")
    n2 = collect("n2", "U_full")
    assert len(h2o) == 60, len(h2o)
    assert len(n2) == 66, len(n2)

    plot_mol(
        h2o,
        "h2o",
        "OH bond, A",
        r"H$_2$O document-exact + full-$U$: FCI-ref OO + $K$ (sto-3g)",
        "iterative_h2o_document_exact_full",
    )
    plot_mol(
        n2,
        "n2",
        "N--N bond, A",
        r"N$_2$ document-exact + full-$U$: FCI-ref OO + $K$ (sto-3g)",
        "iterative_n2_document_exact_full",
    )

    snippet = "\n".join(
        [
            endpoint_table_tex(h2o, r"$\mathrm{H}_2\mathrm{O}$", "NC", "tab:h2o-nc-doc"),
            endpoint_table_tex(
                h2o, r"$\mathrm{H}_2\mathrm{O}$", "variance", "tab:h2o-var-doc"
            ),
            endpoint_table_tex(n2, r"$\mathrm{N}_2$", "NC", "tab:n2-nc-doc"),
            endpoint_table_tex(n2, r"$\mathrm{N}_2$", "variance", "tab:n2-var-doc"),
        ]
    )
    (OUT_DIR / "endpoint_document_exact_full_U_tables.tex").write_text(
        snippet, encoding="utf-8"
    )

    summary = {
        "n_h2o": len(h2o),
        "n_n2": len(n2),
        "n_exact_h2o": sorted({str(r["n_exact"]) for r in h2o}),
        "n_exact_n2": sorted({str(r["n_exact"]) for r in n2}),
        "h2o_K_sat_261": sum(1 for r in h2o if int(r["K"]) == 261),
        "n2_K_all_3584": all(int(r["K"]) == 3584 for r in n2),
        "h2o_med_ediff": sorted(r["ediff"] for r in h2o)[len(h2o) // 2],
        "n2_med_ediff": sorted(r["ediff"] for r in n2)[len(n2) // 2],
    }
    (OUT_DIR / "endpoint_document_exact_full_U_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
