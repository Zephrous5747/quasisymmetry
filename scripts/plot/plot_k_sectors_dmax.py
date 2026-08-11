"""Three-panel bond-scan plots: K, sectors, D_max — post-fix endpoint grid.

Replaces the older ``plot_k_sectors_dim.py`` figures in two ways:

  * reads ``results/*/bond_*/U_*/<method>/<cost>/metrics.json`` directly rather
    than the append-only endpoint_grid.csv, which interleaves the n_exact=1 and
    document-exact campaigns and cannot be filtered by era reliably;
  * third panel is ``D_max`` (largest single retained sector) instead of
    ``dim`` (= relevant_sectors_total_dim). D_max is the quantity the PDF
    tabulates and the one that bounds the block a solver actually has to
    diagonalise; ``dim`` merely counts everything the K search touched.

Only post-fix points are plotted; a pre-fix metrics.json (no
``reference_weight_ok`` / ``exact_sector_source`` key) is skipped and reported.

Usage
-----
    python scripts/plot/plot_k_sectors_dmax.py                 # irrep, both mols
    python scripts/plot/plot_k_sectors_dmax.py --packing full
    python scripts/plot/plot_k_sectors_dmax.py --compare-methods
"""

from __future__ import annotations

import argparse
import glob
import json
from json import JSONDecoder
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

COST_STYLES = {
    "NC": ("#009E73", "s", "--"),
    "variance": ("#D55E00", "s", ":"),
}
ITERATIVE_COST_STYLES = {
    "NC": ("#0072B2", "o", "--"),
    "variance": ("#E69F00", "o", ":"),
}
METHOD_LABEL = {
    "mixed_disjoint": "Greedy Mixed (disjoint)",
    "mixed_overlap": "Greedy Mixed (overlap)",
    "iterative": "Iterative NC-ranked LAS",
}
METHOD_STYLE = {
    "mixed_disjoint": ("#009E73", "s", "--"),
    "mixed_overlap": ("#56B4E9", "^", "-."),
    "iterative": ("#0072B2", "o", "-"),
}
MOL_LABEL = {"h2o": "H$_2$O", "n2": "N$_2$"}
XLABEL = {"h2o": "OH bond, $\\AA$", "n2": "NN bond, $\\AA$"}


def load_stream(path: Path) -> dict | None:
    """Last JSON object in a file that may hold concatenated records."""
    text = path.read_text()
    dec = JSONDecoder()
    i, obj = 0, None
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
    return obj


def collect(packing: str) -> tuple[dict, list[str]]:
    """{(mol, method, cost): [(bond, K, sectors, D_max, converged), ...]}"""
    data: dict[tuple[str, str, str], list] = {}
    skipped: list[str] = []
    pattern = f"results/*_endpoint_grid/bond_*/U_{packing}/*/*/metrics.json"
    for f in sorted(glob.glob(str(REPO / pattern))):
        p = Path(f).parts
        mol = p[-6].split("_")[0]
        bond = float(p[-5].replace("bond_", "").replace("p", "."))
        method, cost = p[-3], p[-2]
        try:
            d = load_stream(Path(f))
        except Exception:  # noqa: BLE001
            skipped.append(f)
            continue
        if d is None:
            skipped.append(f)
            continue
        if "reference_weight_ok" not in d and "exact_sector_source" not in d:
            skipped.append(f"{f}  (pre-fix)")
            continue
        data.setdefault((mol, method, cost), []).append(
            (bond, d.get("K"), d.get("relevant_sectors_count"),
             d.get("D_max"), bool(d.get("converged")),
             d.get("relevant_sectors_total_dim"))
        )
    for v in data.values():
        v.sort(key=lambda t: t[0])
    return data, skipped


def three_panel(series, title, xlabel, out_path, with_dim=False):
    """series: list of (label, (color, marker, ls), rows)."""
    panels = [("$K$", 1), ("sectors", 2), ("$D_{\\max}$", 3)]
    if with_dim:
        # D_max is fixed by the exact sector + partition (21 for H2O, 120 for
        # N2) and carries no bond or method dependence, so on its own the third
        # panel is flat. dim = relevant_sectors_total_dim is what actually moves.
        panels.append(("dim (retained)", 5))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.35 * len(panels), 4.1))
    for ax, (ylabel, idx) in zip(axes, panels):
        for label, (color, marker, ls) in series:
            rows = series_rows[label]
            x = [r[0] for r in rows]
            y = [r[idx] for r in rows]
            ax.plot(x, y, marker=marker, linestyle=ls, color=color,
                    label=label, markersize=6, linewidth=1.6)
            # ring any non-converged point so it cannot be read as a datum
            bad = [(r[0], r[idx]) for r in rows if not r[4]]
            if bad:
                ax.scatter([b[0] for b in bad], [b[1] for b in bad],
                           s=140, facecolors="none", edgecolors="red",
                           linewidths=1.4, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packing", default="irrep", choices=("irrep", "full"),
                    help="irrep is the sound arm: full-U breaks the PG taper")
    ap.add_argument("--with-dim", action="store_true",
                    help="add a 4th panel with relevant_sectors_total_dim")
    ap.add_argument("--compare-methods", action="store_true",
                    help="also emit one figure per cost overlaying the methods")
    args = ap.parse_args()

    data, skipped = collect(args.packing)
    if skipped:
        print(f"[plot] skipped {len(skipped)} non-post-fix / unreadable point(s)")
        for s in skipped[:8]:
            print("   ", s)

    global series_rows

    # --- per method: NC vs variance (the layout of the old figures) ----------
    for mol in ("h2o", "n2"):
        for method in ("mixed_disjoint", "mixed_overlap", "iterative"):
            styles = (ITERATIVE_COST_STYLES if method == "iterative"
                      else COST_STYLES)
            series, series_rows = [], {}
            for cost in ("NC", "variance"):
                rows = data.get((mol, method, cost))
                if not rows:
                    continue
                label = f"{cost} fci"
                series.append((label, styles[cost]))
                series_rows[label] = rows
            if not series:
                continue
            three_panel(
                series,
                f"{MOL_LABEL[mol]} {METHOD_LABEL[method]} FCI-ref OO + K "
                f"(sto-3g, U={args.packing})",
                XLABEL[mol],
                REPO / "tables" / mol /
                f"{mol}_{method}_k_sectors_dmax_{args.packing}.png",
                with_dim=args.with_dim,
            )

    # --- optional: methods overlaid at fixed cost ---------------------------
    if args.compare_methods:
        for mol in ("h2o", "n2"):
            for cost in ("NC", "variance"):
                series, series_rows = [], {}
                for method in ("mixed_disjoint", "mixed_overlap", "iterative"):
                    rows = data.get((mol, method, cost))
                    if not rows:
                        continue
                    label = METHOD_LABEL[method]
                    series.append((label, METHOD_STYLE[method]))
                    series_rows[label] = rows
                if not series:
                    continue
                three_panel(
                    series,
                    f"{MOL_LABEL[mol]} method comparison, {cost} cost "
                    f"(sto-3g, U={args.packing})",
                    XLABEL[mol],
                    REPO / "tables" / mol /
                    f"{mol}_methods_{cost}_k_sectors_dmax_{args.packing}.png",
                    with_dim=args.with_dim,
                )


series_rows: dict[str, list] = {}

if __name__ == "__main__":
    main()
