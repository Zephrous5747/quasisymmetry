"""Three-panel diagnostics plot for H2O FCI selection results.

Panels (Quasi_Symmetries style):
  1. pool / OO cost function
  2. |E_dec - E_FCI|
  3. coupled-sector dimension K

Reads the CSV written by ``trillium_h2o_fci_diagnostics.sh``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SERIES_STYLES = {
    ("greedy", "NC"): ("#0072B2", "s", "--"),
    ("greedy", "variance"): ("#56B4E9", "s", "--"),
    ("iterative", "NC"): ("#D55E00", "o", "-"),
    ("iterative", "variance"): ("#E69F00", "o", "-"),
}


def _f(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _label(row: dict[str, str]) -> str:
    select = row.get("select", "?")
    cost = row.get("cost_function", "?")
    extras = []
    if select == "greedy":
        ns, nq = row.get("n_singles"), row.get("n_quartets")
        if ns and nq:
            extras.append(f"s{ns}q{nq}")
    else:
        nsym = row.get("n_sym")
        if nsym:
            extras.append(f"n{nsym}")
    suffix = f" ({','.join(extras)})" if extras else ""
    return f"{select}\n{cost}{suffix}"


def plot_three_panels(rows: list[dict[str, str]], output: Path, title: str) -> None:
    if not rows:
        raise SystemExit("no diagnostic rows to plot")

    # Prefer geometry axis when multiple OH lengths are present; else categorical.
    geoms = [ _f(r, "geometry_param") for r in rows ]
    use_geom = (
        sum(1 for g in geoms if math.isfinite(g)) >= 2
        and len({round(g, 8) for g in geoms if math.isfinite(g)}) >= 2
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=not use_geom)
    panels = [
        ("cost", r"Cost function", True),
        ("edec_error", r"$|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
        ("K", r"Coupled-sector dimension $K$", False),
    ]

    if use_geom:
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (row.get("select", ""), row.get("cost_function", ""))
            groups.setdefault(key, []).append(row)
        for key, group in groups.items():
            group = sorted(group, key=lambda r: _f(r, "geometry_param"))
            color, marker, ls = SERIES_STYLES.get(key, ("#333333", "o", "-"))
            xs = [_f(r, "geometry_param") for r in group]
            series = {
                "cost": [_f(r, "cost") for r in group],
                "edec_error": [_f(r, "edec_error") for r in group],
                "K": [_f(r, "K") for r in group],
            }
            label = f"{key[0]} {key[1]}"
            for ax, (field, ylabel, use_log) in zip(axes, panels):
                ys = series[field]
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
                if use_log:
                    positive = [y for y in ys if y > 0 and math.isfinite(y)]
                    if positive:
                        ax.set_yscale("log")
        for ax in axes:
            ax.set_xlabel(r"OH bond length ($\AA$)")
        axes[0].legend(loc="best", frameon=False, fontsize=8)
    else:
        labels = [_label(r) for r in rows]
        xs = list(range(len(rows)))
        colors = [
            SERIES_STYLES.get(
                (r.get("select", ""), r.get("cost_function", "")),
                ("#333333", "o", "-"),
            )[0]
            for r in rows
        ]
        series = {
            "cost": [_f(r, "cost") for r in rows],
            "edec_error": [_f(r, "edec_error") for r in rows],
            "K": [_f(r, "K") for r in rows],
        }
        for ax, (field, ylabel, use_log) in zip(axes, panels):
            ys = series[field]
            ax.bar(xs, ys, color=colors, width=0.65)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25, axis="y")
            if use_log:
                positive = [y for y in ys if y > 0 and math.isfinite(y)]
                if positive:
                    ax.set_yscale("log")
        axes[1].set_xlabel("FCI-referenced selection run")

    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="tables/h2o/fci_select_diagnostics.csv",
        help="diagnostics CSV from the FCI metrics sbatch",
    )
    parser.add_argument(
        "--output",
        default="tables/h2o/fci_select_diagnostics_three_panel.png",
        help="PNG path",
    )
    parser.add_argument(
        "--title",
        default="H2O FCI selection diagnostics",
        help="figure title",
    )
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("status") or "ok") == "ok"]
    plot_three_panels(rows, Path(args.output), args.title)


if __name__ == "__main__":
    main()
