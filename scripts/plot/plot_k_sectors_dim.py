"""Three-panel bond-scan plot: K, sectors, dim (N2_metrics_LNE style).

Reads a diagnostics CSV with columns:
  bond, select, cost_function, K, sectors, dim, status
and overlays series such as ``NC fci`` / ``variance fci`` (optionally
prefixed by selection name).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COST_STYLES = {
    "NC": ("#009E73", "s", "--"),
    "variance": ("#D55E00", "s", ":"),
}

# Prefer iterative markers when the filtered class is iterative.
ITERATIVE_COST_STYLES = {
    "NC": ("#0072B2", "o", "--"),
    "variance": ("#E69F00", "o", ":"),
}

SERIES_STYLES = {
    ("iterative", "NC"): ITERATIVE_COST_STYLES["NC"],
    ("iterative", "variance"): ITERATIVE_COST_STYLES["variance"],
    ("greedy", "NC"): COST_STYLES["NC"],
    ("greedy", "variance"): COST_STYLES["variance"],
}


def _f(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _dedupe_ok(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the last ok row per (bond, select, cost_function)."""
    last: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if (row.get("status") or "ok") != "ok":
            continue
        key = (
            f"{_f(row, 'bond'):.10g}",
            row.get("select", ""),
            row.get("cost_function", ""),
        )
        last[key] = row
    return list(last.values())


def _series_label(select: str, cost: str, *, short: bool) -> str:
    if short:
        return f"{cost} fci"
    return f"{select} {cost} fci"


def plot_k_sectors_dim(
    rows: list[dict[str, str]],
    output: Path,
    *,
    title: str,
    xlabel: str,
    select: str | None = None,
    short_labels: bool = False,
) -> None:
    ok = _dedupe_ok(rows)
    if select is not None:
        ok = [r for r in ok if r.get("select") == select]
    if not ok:
        raise SystemExit("no ok rows to plot")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        groups[(row.get("select", ""), row.get("cost_function", ""))].append(row)

    # Single-class figures: use short NC/variance labels by default.
    if short_labels or (select is not None and len({k[0] for k in groups}) == 1):
        short = True
    else:
        short = False

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    panels = [
        ("K", r"$K$"),
        ("sectors", "sectors"),
        ("D_max", r"$D_{\max}$"),
    ]

    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: _f(r, "bond"))
        sel, cost = key
        if short and sel == "iterative":
            color, marker, ls = ITERATIVE_COST_STYLES.get(
                cost, ("#333333", "o", "-")
            )
        elif short:
            color, marker, ls = COST_STYLES.get(cost, ("#333333", "o", "-"))
        else:
            color, marker, ls = SERIES_STYLES.get(key, ("#333333", "o", "-"))
        xs = [_f(r, "bond") for r in group]
        label = _series_label(sel, cost, short=short)
        for ax, (field, ylabel) in zip(axes, panels):
            if field == "D_max":
                ys = [
                    _f(r, "D_max") if (r.get("D_max") or "").strip() else _f(r, "dim")
                    for r in group
                ]
            else:
                ys = [_f(r, field) for r in group]
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
    axes[-1].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="OO+metrics diagnostics CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="PNG path (default: <csv_stem>_k_sectors_dim.png)",
    )
    parser.add_argument("--title", default="FCI-referenced OO + metrics")
    parser.add_argument(
        "--xlabel",
        default="bond, A",
        help="x-axis label",
    )
    parser.add_argument(
        "--select",
        default=None,
        choices=("greedy", "iterative"),
        help="Keep only this selection method",
    )
    parser.add_argument(
        "--short-labels",
        action="store_true",
        help="Legend as 'NC fci' / 'variance fci' (no select prefix)",
    )
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = Path(args.output) if args.output else path.with_name(
        path.stem + "_k_sectors_dim.png"
    )
    plot_k_sectors_dim(
        rows,
        out,
        title=args.title,
        xlabel=args.xlabel,
        select=args.select,
        short_labels=args.short_labels,
    )


if __name__ == "__main__":
    main()
