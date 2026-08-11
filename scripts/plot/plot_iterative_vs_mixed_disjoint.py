"""Three-panel K/sectors/dim: iterative vs mixed-disjoint (NC + variance).

Overwrites tables/{h2o,n2}/iterative_{h2o,n2}.png with four series:
  iterative NC/variance fci, mixed NC/variance fci (disjoint Mixed pool).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

STYLES = {
    ("iterative", "NC"): ("#0072B2", "o", "--", "iterative NC fci"),
    ("iterative", "variance"): ("#E69F00", "o", ":", "iterative variance fci"),
    ("mixed", "NC"): ("#009E73", "s", "--", "mixed NC fci"),
    ("mixed", "variance"): ("#D55E00", "s", ":", "mixed variance fci"),
}


def _f(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _load_ok(path: Path, select: str | None = None) -> list[dict[str, str]]:
    last: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("status") or "ok") != "ok":
                continue
            if select is not None and row.get("select") != select:
                continue
            key = (
                f"{_f(row, 'bond'):.10g}",
                row.get("select", ""),
                row.get("cost_function", ""),
            )
            last[key] = row
    return list(last.values())


def _plot(
    series: dict[tuple[str, str], list[dict[str, str]]],
    output: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    panels = [("K", r"$K$"), ("sectors", "sectors"), ("D_max", r"$D_{\max}$")]
    for key in (("iterative", "NC"), ("iterative", "variance"), ("mixed", "NC"), ("mixed", "variance")):
        group = series.get(key)
        if not group:
            continue
        group = sorted(group, key=lambda r: _f(r, "bond"))
        color, marker, ls, label = STYLES[key]
        xs = [_f(r, "bond") for r in group]
        for ax, (field, ylabel) in zip(axes, panels):
            if field == "D_max":
                ys = [
                    _f(r, "D_max") if not math.isnan(_f(r, "D_max")) else _f(r, "dim")
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


def _combine(
    iterative_csv: Path,
    mixed_csv: Path,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _load_ok(iterative_csv, select="iterative"):
        out[("iterative", row["cost_function"])].append(row)
    for row in _load_ok(mixed_csv, select="greedy"):
        out[("mixed", row["cost_function"])].append(row)
    return out


def main() -> None:
    jobs = [
        {
            "iter": REPO / "tables/h2o/fci_oo_metrics_grid.csv",
            "mixed": REPO / "tables/h2o/fci_oo_metrics_mixed_disjoint.csv",
            "out": REPO / "tables/h2o/iterative_h2o.png",
            "title": "H2O FCI-ref OO + K (sto-3g)",
            "xlabel": "OH bond, A",
        },
        {
            "iter": REPO / "tables/n2/fci_oo_metrics_grid.csv",
            "mixed": REPO / "tables/n2/fci_oo_metrics_mixed_disjoint.csv",
            "out": REPO / "tables/n2/iterative_n2.png",
            "title": "N2 FCI-ref OO + K (sto-3g)",
            "xlabel": "N--N bond, A",
        },
    ]
    for job in jobs:
        series = _combine(job["iter"], job["mixed"])
        _plot(series, job["out"], title=job["title"], xlabel=job["xlabel"])


if __name__ == "__main__":
    main()
