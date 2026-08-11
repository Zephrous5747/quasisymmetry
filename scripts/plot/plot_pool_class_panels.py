"""Generate per-class 3-panel K/sectors/dim plots for H2O and N2.

Classes: iterative, greedy mixed (overlap-allowed), mixed disjoint.
Each figure has NC and variance cost series only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from plot_k_sectors_dim import plot_k_sectors_dim

REPO = Path(__file__).resolve().parents[2]


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    jobs = [
        # H2O
        {
            "csv": REPO / "tables/h2o/fci_oo_metrics_grid.csv",
            "select": "iterative",
            "out": REPO / "tables/h2o/h2o_iterative_k_sectors_dim.png",
            "title": "H2O Iterative FCI-ref OO + K (sto-3g)",
            "xlabel": "OH bond, A",
        },
        {
            "csv": REPO / "tables/h2o/fci_oo_metrics_grid.csv",
            "select": "greedy",
            "out": REPO / "tables/h2o/h2o_greedy_mixed_k_sectors_dim.png",
            "title": "H2O Greedy Mixed FCI-ref OO + K (sto-3g)",
            "xlabel": "OH bond, A",
        },
        {
            "csv": REPO / "tables/h2o/fci_oo_metrics_mixed_disjoint.csv",
            "select": "greedy",
            "out": REPO / "tables/h2o/h2o_mixed_disjoint_k_sectors_dim.png",
            "title": "H2O Mixed Disjoint FCI-ref OO + K (sto-3g)",
            "xlabel": "OH bond, A",
        },
        # N2
        {
            "csv": REPO / "tables/n2/fci_oo_metrics_grid.csv",
            "select": "iterative",
            "out": REPO / "tables/n2/n2_iterative_k_sectors_dim.png",
            "title": "N2 Iterative FCI-ref OO + K (sto-3g)",
            "xlabel": "N--N bond, A",
        },
        {
            "csv": REPO / "tables/n2/fci_oo_metrics_grid.csv",
            "select": "greedy",
            "out": REPO / "tables/n2/n2_greedy_mixed_k_sectors_dim.png",
            "title": "N2 Greedy Mixed FCI-ref OO + K (sto-3g)",
            "xlabel": "N--N bond, A",
        },
        {
            "csv": REPO / "tables/n2/fci_oo_metrics_mixed_disjoint.csv",
            "select": "greedy",
            "out": REPO / "tables/n2/n2_mixed_disjoint_k_sectors_dim.png",
            "title": "N2 Mixed Disjoint FCI-ref OO + K (sto-3g)",
            "xlabel": "N--N bond, A",
        },
    ]

    # Also mirror into tables/analysis for convenience.
    analysis = REPO / "tables/analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        rows = _load(job["csv"])
        plot_k_sectors_dim(
            rows,
            job["out"],
            title=job["title"],
            xlabel=job["xlabel"],
            select=job["select"],
            short_labels=True,
        )
        mirror = analysis / job["out"].name
        plot_k_sectors_dim(
            rows,
            mirror,
            title=job["title"],
            xlabel=job["xlabel"],
            select=job["select"],
            short_labels=True,
        )


if __name__ == "__main__":
    main()
