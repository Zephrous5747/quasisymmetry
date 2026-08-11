"""Plot H2O endpoint K/sectors/dim (iterative vs mixed) + print analysis."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

STYLES = {
    ("iterative", "NC"): ("#0072B2", "o", "--", "iterative NC"),
    ("iterative", "variance"): ("#E69F00", "o", ":", "iterative variance"),
    ("mixed_disjoint", "NC"): ("#009E73", "s", "--", "mixed-disjoint NC"),
    ("mixed_disjoint", "variance"): ("#D55E00", "s", ":", "mixed-disjoint variance"),
    ("mixed_overlap", "NC"): ("#56B4E9", "^", "--", "mixed-overlap NC"),
    ("mixed_overlap", "variance"): ("#CC79A7", "^", ":", "mixed-overlap variance"),
}


def _f(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def load_endpoint_latest(path: Path) -> dict[tuple, dict]:
    latest: dict[tuple, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = (f"{_f(row, 'bond'):.10g}", row["method"], row["cost_function"])
            latest[key] = row
    return latest


def _metric_field(row: dict, *keys: str) -> float:
    """Prefer D_max over legacy dim when reading CSV rows."""
    for key in keys:
        raw = (row.get(key) or "").strip()
        if raw == "":
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return math.nan


def plot_endpoint(
    latest: dict[tuple, dict],
    output: Path,
    *,
    methods: tuple[str, ...] = (
        "iterative",
        "mixed_disjoint",
        "mixed_overlap",
    ),
    title: str = "H2O endpoint grid: FCI-ref OO + K (sto-3g)",
) -> None:
    series: dict[tuple[str, str], list] = defaultdict(list)
    for (_b, method, cost), row in latest.items():
        if method in methods:
            series[(method, cost)].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    panels = [
        ("K", r"$K$", ("K",)),
        ("sectors", "sectors", ("sectors", "n_sectors_total")),
        ("D_max", r"$D_{\max}$", ("D_max", "dim")),
    ]
    order = [(m, c) for m in methods for c in ("NC", "variance")]
    for key in order:
        group = series.get(key)
        if not group:
            continue
        group = sorted(group, key=lambda r: _f(r, "bond"))
        color, marker, ls, label = STYLES[key]
        xs = [_f(r, "bond") for r in group]
        for ax, (_field, ylabel, keys) in zip(axes, panels):
            ax.plot(
                xs,
                [_metric_field(r, *keys) for r in group],
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
        ax.set_xlabel("OH bond, A")
    axes[-1].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output}")


def ediff(row: dict) -> float:
    e_d = _f(row, "E_decoupled")
    e_f = _f(row, "E_FCI")
    if math.isnan(e_d) or math.isnan(e_f):
        return math.nan
    return abs(e_d - e_f)


def analyze_endpoint(latest: dict[tuple, dict]) -> None:
    print("\n=== H2O endpoint (latest ok) ===")
    bonds = sorted({float(b) for (b, _, _) in latest})
    print(f"bonds ({len(bonds)}): {bonds}")
    for method in ("mixed_disjoint", "mixed_overlap", "iterative"):
        for cost in ("NC", "variance"):
            rows = [
                latest[k]
                for k in sorted(latest, key=lambda t: float(t[0]))
                if k[1] == method and k[2] == cost
            ]
            ks = [_f(r, "K") for r in rows]
            dims = [_metric_field(r, "D_max", "dim") for r in rows]
            errs = [ediff(r) for r in rows]
            print(
                f"{method:16s} {cost:8s}: "
                f"K={ks}  D_max={dims}  "
                f"|E_dec-E_FCI| max={max(errs):.3e} median={sorted(errs)[len(errs)//2]:.3e}"
            )

    # Head-to-head iterative vs mixed_disjoint on K
    print("\nHead-to-head iterative vs mixed_disjoint (same bond×cost):")
    wins = {"it_K": 0, "md_K": 0, "tie_K": 0, "it_dim": 0, "md_dim": 0, "tie_dim": 0}
    for bond in bonds:
        for cost in ("NC", "variance"):
            it = latest.get((f"{bond:.10g}", "iterative", cost))
            md = latest.get((f"{bond:.10g}", "mixed_disjoint", cost))
            if not it or not md:
                # try alternate float formatting
                it = it or next(
                    (
                        latest[k]
                        for k in latest
                        if abs(float(k[0]) - bond) < 1e-9
                        and k[1] == "iterative"
                        and k[2] == cost
                    ),
                    None,
                )
                md = md or next(
                    (
                        latest[k]
                        for k in latest
                        if abs(float(k[0]) - bond) < 1e-9
                        and k[1] == "mixed_disjoint"
                        and k[2] == cost
                    ),
                    None,
                )
            if not it or not md:
                continue
            ik, mk = _f(it, "K"), _f(md, "K")
            id_, md_ = _metric_field(it, "D_max", "dim"), _metric_field(md, "D_max", "dim")
            if ik < mk:
                wins["it_K"] += 1
            elif mk < ik:
                wins["md_K"] += 1
            else:
                wins["tie_K"] += 1
            if id_ < md_:
                wins["it_dim"] += 1
            elif md_ < id_:
                wins["md_dim"] += 1
            else:
                wins["tie_dim"] += 1
    print(wins)


def analyze_checklist(path: Path) -> None:
    print("\n=== H2O checklist supplement ===")
    last: dict[tuple, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = (f"{_f(row, 'bond'):.10g}", row["cost_function"], row["protocol"])
            last[key] = row
    print(f"unique ok protocols: {len(last)}")
    for key in sorted(last, key=lambda t: (float(t[0]), t[1], t[2])):
        r = last[key]
        print(
            f"  R={float(key[0]):.4g} {key[2]:28s} "
            f"K {r['K_pre']}->{r['K_post']}  "
            f"dim {r['dim_pre']}->{r['dim_post']}  "
            f"cost {_f(r,'cost_before'):.4g}->{_f(r,'cost_after'):.4g}"
        )

    # OO effect: K_pre vs K_post by protocol family
    print("\nOO ΔK (post-pre), by protocol family:")
    fam: dict[str, list[float]] = defaultdict(list)
    for (_b, _c, proto), r in last.items():
        fam[proto].append(_f(r, "K_post") - _f(r, "K_pre"))
    for proto, deltas in sorted(fam.items()):
        print(f"  {proto}: ΔK={deltas}")


def main() -> None:
    latest = load_endpoint_latest(REPO / "tables/h2o/endpoint_grid.csv")
    plot_endpoint(
        latest,
        REPO / "tables/h2o/h2o_endpoint_k_sectors_dim.png",
        methods=("iterative", "mixed_disjoint", "mixed_overlap"),
        title="H2O endpoint: iterative vs mixed (sto-3g)",
    )
    # Classic overlay matching earlier iterative_h2o.png style
    plot_endpoint(
        latest,
        REPO / "tables/h2o/iterative_h2o_endpoint.png",
        methods=("iterative", "mixed_disjoint"),
        title="H2O FCI-ref OO + K: iterative vs mixed-disjoint (sto-3g)",
    )
    analyze_endpoint(latest)
    analyze_checklist(REPO / "tables/h2o/checklist_supplement_manifest.csv")


if __name__ == "__main__":
    main()
