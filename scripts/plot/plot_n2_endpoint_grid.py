"""N2 endpoint/checklist analysis + K/sectors/dim plot + OO convergence check."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
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


def _f(row: dict, key: str) -> float:
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


def plot_endpoint(latest: dict[tuple, dict], output: Path) -> None:
    series: dict[tuple[str, str], list] = defaultdict(list)
    for (_b, method, cost), row in latest.items():
        series[(method, cost)].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    panels = [("K", r"$K$"), ("sectors", "sectors"), ("D_max", r"$D_{\max}$")]
    order = [
        ("iterative", "NC"),
        ("iterative", "variance"),
        ("mixed_disjoint", "NC"),
        ("mixed_disjoint", "variance"),
        ("mixed_overlap", "NC"),
        ("mixed_overlap", "variance"),
    ]
    for key in order:
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
        ax.set_xlabel("N--N bond, A")
    axes[-1].legend(loc="best", frameon=False, fontsize=7.5)
    fig.suptitle("N2 endpoint grid: FCI-ref OO + K (sto-3g)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output}")


def ediff(row: dict) -> float:
    e_d, e_f = _f(row, "E_decoupled"), _f(row, "E_FCI")
    if math.isnan(e_d) or math.isnan(e_f):
        return math.nan
    return abs(e_d - e_f)


def analyze_endpoint(latest: dict[tuple, dict]) -> dict:
    bonds = sorted({float(b) for (b, _, _) in latest})
    print("\n=== N2 endpoint (latest ok) ===")
    print(f"bonds ({len(bonds)}): {bonds}")
    summary = {"series": {}, "wins": Counter(), "bonds": bonds}
    for method in ("mixed_disjoint", "mixed_overlap", "iterative"):
        for cost in ("NC", "variance"):
            rows = [
                latest[k]
                for k in sorted(latest, key=lambda t: float(t[0]))
                if k[1] == method and k[2] == cost
            ]
            ks = [_f(r, "K") for r in rows]
            dims = [
                _f(r, "D_max") if not math.isnan(_f(r, "D_max")) else _f(r, "dim")
                for r in rows
            ]
            errs = [ediff(r) for r in rows]
            costs = [_f(r, "cost_after") for r in rows]
            summary["series"][f"{method}_{cost}"] = {
                "K": ks,
                "D_max": dims,
                "dim": dims,
                "err": errs,
                "cost_after": costs,
            }
            print(
                f"{method:16s} {cost:8s}: K={ks}\n"
                f"{'':25s} D_max={dims}\n"
                f"{'':25s} |dE| max={max(errs):.3e} med={sorted(errs)[len(errs)//2]:.3e} "
                f"cost_after med={sorted(costs)[len(costs)//2]:.4g}"
            )

    # iterative vs mixed_disjoint head-to-head
    for bond in bonds:
        for cost in ("NC", "variance"):
            def find(method: str):
                for k, r in latest.items():
                    if abs(float(k[0]) - bond) < 1e-9 and k[1] == method and k[2] == cost:
                        return r
                return None

            it, md, mo = find("iterative"), find("mixed_disjoint"), find("mixed_overlap")
            if it and md:
                ik, mk = _f(it, "K"), _f(md, "K")
                if ik < mk:
                    summary["wins"]["it_vs_md_K"] += 1
                elif mk < ik:
                    summary["wins"]["md_vs_it_K"] += 1
                else:
                    summary["wins"]["tie_it_md_K"] += 1
                ie, me = ediff(it), ediff(md)
                if ie < me:
                    summary["wins"]["it_vs_md_E"] += 1
                elif me < ie:
                    summary["wins"]["md_vs_it_E"] += 1
                else:
                    summary["wins"]["tie_it_md_E"] += 1
            if it and mo:
                ik, ok = _f(it, "K"), _f(mo, "K")
                if ik < ok:
                    summary["wins"]["it_vs_mo_K"] += 1
                elif ok < ik:
                    summary["wins"]["mo_vs_it_K"] += 1
                else:
                    summary["wins"]["tie_it_mo_K"] += 1
    print("wins:", dict(summary["wins"]))
    return summary


def analyze_checklist(path: Path) -> list[dict]:
    print("\n=== N2 checklist ===")
    last: dict[tuple, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (
                f"{_f(row, 'bond'):.10g}",
                row["cost_function"],
                row["protocol"],
            )
            last[key] = row
    rows_out = []
    for key in sorted(last, key=lambda t: (float(t[0]), t[1], t[2])):
        r = last[key]
        rows_out.append(r)
        print(
            f"  R={float(key[0]):.2f} {key[2]:28s} status={r['status']:4s} "
            f"K {r.get('K_pre','')}->{r.get('K_post','')}  "
            f"dim {r.get('dim_pre','')}->{r.get('dim_post','')}  "
            f"cost {_f(r,'cost_before'):.4g}->{_f(r,'cost_after'):.4g}"
        )
    return rows_out


def inspect_oo_convergence() -> list[dict]:
    """Check iterative OO JSON/logs for maxiter hits vs true convergence."""
    print("\n=== OO convergence (iterative) ===")
    records = []
    roots = [
        REPO / "results/n2_endpoint_grid",
        REPO / "results/n2_checklist_supplement",
    ]
    for root in roots:
        for oo in sorted(root.rglob("**/iterative/*/oo.json")) + sorted(
            root.rglob("**/iterative_*_oo.json")
        ):
            try:
                data = json.loads(oo.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {oo}: {exc}")
                continue
            # gather round info if present
            hist = data.get("selection_history") or data.get("history") or {}
            rounds = hist.get("rounds") if isinstance(hist, dict) else None
            if rounds is None:
                rounds = data.get("rounds") or []
            msg = str(data.get("message") or "")
            success = data.get("success")
            nit = data.get("nit")
            nfev = data.get("nfev")
            maxiter_hit = "ITERATION" in msg.upper() or "LIMIT" in msg.upper()
            # also scan nested round results
            round_summaries = []
            if isinstance(rounds, list):
                for i, rnd in enumerate(rounds):
                    if not isinstance(rnd, dict):
                        continue
                    # various shapes
                    opt = rnd.get("optimization") or rnd.get("oo") or rnd
                    rmsg = str(opt.get("message") or rnd.get("message") or "")
                    rnit = opt.get("nit", rnd.get("nit"))
                    rsucc = opt.get("converged", opt.get("success", rnd.get("success")))
                    round_summaries.append(
                        {
                            "i": i,
                            "nit": rnit,
                            "success": rsucc,
                            "message": rmsg[:80],
                            "cost_before": opt.get("cost_before", rnd.get("cost_before")),
                            "cost_after": opt.get("cost_after", rnd.get("cost_after")),
                            "elapsed": opt.get("elapsed", rnd.get("elapsed")),
                        }
                    )
                    if "ITERATION" in rmsg.upper() or "LIMIT" in rmsg.upper():
                        maxiter_hit = True
                    if rsucc is False:
                        maxiter_hit = maxiter_hit or True

            # oo_trace length as proxy for L-BFGS steps
            trace_path = Path(str(oo) + ".oo_trace.json")
            n_trace = None
            if not trace_path.exists():
                alt = oo.with_suffix(oo.suffix + ".oo_trace.json")
                if alt.exists():
                    trace_path = alt
            if trace_path.exists():
                try:
                    tr = json.loads(trace_path.read_text(encoding="utf-8"))
                    n_trace = len(tr) if isinstance(tr, list) else None
                except Exception:
                    pass

            # parse log for maxiter / oo_ref / caps
            log_path = Path(str(oo) + ".log")
            if not log_path.exists():
                log_path = oo.with_suffix(oo.suffix + ".log")
            log_snip = ""
            caps = {}
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if "max_macro" in line or "maxiter" in line.lower() or "oo_ref" in line:
                        log_snip += line.strip()[:120] + " | "
                    if "STOP:" in line or "CONVERGENCE" in line.upper() or "iterations" in line.lower():
                        if "nit" in line.lower() or "STOP" in line or "CONVERGENCE" in line:
                            log_snip += line.strip()[:100] + " | "
                # crude caps from command line in log
                if "--maxiter" in text:
                    import re

                    m = re.search(r"--maxiter\s+(\d+)", text)
                    if m:
                        caps["maxiter"] = int(m.group(1))
                    m = re.search(r"--max_macroiterations\s+(\d+)", text)
                    if m:
                        caps["max_macro"] = int(m.group(1))

            rel = str(oo.relative_to(REPO))
            rec = {
                "path": rel,
                "success": success,
                "nit": nit,
                "nfev": nfev,
                "message": msg[:120],
                "maxiter_hit": maxiter_hit,
                "n_rounds": len(round_summaries),
                "rounds": round_summaries,
                "n_trace": n_trace,
                "caps": caps,
                "log_snip": log_snip[:200],
            }
            records.append(rec)

    # print compact
    hit = sum(1 for r in records if r["maxiter_hit"])
    ok = sum(1 for r in records if r["success"] is True)
    print(f"iterative oo.json files: {len(records)}; success=True: {ok}; maxiter/limit signals: {hit}")
    for r in records:
        nit_rounds = [x.get("nit") for x in r["rounds"] if x.get("nit") is not None]
        succ_rounds = [x.get("success") for x in r["rounds"]]
        print(
            f"  {r['path']}\n"
            f"    success={r['success']} nit={r['nit']} nfev={r['nfev']} "
            f"n_trace={r['n_trace']} caps={r['caps']} maxiter_hit={r['maxiter_hit']}\n"
            f"    msg={r['message']!r}\n"
            f"    rounds={r['n_rounds']} nit={nit_rounds} succ={succ_rounds}"
        )
    return records


def inspect_one_oo_json_structure() -> None:
    samples = list((REPO / "results/n2_endpoint_grid").rglob("**/iterative/**/oo.json"))
    samples += list((REPO / "results/n2_checklist_supplement").rglob("**/iterative_*_oo.json"))
    if not samples:
        print("no oo.json samples")
        return
    p = samples[0]
    data = json.loads(p.read_text(encoding="utf-8"))
    print(f"\nSample oo.json keys ({p.relative_to(REPO)}): {sorted(data.keys())[:40]}")
    for k in ("success", "message", "nit", "nfev", "cost_before", "cost_after", "optimizer_maxiter"):
        if k in data:
            print(f"  {k}: {data[k]}")
    # nested
    for k in ("selection_meta", "history", "selection_history", "rounds", "iterative"):
        if k in data:
            v = data[k]
            print(f"  {k} type={type(v).__name__}", end="")
            if isinstance(v, dict):
                print(f" keys={list(v.keys())[:20]}")
            elif isinstance(v, list):
                print(f" len={len(v)}")
                if v and isinstance(v[0], dict):
                    print(f"    [0] keys={list(v[0].keys())[:20]}")
            else:
                print()


def main() -> None:
    latest = load_endpoint_latest(REPO / "tables/n2/endpoint_grid.csv")
    plot_endpoint(latest, REPO / "tables/n2/n2_endpoint_k_sectors_dim.png")
    plot_endpoint_it_md = latest  # also write classic overlay
    # filter plot for iterative vs mixed-disjoint only — reuse by rewriting
    series_only = {
        k: v
        for k, v in latest.items()
        if k[1] in ("iterative", "mixed_disjoint")
    }
    # temporary: call plot with filtered by monkey patching via second function write
    series: dict[tuple[str, str], list] = defaultdict(list)
    for (_b, method, cost), row in series_only.items():
        series[(method, cost)].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    panels = [("K", r"$K$"), ("sectors", "sectors"), ("D_max", r"$D_{\max}$")]
    for key in (
        ("iterative", "NC"),
        ("iterative", "variance"),
        ("mixed_disjoint", "NC"),
        ("mixed_disjoint", "variance"),
    ):
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
        ax.set_xlabel("N--N bond, A")
    axes[-1].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("N2 FCI-ref OO + K: iterative vs mixed-disjoint (sto-3g)")
    fig.tight_layout()
    out2 = REPO / "tables/n2/iterative_n2_endpoint.png"
    fig.savefig(out2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out2}")

    analyze_endpoint(latest)
    analyze_checklist(REPO / "tables/n2/checklist_supplement_manifest.csv")
    inspect_one_oo_json_structure()
    inspect_oo_convergence()


if __name__ == "__main__":
    main()
