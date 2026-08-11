"""Extract Mixed/Iterative endpoint tables from local OO+metrics results for the report."""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from json import JSONDecoder
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tables" / "analysis" / "checklist_endpoint_extract.json"


def _f(row: dict, key: str):
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(ast.literal_eval(raw))
    except Exception:
        return float(raw)


def _load_json_last(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
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
        raise ValueError(f"no JSON in {path}")
    return objs[-1]


def _load_csv(path: Path, select: str | None = None) -> dict:
    last = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "ok":
                continue
            if select and r.get("select") != select:
                continue
            key = (round(_f(r, "bond"), 8), r["cost_function"], r.get("select", ""))
            last[key] = r
    return last


def _oo_path(molecule: str, bond: float, kind: str, cost: str) -> Path:
    tag = f"{'greedy' if kind != 'iterative' else 'iterative'}_{cost}_fci"
    bond_tag = f"{bond:.4f}".replace(".", "p")
    if kind == "mixed_disjoint":
        root = REPO / "results" / f"{molecule}_fci_mixed_disjoint"
    else:
        root = REPO / "results" / f"{molecule}_fci_grid"
    return root / f"bond_{bond_tag}" / tag


def _fmt_gen(oo: dict) -> dict:
    if oo.get("selection") == "iterative":
        rounds = oo.get("rounds") or []
        gens = []
        for r in rounds:
            for s in r.get("singles") or []:
                gens.append({"kind": "single", "orbitals": [int(s)], "round": r.get("accumulated_rank")})
            for q in r.get("quartets") or []:
                gens.append({"kind": "quartet", "orbitals": [int(x) for x in q], "round": r.get("accumulated_rank")})
        return {
            "selection_rule": oo.get("selection_rule"),
            "singles": None,
            "quartets": None,
            "generators_by_round": gens,
            "accumulated_orbitals": oo.get("accumulated_orbitals"),
            "selected_costs": oo.get("selected_costs"),
            "cost_before": oo.get("cost_before"),
            "cost_after": oo.get("cost_after"),
            "nit": oo.get("nit"),
            "nfev": oo.get("nfev"),
            "message": oo.get("message"),
            "m_round": oo.get("m_round"),
            "rounds_summary": [
                {
                    "rank": r.get("accumulated_rank"),
                    "singles": r.get("singles"),
                    "quartets": r.get("quartets"),
                    "masks": r.get("masks"),
                    "additive_cost": r.get("additive_cost"),
                    "cost_before": (r.get("optimization") or {}).get("cost_before"),
                    "cost_after": (r.get("optimization") or {}).get("cost_after"),
                    "nit": (r.get("optimization") or {}).get("nit"),
                    "clifford_basis_rows": (r.get("clifford") or {}).get("basis_rows"),
                }
                for r in rounds
            ],
        }
    return {
        "selection_rule": oo.get("selection_rule"),
        "singles": oo.get("singles"),
        "quartets": oo.get("quartets"),
        "selected_costs": oo.get("selected_costs"),
        "selected_indices": oo.get("selected_indices"),
        "cost_before": oo.get("cost_before"),
        "cost_after": oo.get("cost_after"),
        "nit": oo.get("nit"),
        "nfev": oo.get("nfev"),
        "message": oo.get("message"),
    }


def main() -> None:
    sources = {
        "h2o": {
            "greedy": _load_csv(REPO / "tables/h2o/fci_oo_metrics_grid.csv", "greedy"),
            "iterative": _load_csv(REPO / "tables/h2o/fci_oo_metrics_grid.csv", "iterative"),
            "mixed_disjoint": _load_csv(
                REPO / "tables/h2o/fci_oo_metrics_mixed_disjoint.csv", "greedy"
            ),
        },
        "n2": {
            "greedy": _load_csv(REPO / "tables/n2/fci_oo_metrics_grid.csv", "greedy"),
            "iterative": _load_csv(REPO / "tables/n2/fci_oo_metrics_grid.csv", "iterative"),
            "mixed_disjoint": _load_csv(
                REPO / "tables/n2/fci_oo_metrics_mixed_disjoint.csv", "greedy"
            ),
        },
    }

    out: dict = {"molecules": {}}
    for mol, blocks in sources.items():
        mol_out: dict = {"methods": {}}
        for method, rows in blocks.items():
            method_rows = []
            for (bond, cost, _sel), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                kind = "iterative" if method == "iterative" else (
                    "mixed_disjoint" if method == "mixed_disjoint" else "greedy"
                )
                oo_path = _oo_path(mol, bond, kind if method != "greedy" else "greedy", cost)
                # greedy overlap lives in fci_grid
                if method == "greedy":
                    oo_path = REPO / "results" / f"{mol}_fci_grid" / f"bond_{bond:.4f}".replace(".", "p") / f"greedy_{cost}_fci"
                entry = {
                    "bond": bond,
                    "cost_function": cost,
                    "K": _f(r, "K"),
                    "sectors": _f(r, "sectors"),
                    "dim": _f(r, "dim"),
                    "edec_error": _f(r, "edec_error"),
                    "csv_cost": _f(r, "cost"),
                    "E_FCI": _f(r, "E_FCI"),
                    "E_decoupled": _f(r, "E_decoupled"),
                    "oo_path": str(oo_path),
                }
                if oo_path.is_file():
                    oo = _load_json_last(oo_path)
                    entry["oo"] = _fmt_gen(oo)
                    entry["cost_before"] = oo.get("cost_before")
                    entry["cost_after"] = oo.get("cost_after")
                method_rows.append(entry)
            mol_out["methods"][method] = method_rows
        # NC vs variance win counts for mixed_disjoint and iterative
        for method in ("mixed_disjoint", "iterative"):
            by_bond = defaultdict(dict)
            for e in mol_out["methods"][method]:
                by_bond[e["bond"]][e["cost_function"]] = e
            wins = {"NC": 0, "variance": 0, "tie_K": 0}
            pairs = []
            for bond, d in sorted(by_bond.items()):
                if "NC" not in d or "variance" not in d:
                    continue
                kn, kv = d["NC"]["K"], d["variance"]["K"]
                dn, dv = d["NC"]["dim"], d["variance"]["dim"]
                if kn < kv:
                    winner = "NC"
                    wins["NC"] += 1
                elif kv < kn:
                    winner = "variance"
                    wins["variance"] += 1
                else:
                    wins["tie_K"] += 1
                    if dn is not None and dv is not None and dn != dv:
                        winner = "tie_K_dimNC" if dn < dv else "tie_K_dimVar"
                    else:
                        winner = "tie"
                pairs.append(
                    {
                        "bond": bond,
                        "K_NC": kn,
                        "K_var": kv,
                        "dim_NC": dn,
                        "dim_var": dv,
                        "winner": winner,
                    }
                )
            mol_out[f"nc_vs_var_{method}"] = {"wins": wins, "pairs": pairs}
        out["molecules"][mol] = mol_out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
