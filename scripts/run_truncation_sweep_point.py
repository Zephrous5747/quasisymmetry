"""Truncation sweep: K and W_captured vs states_per_sector on a fixed OO JSON.

Reuses an existing OO / parity result (no re-selection / re-OO). Reports
chemical-accuracy K and captured FCI weight for each truncation level,
including a full-sector spectrum pass.

Example:
  python scripts/run_truncation_sweep_point.py \\
    --oo_json results/.../mixed_disjoint_NC_oo.json \\
    --molecule n2 --bond 1.8 --cost_function NC --protocol mixed_disjoint
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from json import JSONDecoder
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# "full" uses a huge root count so metrics falls back to dense eigh per sector.
DEFAULT_LEVELS = (200, 300, 500, 10**9)


def _load_last_json(path: Path) -> dict:
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
        raise ValueError(f"no JSON object in {path}")
    return objs[-1]


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): see {log_path}")


def _level_tag(n: int) -> str:
    return "full" if n >= 10**6 else str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oo_json", type=Path, required=True)
    parser.add_argument("--molecule", choices=("h2o", "n2"), required=True)
    parser.add_argument("--bond", type=float, required=True)
    parser.add_argument("--cost_function", choices=("NC", "variance"), required=True)
    parser.add_argument("--protocol", default="unknown")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="states_per_sector values (use >=1e6 for full sector spectra)",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    oo_path = args.oo_json if args.oo_json.is_absolute() else REPO / args.oo_json
    oo = _load_last_json(oo_path)
    bond_tag = f"{args.bond:.4f}".replace(".", "p")
    out_dir = args.out_dir or (
        REPO
        / "results"
        / f"{args.molecule}_truncation_sweep"
        / f"bond_{bond_tag}"
        / args.cost_function
        / args.protocol
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for level in args.levels:
        tag = _level_tag(int(level))
        metrics_path = out_dir / f"metrics_sps_{tag}.json"
        log_path = out_dir / f"metrics_sps_{tag}.log"
        cmd = [
            sys.executable,
            "-u",
            "metrics.py",
            str(oo_path),
            "--backend",
            "fci",
            "--coupled_energy_method",
            "reference",
            "--overlap_reference",
            "fci",
            "--states_per_sector",
            str(int(level)),
            "--outname",
            str(metrics_path),
            "--verify_fci_rotation",
        ]
        _run(cmd, log_path)
        m = _load_last_json(metrics_path)
        row = {
            "molecule": args.molecule,
            "bond": repr(float(args.bond)),
            "cost_function": args.cost_function,
            "protocol": args.protocol,
            "states_per_sector": tag,
            "states_per_sector_raw": int(level),
            "K": m.get("K"),
            "D_max": m.get("D_max"),
            "dim": m.get("D_max", m.get("relevant_sectors_total_dim")),
            "sectors": m.get("n_sectors_total", m.get("relevant_sectors_count")),
            "W_captured": m.get("W_captured"),
            "W_captured_for_K": m.get("W_captured_for_K"),
            "M_eff": m.get("M_eff"),
            "raw_rank": m.get("raw_rank"),
            "n_retained_states": m.get("n_retained_states"),
            "fci_rotation_ok": (m.get("fci_rotation_checks") or {}).get("ok"),
            "metrics_json": str(metrics_path),
        }
        rows.append(row)
        print(
            f"[level {tag}] K={row['K']} W_captured={row['W_captured']} "
            f"M_eff={row['M_eff']}",
            flush=True,
        )

    summary = {
        "molecule": args.molecule,
        "bond": args.bond,
        "cost_function": args.cost_function,
        "protocol": args.protocol,
        "oo_json": str(oo_path),
        "levels": rows,
        "elapsed_s": time.time() - started,
    }
    (out_dir / "truncation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    manifest = REPO / "tables" / args.molecule / "truncation_sweep_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[ok] wrote {out_dir / 'truncation_summary.json'}")


if __name__ == "__main__":
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
