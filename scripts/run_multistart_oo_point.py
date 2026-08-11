"""Multi-start OO for N2 anomaly geometries (critique §8).

Runs several L-BFGS starts (identity + random x0) for a fixed discrete pool
(parity from a previous select+OO), records optimizer status, and evaluates
post-OO K via metrics.

Example:
  python scripts/run_multistart_oo_point.py \\
    --chk hamiltonians/N2_bond1.8000sto-3g.chk \\
    --parity results/.../mixed_disjoint_NC_parity.txt \\
    --cost_function NC --molecule n2 --bond 1.8 --n_starts 5
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

import numpy as np

REPO = Path(__file__).resolve().parents[1]


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


def _n_params(norb: int) -> int:
    return norb * (norb - 1) // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--molecule", choices=("h2o", "n2"), required=True)
    parser.add_argument("--bond", type=float, required=True)
    parser.add_argument("--cost_function", choices=("NC", "variance"), required=True)
    parser.add_argument("--protocol", default="fixed_parity")
    parser.add_argument("--n_starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--states_per_sector", type=int, default=500)
    parser.add_argument("--scale", type=float, default=0.3, help="random x0 amplitude")
    parser.add_argument("--norb", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    chk = args.chk if args.chk.is_absolute() else REPO / args.chk
    parity = args.parity if args.parity.is_absolute() else REPO / args.parity
    norb = args.norb or (7 if args.molecule == "h2o" else 10)
    bond_tag = f"{args.bond:.4f}".replace(".", "p")
    out_dir = args.out_dir or (
        REPO
        / "results"
        / f"{args.molecule}_multistart_oo"
        / f"bond_{bond_tag}"
        / args.cost_function
        / args.protocol
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    npar = _n_params(norb)
    starts = [np.zeros(npar)]
    for _ in range(max(0, args.n_starts - 1)):
        starts.append(args.scale * rng.standard_normal(npar))

    rows = []
    started = time.time()
    for i, x0 in enumerate(starts):
        tag = f"start{i}"
        x0_path = out_dir / f"{tag}_x0.txt"
        np.savetxt(x0_path, x0)
        oo_path = out_dir / f"{tag}_oo.json"
        cmd = [
            sys.executable,
            "-u",
            "optimize_symmetries.py",
            str(chk),
            str(parity),
            "--reference",
            "fci",
            "--cost_function",
            args.cost_function,
            "--maxiter",
            str(args.maxiter),
            "--x0",
            str(x0_path),
            "--outname",
            str(oo_path),
            "--verbose",
        ]
        _run(cmd, out_dir / f"{tag}_oo.log")
        oo = _load_last_json(oo_path)

        metrics_path = out_dir / f"{tag}_metrics.json"
        mcmd = [
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
            str(args.states_per_sector),
            "--outname",
            str(metrics_path),
            "--verify_fci_rotation",
        ]
        _run(mcmd, out_dir / f"{tag}_metrics.log")
        m = _load_last_json(metrics_path)

        row = {
            "molecule": args.molecule,
            "bond": repr(float(args.bond)),
            "cost_function": args.cost_function,
            "protocol": args.protocol,
            "start": i,
            "x0_norm": float(np.linalg.norm(x0)),
            "cost_before": oo.get("cost_before"),
            "cost_after": oo.get("cost_after"),
            "converged": oo.get("converged"),
            "nit": oo.get("nit"),
            "nfev": oo.get("nfev"),
            "message": oo.get("message"),
            "K": m.get("K"),
            "D_max": m.get("D_max"),
            "dim": m.get("D_max", m.get("relevant_sectors_total_dim")),
            "M_eff": m.get("M_eff"),
            "W_captured": m.get("W_captured"),
            "fci_rotation_ok": (m.get("fci_rotation_checks") or {}).get("ok"),
            "oo_json": str(oo_path),
            "metrics_json": str(metrics_path),
        }
        rows.append(row)
        print(
            f"[start {i}] cost_after={row['cost_after']} K={row['K']} "
            f"converged={row['converged']} nit={row['nit']}",
            flush=True,
        )

    # Best by (K, dim, cost_after)
    def _key(r):
        k = r["K"] if r["K"] is not None else 10**9
        d = r["dim"] if r["dim"] is not None else 10**9
        c = r["cost_after"] if r["cost_after"] is not None else 1e300
        return (k, d, c)

    best = min(rows, key=_key) if rows else None
    summary = {
        "molecule": args.molecule,
        "bond": args.bond,
        "cost_function": args.cost_function,
        "protocol": args.protocol,
        "n_starts": len(starts),
        "seed": args.seed,
        "maxiter": args.maxiter,
        "starts": rows,
        "best_start": best,
        "elapsed_s": time.time() - started,
    }
    (out_dir / "multistart_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    manifest = REPO / "tables" / args.molecule / "multistart_oo_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[ok] wrote {out_dir / 'multistart_summary.json'}")


if __name__ == "__main__":
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
