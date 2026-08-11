"""One geometry point: build ham → FCI-ref select+OO → FCI-ref K metrics → CSV.

Used by Trillium H2O / N2 bond-scan array scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FIELDNAMES = [
    "bond",
    "molecule",
    "select",
    "cost_function",
    "n_singles",
    "n_quartets",
    "n_sym",
    "m_round",
    "basis",
    "cost",
    "E_FCI",
    "E_decoupled",
    "edec_error",
    "K",
    "sectors",
    "dim",
    "D_max",
    "oo_json",
    "metrics_json",
    "chk",
    "status",
    "message",
    "elapsed_s",
    "reference",
    "overlap_reference",
    "backend",
]


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if sys.platform != "win32":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_oo(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    objs: list[dict] = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] != "{":
            break
        obj, end = dec.raw_decode(raw, i)
        if isinstance(obj, dict):
            objs.append(obj)
        i = end
    if not objs:
        raise ValueError(f"no JSON object in {path}")
    data = dict(objs[-1])
    if data.get("parity") in (None, "", "null") and data.get("parity_output"):
        data["parity"] = data["parity_output"]
    return data


def _pool_cost(oo: dict) -> float | None:
    rounds = oo.get("rounds") or []
    if rounds:
        opt = rounds[-1].get("optimization") or {}
        if opt.get("cost_after") is not None:
            return float(opt["cost_after"])
    costs = oo.get("selected_costs")
    if costs:
        return float(sum(float(c) for c in costs))
    return None


def _chk_path(molecule: str, bond: float, basis: str, hoh_angle: float) -> Path:
    if molecule == "h2o":
        stem = f"H2O_OH{bond:.4f}_{hoh_angle:.4f}{basis}"
    elif molecule == "n2":
        stem = f"N2_bond{bond:.4f}{basis}"
    else:
        raise ValueError(molecule)
    return Path("hamiltonians") / f"{stem}.chk"


def _ensure_chk(
    molecule: str,
    bond: float,
    basis: str,
    hoh_angle: float,
) -> Path:
    chk = _chk_path(molecule, bond, basis, hoh_angle)
    if chk.is_file():
        print(f"[ham] reuse {chk}")
        return chk
    cmd = [
        sys.executable,
        "-u",
        "make_pyscf_hamiltonian.py",
        molecule,
        str(bond),
        "--basis",
        basis,
    ]
    if molecule == "h2o":
        cmd.extend(["--mol_parameter_2", str(hoh_angle)])
    print("[ham]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not chk.is_file():
        raise FileNotFoundError(f"expected checkpoint {chk}")
    return chk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=("h2o", "n2"), required=True)
    parser.add_argument("--bond", type=float, required=True)
    parser.add_argument("--select", choices=("greedy", "iterative"), required=True)
    parser.add_argument("--cost_function", choices=("NC", "variance"), required=True)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--hoh_angle", type=float, default=104.5)
    parser.add_argument("--n_singles", type=int, default=None)
    parser.add_argument("--n_quartets", type=int, default=None)
    parser.add_argument("--n_sym", type=int, default=None)
    parser.add_argument("--m_round", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--states_per_sector", type=int, default=200)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_root", default="results")
    parser.add_argument(
        "--grid_name",
        default=None,
        help="results/<grid_name>/bond_… (default: <molecule>_fci_grid)",
    )
    args = parser.parse_args()

    started = time.time()
    bond_tag = f"{args.bond:.4f}".replace(".", "p")
    tag = f"{args.select}_{args.cost_function}_fci"
    grid_name = args.grid_name or f"{args.molecule}_fci_grid"
    out_dir = Path(args.out_root) / grid_name / f"bond_{bond_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    oo_path = out_dir / tag
    metrics_path = out_dir / f"{tag}_metrics.json"

    n_singles = args.n_singles
    n_quartets = args.n_quartets
    n_sym = args.n_sym
    if args.molecule == "h2o":
        n_singles = 3 if n_singles is None else n_singles
        n_quartets = 2 if n_quartets is None else n_quartets
    else:
        n_singles = 4 if n_singles is None else n_singles
        n_quartets = 3 if n_quartets is None else n_quartets
    if n_sym is None:
        n_sym = int(n_singles) + int(n_quartets)

    row = {
        "bond": repr(float(args.bond)),
        "molecule": args.molecule,
        "select": args.select,
        "cost_function": args.cost_function,
        "n_singles": str(n_singles) if args.select == "greedy" else "",
        "n_quartets": str(n_quartets) if args.select == "greedy" else "",
        "n_sym": str(n_sym),
        "m_round": str(args.m_round) if args.select == "iterative" else "",
        "basis": args.basis,
        "cost": "",
        "E_FCI": "",
        "E_decoupled": "",
        "edec_error": "",
        "K": "",
        "sectors": "",
        "dim": "",
        "oo_json": str(oo_path),
        "metrics_json": str(metrics_path),
        "chk": "",
        "status": "failed",
        "message": "",
        "elapsed_s": "",
        "reference": "fci",
        "overlap_reference": "fci",
        "backend": "fci",
    }

    try:
        chk = _ensure_chk(args.molecule, args.bond, args.basis, args.hoh_angle)
        row["chk"] = str(chk)

        oo_cmd = [
            sys.executable,
            "-u",
            "optimize_symmetries.py",
            str(chk),
            "--reference",
            "fci",
            "--select",
            args.select,
            "--cost_function",
            args.cost_function,
            "--candidates",
            "senquart",
            "--maxiter",
            str(args.maxiter),
            "--outname",
            str(oo_path),
        ]
        if args.select == "greedy":
            oo_cmd.extend(
                ["--n_singles", str(n_singles), "--n_quartets", str(n_quartets)]
            )
        else:
            oo_cmd.extend(["--n_sym", str(n_sym), "--m_round", str(args.m_round)])
        print("[oo]", " ".join(oo_cmd))
        subprocess.run(oo_cmd, check=True)

        oo = _load_oo(oo_path)
        # Rewrite a clean single-object OO JSON with parity filled for metrics.
        clean_oo = dict(oo)
        if clean_oo.get("parity") in (None, "", "null") and clean_oo.get("parity_output"):
            clean_oo["parity"] = clean_oo["parity_output"]
        clean_path = Path(str(oo_path) + ".metrics_in.json")
        clean_path.write_text(json.dumps(clean_oo, indent=2), encoding="utf-8")
        c = _pool_cost(oo)
        if c is not None:
            row["cost"] = repr(c)

        metrics_cmd = [
            sys.executable,
            "-u",
            "metrics.py",
            str(clean_path),
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
        ]
        print("[metrics]", " ".join(metrics_cmd))
        subprocess.run(metrics_cmd, check=True)

        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        # metrics may append; load last object if needed
        if not isinstance(metrics, dict) or "K" not in metrics:
            raw = Path(metrics_path).read_text(encoding="utf-8")
            dec = json.JSONDecoder()
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
            metrics = objs[-1]

        e_fci = metrics.get("E_FCI")
        e_dec = metrics.get("E_decoupled")
        k = metrics.get("K")
        sectors = metrics.get("n_sectors_total", metrics.get("relevant_sectors_count"))
        dim = metrics.get("D_max", metrics.get("relevant_sectors_total_dim"))
        row["E_FCI"] = "" if e_fci is None else repr(float(e_fci))
        row["E_decoupled"] = "" if e_dec is None else repr(float(e_dec))
        if e_fci is not None and e_dec is not None:
            row["edec_error"] = repr(abs(float(e_dec) - float(e_fci)))
        if k is not None:
            kf = float(k)
            row["K"] = repr(int(kf) if kf.is_integer() else kf)
        if sectors is not None:
            row["sectors"] = repr(int(sectors))
        if dim is not None:
            row["dim"] = repr(int(dim))
            row["D_max"] = repr(int(dim))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["message"] = str(exc)
        row["status"] = "failed"
        print(f"[error] {exc}", file=sys.stderr)
        row["elapsed_s"] = repr(time.time() - started)
        _append_row(Path(args.csv), row)
        raise SystemExit(1) from exc

    row["elapsed_s"] = repr(time.time() - started)
    _append_row(Path(args.csv), row)
    print(
        f"[ok] bond={args.bond} {args.select} {args.cost_function} "
        f"K={row['K']} sectors={row['sectors']} dim={row['dim']}"
    )


if __name__ == "__main__":
    main()
