"""One endpoint job: select + OO + FCI-overlap K metrics.

Methods:
  mixed_disjoint  — greedy quota, --disjoint_orbitals 1
  mixed_overlap   — greedy quota, --disjoint_orbitals 0
  iterative       — fixed-M NC-ranked LAS (Route A exact taper)
  exhaustive      — self-consistent fixed point over the complete quotient pool

Orbital packing: ``--orbital_rotation {full,irrep}`` (irrep needs a
point-group chk so PG exacts stay Pauli-Z). Exact E: STO-3G spatial PG
rows (H2O: Q_B1/Q_B2; N2: Q_pix/Q_piy/Q_u); Clifford metrics also include
spin-resolved ``N_alpha``/``N_beta`` so document ``r = r_spatial + 2``.
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
# Running as ``python scripts/....py`` puts ``scripts/`` on sys.path[0], not the
# repo root — so ``import src`` fails unless we add REPO explicitly.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Bump when chk/irrep wiring changes — must appear in job .out if sync worked.
ENDPOINT_SCRIPT_VERSION = "2026-08-10-q4-budget-v5"


def _bond_tag(bond: float) -> str:
    return f"{bond:.4f}".replace(".", "p")


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


def _ensure_chk(
    molecule: str,
    bond: float,
    basis: str,
    hoh_angle: float,
    point_group: str | None,
    *,
    require_irreps: bool = False,
) -> Path:
    # Fingerprint: if .err still mentions diagnose_chk_irreps inside _ensure_chk,
    # the cluster is running a stale copy of this script — re-sync scripts/.
    from src.pyscf_chk import ensure_hamiltonian_chk

    return ensure_hamiltonian_chk(
        REPO,
        molecule,
        bond,
        basis,
        hoh_angle,
        point_group,
        require_irreps=require_irreps,
        log_dir="results/_endpoint_ham",
    )


def _ensure_exact_parity(
    molecule: str,
    norb: int,
    path: Path,
    extra: str | None,
    *,
    force: bool = False,
    chk: Path | None = None,
) -> Path:
    """Ensure the STO-3G exact matrix exists (PG spatial rows, not all-ones).

    ``chk`` must be supplied so the generator supports are derived from THIS
    geometry's MO irrep labels. The hardcoded index sets in
    ``src/sto3g_exact_symmetries.py`` are only valid at the PDF reference
    geometry; using them elsewhere produces Z products that do not commute with
    H, which is what invalidated the 2026-08 campaign.
    """
    if path.is_file() and extra is None and not force:
        return path
    cmd = [
        sys.executable,
        "-u",
        "scripts/write_default_exact_parity.py",
        "--molecule",
        molecule,
        "--norb",
        str(norb),
        "-o",
        str(path),
    ]
    if chk is not None:
        cmd.extend(["--chk", str(chk)])
    if extra:
        cmd.extend(["--extra", extra])
    _run(cmd, path.with_suffix(path.suffix + ".log"))
    return path


def _report_supports(molecule: str, chk: Path) -> bool:
    """Log derived vs hardcoded generator supports; return whether they agree."""
    try:
        from scripts.write_default_exact_parity import _orbsym_from_chk
    except Exception:  # noqa: BLE001 - running as a script, not a package
        sys.path.insert(0, str(REPO / "scripts"))
        from write_default_exact_parity import _orbsym_from_chk
    from src.sto3g_exact_symmetries import (
        MOLECULE_POINT_GROUP,
        exact_spatial_sets_from_orbsym,
        hardcoded_supports_valid,
    )

    orbsym, _pg, _norb = _orbsym_from_chk(chk)
    derived = {
        name: sorted(sup)
        for name, sup in exact_spatial_sets_from_orbsym(
            orbsym, MOLECULE_POINT_GROUP[molecule]
        )
    }
    ok = hardcoded_supports_valid(orbsym, molecule)
    print(f"[endpoint] orbsym  = {orbsym}", flush=True)
    print(f"[endpoint] supports= {derived}", flush=True)
    print(
        f"[endpoint] hardcoded indices valid at this geometry: "
        f"{'yes' if ok else 'NO (derived supports used)'}",
        flush=True,
    )
    return bool(ok)


def _append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=("h2o", "n2"), required=True)
    parser.add_argument("--bond", type=float, required=True)
    parser.add_argument(
        "--method",
        choices=("mixed_disjoint", "mixed_overlap", "iterative", "exhaustive"),
        required=True,
        help="exhaustive = self-consistent exhaustive quotient-pool fixed point",
    )
    parser.add_argument("--n_singles", type=int, default=None,
                        help="greedy quota: number of single-Zbar rows")
    parser.add_argument("--n_quartets", type=int, default=None,
                        help="greedy quota: number of pair rows")
    parser.add_argument("--cost_function", choices=("NC", "variance"), required=True)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--hoh_angle", type=float, default=104.5)
    parser.add_argument(
        "--point_group",
        default=None,
        help="PySCF group for chk (default C2v/D2h); required for irrep U",
    )
    parser.add_argument(
        "--orbital_rotation",
        choices=("full", "irrep"),
        default=os.environ.get("ORBITAL_ROTATION", "full"),
        help="SO(n) full or intra-irrep packing (default: full or $ORBITAL_ROTATION)",
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--m_round", type=int, default=1)
    parser.add_argument(
        "--max_macroiterations",
        type=int,
        default=None,
        help="iterative macro cap (default: max(2*norb, 8))",
    )
    parser.add_argument(
        "--stable_span_iters",
        type=int,
        default=2,
        help="iterative consecutive all-five stop count (default: 2)",
    )
    parser.add_argument(
        "--iterative_reference",
        choices=("exact_taper", "fci_rotate"),
        default="fci_rotate",
        help="ranking reference for iterative select (OO always fci_rotate)",
    )
    parser.add_argument("--states_per_sector", type=int, default=500)
    parser.add_argument(
        "--n_sym",
        type=int,
        default=None,
        help=(
            "LAS budget M (default: n_singles+n_quartets = 5 h2o / 7 n2). "
            "Note r_spatial + n_sym = N at the defaults, which forces "
            "span(E u LAS) to full rank and makes the sector partition the "
            "finest possible -- identical for every selection rule. Choose "
            "n_sym < N - r_spatial to let the methods differ."
        ),
    )
    parser.add_argument(
        "--strict_reference_weight",
        action="store_true",
        help="fail the point when the retained sector does not hold the reference",
    )
    parser.add_argument("--exact_parity", default=None)
    parser.add_argument("--exact_parity_extra", default=None)
    parser.add_argument("--exact_sector", default=None)
    parser.add_argument(
        "--results_root", default=None,
        help="top-level results directory name, default "
             "'<molecule>_endpoint_grid'. Use a distinct root for a campaign "
             "at a different budget so an earlier one is not overwritten.")
    parser.add_argument("--results_csv", default=None)
    parser.add_argument(
        "--campaign",
        default=os.environ.get("CAMPAIGN", ""),
        help=(
            "era tag written to the CSV. tables/*/endpoint_grid.csv is "
            "append-only and already interleaves the n_exact=1 and "
            "document-exact campaigns; tag new rows so they can be separated."
        ),
    )
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()
    print(
        f"[endpoint] version={ENDPOINT_SCRIPT_VERSION} "
        f"file={Path(__file__).resolve()}",
        flush=True,
    )

    started = time.time()
    # Budget and quota at the corrected bound M = n - r_sp, where r_sp now
    # includes total particle parity. Quota split fixed by decision, not
    # derived: (2,2) for H2O at M=4 and (3,3) for N2 at M=6.
    norb = 7 if args.molecule == "h2o" else 10
    r_spatial = 3 if args.molecule == "h2o" else 4   # includes all-ones
    default_M = norb - r_spatial                     # 4 (H2O) / 6 (N2)
    default_quota = (2, 2) if args.molecule == "h2o" else (3, 3)

    n_sym = int(args.n_sym) if args.n_sym is not None else default_M
    if args.n_singles is not None and args.n_quartets is not None:
        n_singles, n_quartets = int(args.n_singles), int(args.n_quartets)
    elif n_sym == default_M:
        n_singles, n_quartets = default_quota
    else:
        raise SystemExit(
            f"--n_sym={n_sym} differs from the default {default_M}; the greedy "
            "quota split is a recorded choice, so pass --n_singles and "
            "--n_quartets explicitly."
        )
    if n_singles + n_quartets != n_sym and args.method != "iterative":
        raise SystemExit(
            f"quota ({n_singles},{n_quartets}) sums to "
            f"{n_singles + n_quartets}, not n_sym={n_sym}"
        )
    if n_sym > norb - r_spatial:
        raise SystemExit(
            f"[endpoint] M={n_sym} exceeds the admissible budget "
            f"n - r_sp = {norb - r_spatial}."
        )
    if n_sym == norb - r_spatial:
        print(
            f"[endpoint][note] M={n_sym} = n - r_sp: span(E u LAS) is the whole "
            "quotient for every rule, so the sector partition is identical "
            "across rules and any difference in K arises through the optimised "
            "orbitals rather than the pool.",
            flush=True,
        )
    default_pg = "C2v" if args.molecule == "h2o" else "D2h"
    point_group = args.point_group if args.point_group is not None else default_pg
    if point_group in ("", "none", "None"):
        point_group = None
    orbital_rotation = str(args.orbital_rotation).lower()
    if orbital_rotation == "irrep" and not point_group:
        raise SystemExit("irrep orbital_rotation requires --point_group")

    out_dir = (
        REPO
        / "results"
        / (args.results_root or f"{args.molecule}_endpoint_grid")
        / f"bond_{_bond_tag(args.bond)}"
        / f"U_{orbital_rotation}"
        / args.method
        / args.cost_function
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ORDER MATTERS: the chk must exist first, because the exact generator
    # supports are derived from its MO irrep labels. The old order wrote a
    # geometry-independent exact matrix from hardcoded orbital indices.
    chk = _ensure_chk(
        args.molecule,
        args.bond,
        args.basis,
        args.hoh_angle,
        point_group,
        # PG labels are needed for the exact matrix regardless of the packing.
        require_irreps=True,
    )

    # One exact matrix PER GEOMETRY -- a single shared file cannot be correct
    # across a scan where the MO ordering changes.
    exact_path = Path(
        args.exact_parity
        or (
            REPO
            / "exact"
            / f"{args.molecule}_norb{norb}_bond{_bond_tag(args.bond)}_exact.txt"
        )
    )
    _ensure_exact_parity(
        args.molecule,
        norb,
        exact_path,
        args.exact_parity_extra,
        force=True,
        chk=chk,
    )
    supports_ok = _report_supports(args.molecule, chk)
    oo_path = out_dir / "oo.json"
    cmd = [
        sys.executable,
        "-u",
        "optimize_symmetries.py",
        str(chk),
        "--reference",
        "fci",
        "--orbital_rotation",
        orbital_rotation,
        "--cost_function",
        args.cost_function,
        "--candidates",
        "senquart",
        "--exact_parity",
        str(exact_path),
        "--maxiter",
        str(args.maxiter),
        "--outname",
        str(oo_path),
        "--verbose",
        # Sec.10 item 5: per-iteration optimizer trajectory, not just nit/nfev.
        "--oo_trace_json",
        str(out_dir / "oo_trace.json"),
    ]
    if args.exact_sector:
        cmd.extend(["--exact_sector", args.exact_sector])

    if args.method in ("iterative", "exhaustive"):
        max_macro = (
            int(args.max_macroiterations)
            if args.max_macroiterations is not None
            else max(2 * norb, 8)
        )
        cmd.extend(
            [
                "--select",
                "iterative",
                "--n_sym",
                str(n_sym),
                "--m_round",
                str(args.m_round),
                "--max_macroiterations",
                str(max_macro),
                "--stable_span_iters",
                str(int(args.stable_span_iters)),
                "--iterative_reference",
                str(args.iterative_reference),
            ]
        )
    else:
        disjoint = 1 if args.method == "mixed_disjoint" else 0
        cmd.extend(
            [
                "--select",
                "greedy",
                "--n_singles",
                str(n_singles),
                "--n_quartets",
                str(n_quartets),
                "--disjoint_orbitals",
                str(disjoint),
            ]
        )

    status = "ok"
    message = ""
    try:
        if args.method == "exhaustive":
            # Arm 3: rank the complete quotient pool and re-optimise the
            # orbitals against it until the pool is self-consistent. The driver
            # first needs a starting oo.json, which the iterative path provides.
            _run(cmd, out_dir / "oo.log")
            sc_cmd = [
                sys.executable, "-u",
                "scripts/exhaustive_selfconsistent.py",
                "--oo", str(oo_path),
                "--scope", "spatial",
                "--M", str(n_sym),
                "--start", "run",
                "--max-macro", str(args.max_macroiterations or 20),
                "-o", str(out_dir / "exhaustive_selfconsistent.json"),
            ]
            _run(sc_cmd, out_dir / "exhaustive.log")
            oo_path = out_dir / f"oo_selfconsistent_M{n_sym}.json"
            if not oo_path.is_file():
                oo_path = out_dir / "oo_selfconsistent.json"
            oo = _load_last_json(oo_path)
        else:
            _run(cmd, out_dir / "oo.log")
            oo = _load_last_json(oo_path)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        message = str(exc)
        oo = {}

    metrics = {}
    if status == "ok" and not args.skip_metrics:
        try:
            if oo.get("parity") in (None, "", "null") and oo.get("parity_output"):
                oo = dict(oo)
                oo["parity"] = oo["parity_output"]
            metrics_in = out_dir / "metrics.in.json"
            metrics_path = out_dir / "metrics.json"
            metrics_in.write_text(json.dumps(oo, indent=2), encoding="utf-8")
            # METRICS_ENTRY lets the metrics stage run from a compiled
            # snapshot (metrics_good.pyc) while metrics.py source is being
            # recovered. CPython executes a .pyc directly when the magic
            # matches the interpreter.
            metrics_entry = os.environ.get("METRICS_ENTRY", "metrics.py")
            mcmd = [
                sys.executable,
                "-u",
                metrics_entry,
                str(metrics_in),
                # Required for the exact/LAS split: the determinant backend
                # emits no n_exact / exact_sector and fails audit_08.
                "--sector_backend",
                "clifford",
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
            if args.strict_reference_weight:
                mcmd.append("--strict_reference_weight")
            _run(mcmd, out_dir / "metrics.log")
            metrics = _load_last_json(metrics_path)
        except Exception as exc:  # noqa: BLE001
            status = "metrics_failed"
            message = str(exc)

    elapsed = time.time() - started
    row = {
        "molecule": args.molecule,
        "bond": repr(float(args.bond)),
        "method": args.method,
        "cost_function": args.cost_function,
        "basis": args.basis,
        "point_group_ham": point_group or "",
        "orbital_rotation": orbital_rotation,
        "exact_parity": str(exact_path),
        "n_singles": n_singles if args.method != "iterative" else "",
        "n_quartets": n_quartets if args.method != "iterative" else "",
        "n_sym": n_sym,
        "disjoint_orbitals": (
            ""
            if args.method == "iterative"
            else int(args.method == "mixed_disjoint")
        ),
        "iterative_reference": (
            args.iterative_reference if args.method == "iterative" else ""
        ),
        "cost_before": oo.get("cost_before", ""),
        "cost_after": oo.get("cost_after", ""),
        "selection_rule": oo.get("selection_rule", ""),
        "M_eff": metrics.get("M_eff", oo.get("M_eff", "")),
        "K": metrics.get("K", ""),
        "D_max": metrics.get("D_max", ""),
        "dim": metrics.get("relevant_sectors_total_dim", ""),
        "sectors": metrics.get("relevant_sectors_count", ""),
        "E_decoupled": metrics.get("E_decoupled", ""),
        "E_FCI": metrics.get("E_FCI", ""),
        "exact_tapered": metrics.get("exact_tapered", ""),
        "n_exact": metrics.get("n_exact", ""),
        "n_las": metrics.get("n_las", ""),
        "converged": metrics.get("converged", ""),
        "reference_weight_sum": metrics.get("reference_weight_sum", ""),
        "exact_sector": (
            ""
            if metrics.get("exact_sector") is None
            else "".join(str(int(b)) for b in metrics.get("exact_sector", []))
        ),
        "hardcoded_supports_valid": int(bool(supports_ok)),
        "campaign": args.campaign,
        "include_spin_number_exact": oo.get("include_spin_number_exact", ""),
        "oo_json": str(oo_path),
        "status": status,
        "elapsed_s": f"{elapsed:.1f}",
        "message": message,
    }
    csv_path = Path(
        args.results_csv
        or (REPO / "tables" / args.molecule / "endpoint_grid.csv")
    )
    _append_csv(csv_path, row, list(row.keys()))
    print(json.dumps(row, indent=2), flush=True)
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
