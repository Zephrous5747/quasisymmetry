"""Checklist supplement for one (molecule, bond, cost_function) job.

Artifacts under:
  results/<mol>_checklist_supplement/bond_<tag>/<cost>/

Covers remaining Aug-2026 checklist items beyond endpoint grids:
  - Mixed disjoint + Iterative select/OO with selection_trace (rejected near-misses)
  - OO L-BFGS snapshots (--oo_trace_json) and K at 0/25/50/75/100% checkpoints
  - Pre-OO vs post-OO physical diagnostics
  - Iterative per-round metrics + Clifford metadata
  - Clifford completion sensitivity (alternate synthesis options)
  - Cross-objective control: this cost selects the pool; other cost does OO
  - Captured FCI-overlap sum when available from metrics

Parallelize as: one SLURM task per (geometry × cost_function).
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CHECKLIST_SCRIPT_VERSION = "2026-08-06-pyscf_chk-v3"


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
    point_group: str | None = None,
    *,
    require_irreps: bool = False,
) -> Path:
    from src.pyscf_chk import ensure_hamiltonian_chk

    return ensure_hamiltonian_chk(
        REPO,
        molecule,
        bond,
        basis,
        hoh_angle,
        point_group,
        require_irreps=require_irreps,
        log_dir="results/_checklist_ham",
    )


def _ensure_exact_parity(
    molecule: str,
    norb: int,
    path: Path,
    extra: str | None,
    *,
    force: bool = False,
) -> Path:
    """Ensure STO-3G report exact matrix exists (PG spatial rows, not all-ones)."""
    if path.is_file() and not extra and not force:
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
    if extra:
        cmd.extend(["--extra", extra])
    _run(cmd, path.with_suffix(path.suffix + ".log"))
    return path


def _norb(molecule: str) -> int:
    return 7 if molecule == "h2o" else 10


def _write_parity(path: Path, orbitals: list, norb: int) -> None:
    rows = []
    for support in orbitals:
        row = np.zeros(norb, dtype=int)
        for orb in support:
            row[int(orb)] = 1
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.atleast_2d(np.asarray(rows, dtype=int))
    if mat.size == 0:
        mat = np.zeros((0, norb), dtype=int)
    elif mat.shape[1] != norb:
        # Defensive: a lone support list can collapse oddly; rebuild.
        mat = np.zeros((len(orbitals), norb), dtype=int)
        for i, support in enumerate(orbitals):
            for orb in support:
                mat[i, int(orb)] = 1
    np.savetxt(path, mat, fmt="%d")


def _metrics(oo_like: dict, metrics_path: Path, states_per_sector: int, log: Path) -> dict:
    clean = dict(oo_like)
    if clean.get("parity") in (None, "", "null") and clean.get("parity_output"):
        clean["parity"] = clean["parity_output"]
    metrics_in = Path(str(metrics_path) + ".in.json")
    metrics_in.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
        "metrics.py",
        str(metrics_in),
        "--backend",
        "fci",
        "--coupled_energy_method",
        "reference",
        "--overlap_reference",
        "fci",
        "--states_per_sector",
        str(states_per_sector),
        "--outname",
        str(metrics_path),
    ]
    _run(cmd, log)
    return _load_last_json(metrics_path)


def _summarize(m: dict) -> dict:
    e_fci, e_dec = m.get("E_FCI"), m.get("E_decoupled")
    out = {
        "K": m.get("K"),
        "D_max": m.get("D_max"),
        "sectors": m.get("relevant_sectors_count"),
        "dim": m.get("D_max", m.get("relevant_sectors_total_dim")),
        "n_sectors_total": m.get("n_sectors_total"),
        "M_eff": m.get("M_eff"),
        "raw_rank": m.get("raw_rank"),
        "W_captured": m.get("W_captured"),
        "W_captured_for_K": m.get("W_captured_for_K"),
        "E_FCI": e_fci,
        "E_decoupled": e_dec,
        "edec_error": None
        if e_fci is None or e_dec is None
        else abs(float(e_dec) - float(e_fci)),
        "relevant_sectors": m.get("relevant_sectors"),
        "states_per_sector": m.get("states_per_sector"),
        "fci_rotation_ok": (m.get("fci_rotation_checks") or {}).get("ok"),
    }
    # Prefer explicit overlap fields; else sum weights if present on eigenstates.
    for key in ("captured_fci_overlap", "total_fci_overlap", "overlap_sum", "W_captured"):
        if key in m and m[key] is not None:
            out["captured_fci_overlap"] = m[key]
            break
    return out


def _checkpoint_indices(n: int) -> list[int]:
    if n <= 0:
        return []
    fracs = (0.0, 0.25, 0.5, 0.75, 1.0)
    idxs = sorted({min(n - 1, int(round(f * (n - 1)))) for f in fracs})
    return idxs


def _oo_select(
    chk: Path,
    out_path: Path,
    *,
    select: str,
    cost: str,
    n_singles: int,
    n_quartets: int,
    n_sym: int,
    m_round: int,
    maxiter: int,
    norb: int | None = None,
    max_macroiterations: int | None = None,
    stable_span_iters: int = 2,
    iterative_reference: str = "fci_rotate",
    disjoint_orbitals: bool = True,
    exact_parity: Path | None = None,
    exact_sector: str | None = None,
    orbital_rotation: str = "full",
) -> tuple[dict, Path]:
    trace_path = Path(str(out_path) + ".oo_trace.json")
    cmd = [
        sys.executable,
        "-u",
        "optimize_symmetries.py",
        str(chk),
        "--reference",
        "fci",
        "--orbital_rotation",
        orbital_rotation,
        "--select",
        select,
        "--cost_function",
        cost,
        "--candidates",
        "senquart",
        "--maxiter",
        str(maxiter),
        "--outname",
        str(out_path),
        "--oo_trace_json",
        str(trace_path),
        "--verbose",
    ]
    if exact_parity is not None:
        cmd.extend(["--exact_parity", str(exact_parity)])
    if exact_sector:
        cmd.extend(["--exact_sector", exact_sector])
    if select == "greedy":
        cmd.extend(
            [
                "--n_singles",
                str(n_singles),
                "--n_quartets",
                str(n_quartets),
                "--disjoint_orbitals",
                "1" if disjoint_orbitals else "0",
            ]
        )
    else:
        n = int(norb) if norb is not None else int(n_sym) + 1
        max_macro = (
            int(max_macroiterations)
            if max_macroiterations is not None
            else max(2 * n, 8)
        )
        cmd.extend(
            [
                "--n_sym",
                str(n_sym),
                "--m_round",
                str(m_round),
                "--max_macroiterations",
                str(max_macro),
                "--stable_span_iters",
                str(int(stable_span_iters)),
                "--iterative_reference",
                str(iterative_reference),
            ]
        )
    _run(cmd, out_path.with_suffix(out_path.suffix + ".log"))
    return _load_last_json(out_path), trace_path


def _oo_fixed_parity(
    chk: Path,
    parity_path: Path,
    out_path: Path,
    *,
    cost: str,
    maxiter: int,
    orbital_rotation: str = "full",
) -> tuple[dict, Path]:
    trace_path = Path(str(out_path) + ".oo_trace.json")
    cmd = [
        sys.executable,
        "-u",
        "optimize_symmetries.py",
        str(chk),
        str(parity_path),
        "--reference",
        "fci",
        "--orbital_rotation",
        orbital_rotation,
        "--cost_function",
        cost,
        "--maxiter",
        str(maxiter),
        "--outname",
        str(out_path),
        "--oo_trace_json",
        str(trace_path),
        "--verbose",
    ]
    _run(cmd, out_path.with_suffix(out_path.suffix + ".log"))
    return _load_last_json(out_path), trace_path


def _run_oo_checkpoints(
    oo: dict,
    parity_path: Path,
    trace_path: Path,
    out_dir: Path,
    tag: str,
    states_per_sector: int,
) -> list[dict]:
    if not trace_path.is_file():
        return []
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not trace:
        return []
    # Prefer final-label snapshots for greedy; all steps for iterative rounds.
    steps = [t for t in trace if t.get("x") is not None]
    if not steps:
        return []
    idxs = _checkpoint_indices(len(steps))
    rows = []
    for i in idxs:
        snap = steps[i]
        payload = dict(oo)
        payload["parity"] = str(parity_path)
        payload["parity_output"] = str(parity_path)
        payload["rotation"] = snap["x"]
        payload["checklist_stage"] = f"oo_checkpoint_{i}"
        mpath = out_dir / f"{tag}_ckpt{i}_metrics.json"
        m = _metrics(payload, mpath, states_per_sector, out_dir / f"{tag}_ckpt{i}_metrics.log")
        rows.append(
            {
                "checkpoint_index": i,
                "fraction": i / max(1, len(steps) - 1),
                "surrogate_cost": snap.get("cost"),
                "label": snap.get("label"),
                "metrics": _summarize(m),
            }
        )
    (out_dir / f"{tag}_oo_checkpoints.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def _clifford_sensitivity(oo: dict, out_dir: Path, tag: str, norb: int) -> dict:
    """Re-synthesize Clifford with alternate options; compare basis_rows."""
    from openfermion import QubitOperator

    from external_imports import Clifford
    from src.iterative_pool import (
        _mask_from_z_operator,
        _z_operator_from_mask,
        gf2_rank_masks,
    )

    masks = [int(m) for m in (oo.get("accumulated_masks") or [])]
    if not masks and oo.get("rounds"):
        # Rebuild from round masks
        masks = []
        for r in oo["rounds"]:
            masks.extend(int(x) for x in (r.get("masks") or []))
    if not masks:
        return {"status": "skipped", "reason": "no accumulated masks"}

    variants = [
        {
            "name": "default",
            "kwargs": {
                "symmetry_qubits_first": True,
                "synthesis_basis": "Z",
                "generator_mapping": "positive_z",
            },
        },
        {
            "name": "symmetry_qubits_last",
            "kwargs": {
                "symmetry_qubits_first": False,
                "synthesis_basis": "Z",
                "generator_mapping": "positive_z",
            },
        },
    ]
    # Optional mappings if supported by installed Clifford.
    for mapping in ("negative_z", "positive_x"):
        variants.append(
            {
                "name": f"mapping_{mapping}",
                "kwargs": {
                    "symmetry_qubits_first": True,
                    "synthesis_basis": "Z",
                    "generator_mapping": mapping,
                },
            }
        )

    results = []
    for var in variants:
        try:
            clifford = Clifford.from_symmetries(
                [_z_operator_from_mask(mask) for mask in masks],
                n_qubits=norb,
                **var["kwargs"],
            )
            canonical_axes = [
                QubitOperator(((axis, "Z"),), 1.0) for axis in range(norb)
            ]
            basis_rows = [
                _mask_from_z_operator(clifford.inverse_transform(axis), norb)
                for axis in canonical_axes
            ]
            results.append(
                {
                    "name": var["name"],
                    "ok": True,
                    "basis_rows": [int(x) for x in basis_rows],
                    "rank": int(gf2_rank_masks(basis_rows)),
                    "permutation": [int(q) for q in getattr(clifford, "permutation", [])],
                    "kwargs": var["kwargs"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "name": var["name"],
                    "ok": False,
                    "error": str(exc),
                    "kwargs": var["kwargs"],
                }
            )

    ok_rows = [r["basis_rows"] for r in results if r.get("ok")]
    distinct = len({tuple(r) for r in ok_rows})
    report = {
        "n_masks": len(masks),
        "masks": masks,
        "n_ok_variants": len(ok_rows),
        "n_distinct_basis_rows": distinct,
        "changes_frame": distinct > 1,
        "variants": results,
    }
    (out_dir / f"{tag}_clifford_sensitivity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _repro_snapshot() -> dict:
    """Package versions / git commit for checklist reproducibility."""
    import importlib.metadata as md
    import subprocess as sp

    pkgs = [
        "numpy",
        "scipy",
        "pyscf",
        "ffsim",
        "openfermion",
        "block2",
    ]
    versions = {}
    for name in pkgs:
        try:
            versions[name] = md.version(name)
        except Exception:  # noqa: BLE001
            versions[name] = None
    git = {"commit": None, "dirty": None}
    # Scratch copies are often not git checkouts; avoid stderr noise.
    if (REPO / ".git").exists():
        try:
            commit = sp.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO),
                text=True,
                stderr=sp.DEVNULL,
            ).strip()
            dirty = sp.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(REPO),
                text=True,
                stderr=sp.DEVNULL,
            ).strip()
            git = {"commit": commit, "dirty": bool(dirty)}
        except Exception:  # noqa: BLE001
            pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "git": git,
    }


def _dump_rotation_matrix(oo: dict, out_path: Path, norb: int) -> Path | None:
    """Write post-OO U matrix from the parameter vector when available."""
    rot = oo.get("rotation")
    if not isinstance(rot, list) or not rot:
        return None
    try:
        from src.orbital_rotation import params_to_U

        U = params_to_U(np.asarray(rot, dtype=float), norb)
        np.savetxt(out_path, U)
        return out_path
    except Exception as exc:  # noqa: BLE001
        out_path.with_suffix(".err.txt").write_text(str(exc), encoding="utf-8")
        return None


def _append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule",
        "bond",
        "regime",
        "cost_function",
        "protocol",
        "K_pre",
        "K_post",
        "dim_pre",
        "dim_post",
        "cost_before",
        "cost_after",
        "n_oo_checkpoints",
        "status",
        "message",
    ]
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
    parser.add_argument("--cost_function", choices=("NC", "variance"), required=True)
    parser.add_argument("--regime", default="")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--hoh_angle", type=float, default=104.5)
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
    parser.add_argument("--skip_checkpoints", action="store_true")
    parser.add_argument("--skip_cross", action="store_true")
    parser.add_argument("--skip_clifford", action="store_true")
    parser.add_argument(
        "--only_iterative",
        action="store_true",
        help="skip mixed_disjoint/mixed_overlap; only run iterative protocol",
    )
    parser.add_argument("--exact_parity", default=None)
    parser.add_argument("--exact_parity_extra", default=None)
    parser.add_argument("--exact_sector", default=None)
    parser.add_argument(
        "--point_group",
        default=None,
        help="PySCF group for chk (default C2v/D2h); required for irrep U",
    )
    parser.add_argument(
        "--orbital_rotation",
        choices=("full", "irrep"),
        default=os.environ.get("ORBITAL_ROTATION", "full"),
        help="SO(n) full or intra-irrep packing",
    )
    args = parser.parse_args()
    print(
        f"[checklist] version={CHECKLIST_SCRIPT_VERSION} "
        f"file={Path(__file__).resolve()}",
        flush=True,
    )

    started = time.time()
    n_singles, n_quartets = (3, 2) if args.molecule == "h2o" else (4, 3)
    n_sym = n_singles + n_quartets
    norb = _norb(args.molecule)
    cost = args.cost_function
    other_cost = "variance" if cost == "NC" else "NC"
    orbital_rotation = str(args.orbital_rotation).lower()
    out_dir = (
        REPO
        / "results"
        / f"{args.molecule}_checklist_supplement"
        / f"bond_{_bond_tag(args.bond)}"
        / f"U_{orbital_rotation}"
        / cost
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = REPO / "tables" / args.molecule / "checklist_supplement_manifest.csv"

    default_pg = "C2v" if args.molecule == "h2o" else "D2h"
    point_group = args.point_group if args.point_group is not None else default_pg
    if point_group in ("", "none", "None"):
        point_group = None
    if orbital_rotation == "irrep" and not point_group:
        raise SystemExit("irrep orbital_rotation requires --point_group")

    exact_path = Path(
        args.exact_parity
        or (
            REPO
            / "exact"
            / f"{args.molecule}_norb{norb}_sto3g_exact.txt"
        )
    )
    _ensure_exact_parity(
        args.molecule, norb, exact_path, args.exact_parity_extra, force=True
    )

    chk = _ensure_chk(
        args.molecule,
        args.bond,
        args.basis,
        args.hoh_angle,
        point_group,
        require_irreps=(orbital_rotation == "irrep"),
    )
    summary: dict = {
        "molecule": args.molecule,
        "bond": args.bond,
        "regime": args.regime,
        "cost_function": cost,
        "chk": str(chk),
        "exact_parity": str(exact_path),
        "orbital_rotation": orbital_rotation,
        "point_group_ham": point_group,
        "repro": _repro_snapshot(),
        "protocols": {},
    }
    (out_dir / "repro.json").write_text(
        json.dumps(summary["repro"], indent=2), encoding="utf-8"
    )

    protocols = (
        ("mixed_disjoint", "greedy", True),
        ("mixed_overlap", "greedy", False),
        ("iterative", "iterative", True),
    )
    if args.only_iterative:
        protocols = (("iterative", "iterative", True),)

    for label, select, disjoint in protocols:
        tag = f"{label}_{cost}"
        print(f"[protocol] {tag}", flush=True)
        oo, trace_path = _oo_select(
            chk,
            out_dir / f"{tag}_oo.json",
            select=select,
            cost=cost,
            n_singles=n_singles,
            n_quartets=n_quartets,
            n_sym=n_sym,
            m_round=args.m_round,
            maxiter=args.maxiter,
            norb=norb,
            max_macroiterations=args.max_macroiterations,
            stable_span_iters=args.stable_span_iters,
            iterative_reference=args.iterative_reference,
            disjoint_orbitals=disjoint,
            exact_parity=exact_path,
            exact_sector=args.exact_sector,
            orbital_rotation=orbital_rotation,
        )
        parity_path = Path(oo.get("parity_output") or (out_dir / f"{tag}_parity.txt"))
        if not parity_path.is_file():
            acc = oo.get("accumulated_orbitals") or []
            if not acc and oo.get("singles") is not None:
                acc = [[int(s)] for s in oo.get("singles") or []]
                acc.extend([list(map(int, q)) for q in oo.get("quartets") or []])
            parity_path = out_dir / f"{tag}_parity.txt"
            _write_parity(parity_path, acc, norb)

        # Persist selection trace sidecar
        sel_trace = oo.get("selection_trace")
        if not sel_trace and oo.get("rounds"):
            sel_trace = [r.get("selection_trace") for r in oo["rounds"]]
        (out_dir / f"{tag}_selection_trace.json").write_text(
            json.dumps(sel_trace, indent=2), encoding="utf-8"
        )

        post = _metrics(
            oo,
            out_dir / f"{tag}_metrics_post.json",
            args.states_per_sector,
            out_dir / f"{tag}_metrics_post.log",
        )
        pre_payload = dict(oo)
        pre_payload["parity"] = str(parity_path)
        pre_payload["parity_output"] = str(parity_path)
        rot = oo.get("rotation")
        if isinstance(rot, list):
            pre_payload["rotation"] = [0.0] * len(rot)
        pre_payload["checklist_stage"] = "pre_oo_identity"
        pre = _metrics(
            pre_payload,
            out_dir / f"{tag}_metrics_pre.json",
            args.states_per_sector,
            out_dir / f"{tag}_metrics_pre.log",
        )

        ckpts = []
        if not args.skip_checkpoints:
            ckpts = _run_oo_checkpoints(
                oo,
                parity_path,
                trace_path,
                out_dir,
                tag,
                args.states_per_sector,
            )

        rounds_out = []
        if select == "iterative":
            rounds = oo.get("rounds") or []
            for idx, r in enumerate(rounds):
                las_orbs = r.get("las_orbitals") or r.get("orbitals") or []
                r_parity = out_dir / f"{tag}_round{idx}_parity.txt"
                _write_parity(r_parity, las_orbs, norb)
                payload = dict(oo)
                payload["parity"] = str(r_parity)
                payload["parity_output"] = str(r_parity)
                # Keep exact_masks from OO so metrics fix exact sector e.
                params = (r.get("optimization") or {}).get("parameters_after")
                if params is not None:
                    payload["rotation"] = list(params)
                payload["checklist_stage"] = f"iterative_round_{idx}_post_oo"
                m = _metrics(
                    payload,
                    out_dir / f"{tag}_round{idx}_metrics.json",
                    args.states_per_sector,
                    out_dir / f"{tag}_round{idx}_metrics.log",
                )
                rounds_out.append(
                    {
                        "index": idx,
                        "M": r.get("M"),
                        "las_orbitals": las_orbs,
                        "auxiliary_orbitals": r.get("auxiliary_orbitals"),
                        "stop_checklist": r.get("stop_checklist"),
                        "selection_trace": r.get("selection_trace"),
                        "clifford": r.get("clifford"),
                        "cost_before": (r.get("optimization") or {}).get("cost_before"),
                        "cost_after": (r.get("optimization") or {}).get("cost_after"),
                        "metrics": _summarize(m),
                    }
                )

        cliff = None
        if select == "iterative" and not args.skip_clifford:
            cliff = _clifford_sensitivity(oo, out_dir, tag, norb)

        u_path = _dump_rotation_matrix(oo, out_dir / f"{tag}_U.txt", norb)

        proto = {
            "select": select,
            "cost_function": cost,
            "oo_json": str(out_dir / f"{tag}_oo.json"),
            "parity": str(parity_path),
            "selection_rule": oo.get("selection_rule"),
            "singles": oo.get("singles"),
            "quartets": oo.get("quartets"),
            "accumulated_orbitals": oo.get("accumulated_orbitals"),
            "selection_trace_path": str(out_dir / f"{tag}_selection_trace.json"),
            "rotation_U": str(u_path) if u_path else None,
            "cost_before": oo.get("cost_before"),
            "cost_after": oo.get("cost_after"),
            "metrics_pre": _summarize(pre),
            "metrics_post": _summarize(post),
            "oo_checkpoints": ckpts,
            "rounds": rounds_out,
            "clifford_sensitivity": cliff,
        }
        summary["protocols"][tag] = proto
        _append_manifest(
            manifest,
            {
                "molecule": args.molecule,
                "bond": repr(float(args.bond)),
                "regime": args.regime,
                "cost_function": cost,
                "orbital_rotation": orbital_rotation,
                "protocol": tag,
                "K_pre": proto["metrics_pre"].get("K"),
                "K_post": proto["metrics_post"].get("K"),
                "dim_pre": proto["metrics_pre"].get("dim"),
                "dim_post": proto["metrics_post"].get("dim"),
                "cost_before": proto.get("cost_before"),
                "cost_after": proto.get("cost_after"),
                "n_oo_checkpoints": len(ckpts),
                "status": "ok",
            },
        )

    if not args.skip_cross:
        for pool_label in ("mixed_disjoint", "mixed_overlap", "iterative"):
            src = summary["protocols"][f"{pool_label}_{cost}"]
            parity = Path(src["parity"])
            tag = f"cross_{pool_label}_select{cost}_oo{other_cost}"
            print(f"[cross] {tag}", flush=True)
            oo, _trace = _oo_fixed_parity(
                chk,
                parity,
                out_dir / f"{tag}_oo.json",
                cost=other_cost,
                maxiter=args.maxiter,
                orbital_rotation=orbital_rotation,
            )
            post = _metrics(
                oo,
                out_dir / f"{tag}_metrics_post.json",
                args.states_per_sector,
                out_dir / f"{tag}_metrics_post.log",
            )
            summary["protocols"][tag] = {
                "discrete_pool_from": f"{pool_label}_{cost}",
                "oo_cost_function": other_cost,
                "cost_after": oo.get("cost_after"),
                "metrics_post": _summarize(post),
            }

    summary["elapsed_s"] = time.time() - started
    summary_path = out_dir / "supplement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote {summary_path} elapsed={summary['elapsed_s']:.1f}s")


if __name__ == "__main__":
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
