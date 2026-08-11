"""Shared PySCF hamiltonian chk ensure/rebuild for endpoint + checklist jobs.

Irrep jobs need point-group chks with reloadable MO irrep labels. Labels are
stored as ``*.chk.orbsym.txt`` (+ npy / HDF5 ``/orbsym``) by
``make_pyscf_hamiltonian.py``.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _exclusive_lock(lock_path: Path):
    """Cross-process exclusive lock (fcntl on Linux; no-op on Windows)."""

    @contextmanager
    def _cm():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = lock_path.open("a+", encoding="utf-8")
        try:
            if sys.platform != "win32":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if sys.platform != "win32":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    return _cm()


def _load_irreps(path: Path):
    """Load irreps; tolerate older orbital_rotation without retry helper."""
    try:
        from src.orbital_rotation import load_orbital_irreps_retry

        return load_orbital_irreps_retry(path, attempts=5, delay_s=1.0)
    except ImportError:
        from src.orbital_rotation import load_orbital_irreps

        return load_orbital_irreps(path)


def _diagnose(path: Path) -> str:
    try:
        from src.orbital_rotation import diagnose_chk_irreps

        return diagnose_chk_irreps(path)
    except Exception as exc:  # noqa: BLE001
        txt = Path(str(path) + ".orbsym.txt")
        npy = path.with_suffix(path.suffix + ".orbsym.npy")
        return (
            f"chk_exists={path.is_file()} "
            f"chk_size={path.stat().st_size if path.is_file() else 0} "
            f"txt={txt.is_file()} npy={npy.is_file()} "
            f"diag_exc={type(exc).__name__}:{exc}"
        )


def _has_irreps(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        labels = _load_irreps(path)
        return labels is not None and int(np.unique(labels).size) >= 2
    except Exception:  # noqa: BLE001
        return False


def _run(repo: Path, cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): see {log_path}")


def ensure_hamiltonian_chk(
    repo: Path,
    molecule: str,
    bond: float,
    basis: str,
    hoh_angle: float,
    point_group: str | None,
    *,
    require_irreps: bool = False,
    log_dir: str | Path = "results/_endpoint_ham",
) -> Path:
    """Return path to a usable hamiltonians/*.chk (rebuild under lock if needed)."""
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if molecule == "h2o":
        stem = f"H2O_OH{bond:.4f}_{hoh_angle:.4f}{basis}"
    else:
        stem = f"N2_bond{bond:.4f}{basis}"

    if require_irreps:
        if not point_group:
            raise ValueError(
                "irrep orbital rotation needs --point_group for a "
                "symmetry-adapted Hamiltonian"
            )
        pg = str(point_group).strip()
        stem = f"{stem}_{'auto' if pg.lower() == 'auto' else pg}"

    chk = Path(repo) / "hamiltonians" / f"{stem}.chk"
    log_path = Path(repo) / log_dir / f"{stem}.log"

    def _build(*, with_point_group: bool) -> None:
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
        if with_point_group and point_group:
            cmd.extend(["--point_group", str(point_group)])
        _run(Path(repo), cmd, log_path)

    if require_irreps:
        if _has_irreps(chk):
            return chk
        with _exclusive_lock(chk.with_suffix(chk.suffix + ".lock")):
            if _has_irreps(chk):
                print(f"[chk] reusing {chk.name} (built by another task)", flush=True)
                return chk
            print(
                f"[chk] building/repairing {chk.name} "
                f"({_diagnose(chk) if chk.is_file() else 'missing'})",
                flush=True,
            )
            # Build to a new file via make_pyscf atomic replace; do not unlink
            # first (avoids empty-window races). make_pyscf overwrites final.
            _build(with_point_group=True)
            if not _has_irreps(chk):
                raise RuntimeError(
                    f"chk {chk} still has no usable irrep labels after rebuild "
                    f"({_diagnose(chk)})"
                )
        return chk

    if chk.is_file():
        return chk

    with _exclusive_lock(chk.with_suffix(chk.suffix + ".lock")):
        if chk.is_file():
            return chk
        _build(with_point_group=False)
        if not chk.is_file():
            raise FileNotFoundError(chk)
    return chk
