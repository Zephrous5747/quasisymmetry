"""Orbital-rotation packing: full SO(n) or intra-irrep (point-group) pairs.

Parameters ``x`` pack into a skew generator ``A`` with free entries only on the
allowed planes ``(i, j)``, then ``U = expm(A)``. Full mode uses every upper-
triangle pair (same order as ``np.triu_indices(norb, k=1)``). Irrep mode keeps
only pairs that share an irrep label, so
``N_sym = sum_Gamma binom(|Gamma|, 2)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.linalg

PairList = list[tuple[int, int]]

_PG_SUFFIX_RE = re.compile(r"_(C2v|D2h|C2h|D2|C2|Cs|Ci|C1|auto)$", re.IGNORECASE)


def full_pairs(norb: int) -> PairList:
    """All upper-triangle orbital pairs, in ``triu_indices`` order."""
    rows, cols = np.triu_indices(norb, k=1)
    return list(zip(map(int, rows), map(int, cols)))


def irrep_pairs(irreps: Sequence[int]) -> PairList:
    """Intra-irrep pairs only (same label), still in ascending ``(i, j)`` order."""
    irreps = np.asarray(irreps)
    norb = len(irreps)
    pairs: PairList = []
    for i in range(norb):
        for j in range(i + 1, norb):
            if irreps[i] == irreps[j]:
                pairs.append((i, j))
    return pairs


def n_params(norb: int, pairs: PairList | None = None) -> int:
    """Number of free rotation angles for the given packing."""
    if pairs is None:
        return norb * (norb - 1) // 2
    return len(pairs)


def params_to_U(
    x: np.ndarray,
    norb: int,
    pairs: PairList | None = None,
) -> np.ndarray:
    """Build ``U = expm(A(x))`` from upper-triangle (or restricted) parameters."""
    x = np.asarray(x, dtype=float).ravel()
    expected = n_params(norb, pairs)
    if x.size != expected:
        raise ValueError(
            f"rotation parameter length {x.size} does not match packing "
            f"size {expected} (norb={norb}, restricted={pairs is not None})"
        )
    generator = np.zeros((norb, norb), dtype=float)
    if pairs is None:
        generator[np.triu_indices(norb, k=1)] = x
    else:
        for k, (i, j) in enumerate(pairs):
            generator[i, j] = x[k]
    generator -= generator.T
    return scipy.linalg.expm(generator)


def _orbsym_sidecar_path(chk: Path | str) -> Path:
    return Path(chk).with_suffix(Path(chk).suffix + ".orbsym.npy")


def _orbsym_txt_path(chk: Path | str) -> Path:
    return Path(str(chk) + ".orbsym.txt")


def save_orbsym_sidecar(chk: Path | str, labels) -> Path:
    """Persist irrep ids next to the chk (HDF5 strips ``mo_coeff.orbsym``).

    Writes both a ``.orbsym.npy`` and a plain ``.orbsym.txt`` (space-separated
    ints). The text form is the preferred reload source on shared filesystems.
    """
    import os

    arr = np.asarray(labels, dtype=np.int32).ravel()
    chk_path = Path(chk)
    npy = _orbsym_sidecar_path(chk_path)
    txt = _orbsym_txt_path(chk_path)

    tmp_npy = Path(str(npy) + f".tmp.{os.getpid()}.npy")
    np.save(tmp_npy, arr)
    os.replace(tmp_npy, npy)
    try:
        with npy.open("rb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass

    tmp_txt = Path(str(txt) + f".tmp.{os.getpid()}")
    tmp_txt.write_text(" ".join(str(int(x)) for x in arr) + "\n", encoding="utf-8")
    with tmp_txt.open("rb") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_txt, txt)
    try:
        with txt.open("rb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    return txt


def load_orbital_irreps(molpath: str | Path) -> np.ndarray | None:
    """Load per-orbital irrep labels from a symmetry-adapted ``.chk`` or FCIDUMP.

    Returns ``None`` when no useful point-group labels are available (no
    symmetry on the molecule, or FCIDUMP ``ORBSYM`` is missing / all identical).
    """
    path = Path(molpath)
    if path.suffix == ".chk":
        return _irreps_from_chk(path)
    if path.suffix == ".FCIDUMP" or path.name.endswith("FCIDUMP"):
        return _irreps_from_fcidump(path)
    raise ValueError("molpath must be a .chk or FCIDUMP file")


def diagnose_chk_irreps(molpath: str | Path) -> str:
    """Short diagnostic string for irrep-load failures."""
    path = Path(molpath)
    parts = [
        f"chk_exists={path.is_file()}",
        f"chk_size={path.stat().st_size if path.is_file() else 0}",
        f"txt={_orbsym_txt_path(path).is_file()}",
        f"npy={_orbsym_sidecar_path(path).is_file()}",
    ]
    try:
        labels = load_orbital_irreps(path)
        parts.append(f"labels={None if labels is None else labels.tolist()}")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"load_exc={type(exc).__name__}:{exc}")
    return ", ".join(parts)


def load_orbital_irreps_retry(
    molpath: str | Path,
    *,
    attempts: int = 5,
    delay_s: float = 2.0,
) -> np.ndarray | None:
    """Load irreps with short retries (NFS create visibility)."""
    import time

    path = Path(molpath)
    last: np.ndarray | None = None
    for i in range(max(1, int(attempts))):
        last = load_orbital_irreps(path)
        if last is not None and int(np.unique(last).size) >= 2:
            # Ensure text sidecar exists for later tasks.
            txt = _orbsym_txt_path(path)
            if not txt.is_file():
                try:
                    save_orbsym_sidecar(path, last)
                except Exception:  # noqa: BLE001
                    pass
            return last
        if i + 1 < attempts:
            time.sleep(float(delay_s))
    return last


def _point_group_hint(path: Path, mol) -> str | None:
    """Best-effort point-group name from mol attrs or ``*_GROUP.chk`` stem."""
    for attr in ("groupname", "topgroup"):
        g = getattr(mol, attr, None)
        if g not in (None, "", "C1", "c1"):
            return str(g)
    m = _PG_SUFFIX_RE.search(path.stem)
    if m:
        tag = m.group(1)
        return None if tag.lower() == "auto" else tag
    return None


def _finalize_labels(labels) -> np.ndarray | None:
    if labels is None:
        return None
    labels = _canonicalize_irreps(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return labels


def _irreps_from_chk(path: Path) -> np.ndarray | None:
    """Load irrep labels from a PySCF ``.chk``.

    Order: sidecar ``.chk.orbsym.npy``, HDF5 ``/orbsym``, then reconstruct.
    """
    import pyscf
    from pyscf import symm
    from pyscf.lib import chkfile as pyscf_chk

    # 0) Plaintext sidecar (preferred on NFS), then npy, then HDF5 / reconstruct.
    txt = _orbsym_txt_path(path)
    if txt.is_file():
        try:
            raw = np.asarray(
                [int(x) for x in txt.read_text(encoding="utf-8").split()],
                dtype=int,
            )
            done = _finalize_labels(raw)
            if done is not None:
                return done
        except Exception:  # noqa: BLE001
            pass

    side = _orbsym_sidecar_path(path)
    if side.is_file():
        try:
            done = _finalize_labels(np.load(side))
            if done is not None:
                return done
        except Exception:  # noqa: BLE001
            pass

    # 1) Explicit HDF5 dataset.
    try:
        labels = pyscf_chk.load(str(path), "orbsym")
        done = _finalize_labels(labels)
        if done is not None:
            return done
    except Exception:  # noqa: BLE001
        pass

    mol = pyscf.lib.chkfile.load_mol(str(path))
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(str(path))

    # 2) In-memory tag (present only if somehow preserved).
    labels = getattr(mf, "orbsym", None)
    if labels is None:
        labels = getattr(getattr(mf, "mo_coeff", None), "orbsym", None)
    done = _finalize_labels(labels)
    if done is not None:
        return done

    # 3) Re-enable point-group symmetry and re-label MOs.
    group = _point_group_hint(path, mol)
    if group:
        try:
            mol.symmetry = True if str(group).lower() == "auto" else group
            mol.build(False, False)
            mf = pyscf.scf.RHF(mol)
            mf.update_from_chk(str(path))
            if hasattr(mf, "get_orbsym"):
                labels = mf.get_orbsym()
            elif getattr(mol, "symm_orb", None) is not None:
                labels = symm.label_orb_symm(
                    mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff, check=False
                )
            done = _finalize_labels(labels)
            if done is not None:
                return done
        except Exception:  # noqa: BLE001
            pass

    # 4) Last resort: label with whatever symm_orb survived load_mol.
    try:
        if getattr(mol, "symm_orb", None) is not None:
            irrep_ref = getattr(mol, "irrep_id", None) or getattr(
                mol, "irrep_name", None
            )
            if irrep_ref is not None:
                labels = symm.label_orb_symm(
                    mol, irrep_ref, mol.symm_orb, mf.mo_coeff, check=False
                )
                return _finalize_labels(labels)
    except Exception:  # noqa: BLE001
        return None
    return None


def _irreps_from_fcidump(path: Path) -> np.ndarray | None:
    import pyscf

    data = pyscf.tools.fcidump.read(str(path), verbose=False)
    labels = data.get("ORBSYM")
    if labels is None:
        return None
    labels = np.asarray(labels).ravel()
    # Placeholder dumps often set every ORBSYM entry to 1.
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return _canonicalize_irreps(labels)


def _canonicalize_irreps(labels) -> np.ndarray:
    """Map string or int irrep labels to dense integer ids (order of first appearance)."""
    arr = np.asarray(labels)
    if arr.dtype.kind in "iuf":
        return arr.astype(int).ravel()
    mapping: dict = {}
    flat = arr.ravel()
    out = np.empty(flat.size, dtype=int)
    next_id = 0
    for i, lab in enumerate(flat):
        key = str(lab)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out[i] = mapping[key]
    return out


def resolve_orbital_rotation(
    mode: str,
    molpath: str | Path,
    norb: int,
) -> tuple[PairList | None, np.ndarray | None]:
    """Resolve ``(pairs, irreps)`` for ``full`` or ``irrep`` packing.

    ``pairs is None`` means full ``SO(n)`` packing.
    """
    mode = (mode or "full").lower()
    if mode == "full":
        return None, None
    if mode != "irrep":
        raise ValueError("orbital_rotation mode must be 'full' or 'irrep'")

    irreps = load_orbital_irreps(molpath)
    if irreps is None:
        raise ValueError(
            "irrep orbital rotations require a symmetry-adapted Hamiltonian "
            "(regenerate with make_pyscf_hamiltonian.py --point_group, or an "
            "FCIDUMP with distinct ORBSYM labels)"
        )
    if len(irreps) != norb:
        raise ValueError(
            f"irrep length {len(irreps)} does not match norb={norb}"
        )
    pairs = irrep_pairs(irreps)
    if not pairs:
        raise ValueError(
            "irrep packing has no free pairs (all irrep blocks have size 1)"
        )
    return pairs, np.asarray(irreps, dtype=int)


def pairs_from_oo_data(data: dict, norb: int) -> PairList | None:
    """Rebuild packing from OO JSON fields (``orbital_rotation`` / ``irreps``)."""
    mode = str(data.get("orbital_rotation", "full")).lower()
    if mode == "full":
        return None
    if mode != "irrep":
        raise ValueError(f"unknown orbital_rotation mode in OO data: {mode!r}")
    irreps = data.get("irreps")
    if irreps is None:
        molpath = data.get("molpath")
        if molpath is None:
            raise ValueError(
                "OO JSON has orbital_rotation=irrep but neither irreps nor molpath"
            )
        pairs, _ = resolve_orbital_rotation("irrep", molpath, norb)
        return pairs
    return irrep_pairs(irreps)


def rotation_from_oo_data(data: dict, norb: int) -> np.ndarray:
    """Build ``U`` from OO JSON ``rotation`` (+ optional irrep packing)."""
    x = np.asarray(data.get("rotation", []), dtype=float)
    if x.size == 0:
        return np.eye(norb)
    pairs = pairs_from_oo_data(data, norb)
    return params_to_U(x, norb, pairs)
