"""
Parse the self-describing metadata out of an OpenMolcas ``RasOrb``
(``#INPORB``) orbital file: how many basis functions it has, and which of
its orbitals are inactive/core, active, or secondary/virtual.

This module deliberately does NOT parse the ``#ORB`` coefficient block.
Molcas's internal AO ordering for a given basis set is not the same as
PySCF's, so reusing raw ``RasOrb`` coefficients with PySCF-built AO
integrals would silently produce a wrong Hamiltonian.

(An earlier version of this pipeline planned to sidestep that by converting
RasOrb -> Molden with Pegamoid and reading the Molden file with PySCF. That
does not work for a bare RasOrb file: Pegamoid only merges InpOrb-format
coefficients into basis/geometry info it already has from an accompanying
HDF5 file -- it does not carry basis-set metadata on its own, and this
archive ships no HDF5. The approach that actually works is in
``hamiltonians/fes_si_ct1c00589/README.md``: reuse OpenMolcas itself
(GATEWAY+SEWARD+RASSCF with ``LumOrb``/``DMPOnly``) to transform the
integrals into the RasOrb's own orbital basis and dump a native FCIDUMP,
which sidesteps the AO-ordering problem entirely since no other code's AO
convention is ever involved.)

This module only supplies the piece that pipeline still needs from the
RasOrb file itself: which orbitals are inactive/active/secondary.

The ``#INDEX`` section of a RasOrb file looks like::

    #INDEX
    * 1234567890
    0 iiiiiiiiii
    1 iiiiiiiiii
    ...
    8 ii22222222
    9 2222222222
    0 22222222ss
    1 ssssssssss

Each line after the ``* 1234567890`` ruler is ``<decade digit><=10 type
chars>``; concatenating the type characters (in line order, dropping the
leading decade digit) gives one type code per orbital, in orbital order.
Standard OpenMolcas type codes: f=frozen, i=inactive, 1/2/3=RAS1/RAS2/RAS3
(active), s=secondary (virtual), d=deleted.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CORE_CODES = frozenset("fi")
_ACTIVE_CODES = frozenset("123")
_VIRTUAL_CODES = frozenset("sd")


@dataclass
class RasOrbPartition:
    nbas: int
    orb_types: str  # one character per orbital, length nbas
    occ: np.ndarray  # occupation numbers from #OCC, length nbas (or empty)

    @property
    def ncore(self) -> int:
        return sum(1 for c in self.orb_types if c in _CORE_CODES)

    @property
    def ncas(self) -> int:
        return sum(1 for c in self.orb_types if c in _ACTIVE_CODES)

    @property
    def nvirt(self) -> int:
        return sum(1 for c in self.orb_types if c in _VIRTUAL_CODES)

    @property
    def active_slice(self) -> slice:
        return slice(self.ncore, self.ncore + self.ncas)

    def nelec_active(self) -> float:
        """Sum of occupation numbers in the active block (should be close
        to an integer; the caller should round and sanity-check it)."""
        if self.occ.size == 0:
            raise ValueError("no #OCC section was found in this RasOrb file")
        return float(np.sum(self.occ[self.active_slice]))


def _read_index_block(lines, start):
    """lines[start] is the '* 1234567890' ruler line; consume decade rows
    until a non-matching line (next '#' section or EOF)."""
    codes = []
    i = start + 1
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line or line.startswith("#"):
            break
        body = line.strip()
        # "<digit> <=10 type chars>", digit may or may not be separated by
        # whitespace from the type characters.
        parts = body.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            break
        codes.append(parts[1].strip())
        i += 1
    return "".join(codes), i


def _read_float_block(lines, start):
    """lines[start] is a '* ...' comment line introducing a block of
    whitespace-separated floats that continues until the next '#'/'*' line
    or EOF."""
    vals = []
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("*"):
            break
        vals.extend(float(tok) for tok in line.split())
        i += 1
    return vals, i


def parse_rasorb_partition(path) -> RasOrbPartition:
    path = Path(path)
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    nbas = None
    orb_types = ""
    occ = np.array([], dtype=float)

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#INFO"):
            # skip comment lines, then the header-int line, then read
            # exactly one nbas-per-symmetry line. We only support the
            # single-symmetry (C1 / nosym) case used by this workflow.
            j = i + 1
            while j < len(lines) and lines[j].startswith("*"):
                j += 1
            j += 1  # header-int line ("nSym ..." style line)
            nbas_tokens = lines[j].split()
            if len(nbas_tokens) != 1:
                raise ValueError(
                    f"{path}: expected a single-symmetry RasOrb file "
                    f"(nosym/C1), got NBAS line {lines[j]!r}"
                )
            nbas = int(nbas_tokens[0])
            i = j + 1
            continue
        if line.startswith("#OCC"):
            vals, i = _read_float_block(lines, i + 1)
            occ = np.array(vals, dtype=float)
            continue
        if line.startswith("#INDEX"):
            # next line is the "* 1234567890" ruler
            orb_types, i = _read_index_block(lines, i + 1)
            continue
        i += 1

    if nbas is None:
        raise ValueError(f"{path}: could not find #INFO/NBAS header")
    if len(orb_types) != nbas:
        raise ValueError(
            f"{path}: #INDEX gave {len(orb_types)} orbital type codes, "
            f"expected {nbas} (from #INFO header)"
        )
    if occ.size and occ.size != nbas:
        raise ValueError(
            f"{path}: #OCC gave {occ.size} values, expected {nbas}"
        )

    return RasOrbPartition(nbas=nbas, orb_types=orb_types, occ=occ)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        part = parse_rasorb_partition(p)
        line = (
            f"{p}: nbas={part.nbas} ncore={part.ncore} ncas={part.ncas} "
            f"nvirt={part.nvirt}"
        )
        if part.occ.size:
            line += f" nelec_active={part.nelec_active():.6f}"
        print(line)
