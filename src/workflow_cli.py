"""Shared CLI vocabulary for optimize and metrics.

``--reference`` (optimize only)
    Which wavefunction / cost engine to use
    (``fci`` / ``hf`` → CI costs; ``dmrg`` → Block2 MPS costs).

``--orbital_rotation`` (optimize / rotate / ``--U`` tools)
    ``full`` — all ``binom(n,2)`` planes (default).
    ``irrep`` — only intra-irrep pairs; needs a symmetry-adapted
    Hamiltonian from ``make_pyscf_hamiltonian.py --point_group``.

``--backend`` (metrics only)
    Sector eigensolver: ``fci``, ``davidson``, or ``dmrg``.

Metrics K methods (``--coupled_energy_method``)
    ``perturbation``  one-shot PT ordering (no overlap reference needed)
    ``reference``     overlap ordering against a DMRG wavefunction only

``dmrg`` always means Block2. Shared flags: ``--bond_dim``,
``--wavefunction_dir``, ``--n_threads``.
"""

from __future__ import annotations

import argparse
import sys

REFERENCE_CHOICES = ("fci", "hf", "dmrg")
METRICS_BACKEND_CHOICES = ("fci", "dmrg", "davidson")
ORBITAL_ROTATION_CHOICES = ("full", "irrep")
SELECT_CHOICES = ("none", "greedy", "iterative")
CANDIDATE_CHOICES = ("senquart", "seniority")
GREEDY_COST_FUNCTIONS = ("NC", "variance")
SECTOR_COST_FUNCTIONS = ("decoupled", "fixed_sector", "switching_sector")
SELECT_POOL_MODES = ("greedy", "iterative")

OPTIMIZE_EPILOG = """
--reference picks both the wavefunction and the cost engine
-----------------------------------------------------------
  --reference fci     PySCF FCI CI vector + ffsim costs (default)
  --reference hf      Hartree-Fock CI vector + ffsim costs
  --reference dmrg    Block2 MPS + MPS-native NC/variance

  Sector energy costs (decoupled / fixed_sector / switching_sector)
  require --reference fci or hf (CI / ffsim path).

--orbital_rotation packing
--------------------------
  --orbital_rotation full   SO(n), binom(n,2) angles (default)
  --orbital_rotation irrep  intra-irrep pairs only (needs --point_group chk)

--select greedy|iterative
-------------------------
  Build an independent Z-type set of size --n_sym, then run OO on that parity
  matrix. Only valid for --cost_function NC|variance. Omit the parity
  positional when selecting.

  greedy     one-shot Kruskal over {seniority, quartet} (--candidates)
  iterative  fixed-M NC-ranked LAS search with exact-parity quotienting,
             warm-started OO, and Clifford frame update (complete N-r
             ranked basis; --m_round is deprecated / ignored)


examples
--------
  python optimize_symmetries.py mol.FCIDUMP parity.txt
  python optimize_symmetries.py mol.chk parity.txt --orbital_rotation irrep
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference hf
  python optimize_symmetries.py mol.FCIDUMP parity.txt --reference dmrg --bond_dim 250
  python optimize_symmetries.py mol.chk --select greedy --n_sym 4 --cost_function NC
  python optimize_symmetries.py mol.chk --select iterative --n_sym 4 --m_round 2
"""

METRICS_EPILOG = """
--backend (sector eigensolver)
-----------------------------
  --backend fci         eigsh/eigh on each sector block (default)
  --backend davidson    PySCF Davidson on the same blocks
  --backend dmrg        Block2 sector-targeted DMRG

--coupled_energy_method (K selection; CI backends only)
-------------------------------------------------------
  perturbation   one-shot PT ordering (default); needs only an energy target
  reference      overlap ordering vs a reference wavefunction
                 (--overlap_reference fci|dmrg; default fci)

  --backend dmrg always uses one-shot PT for K.
  --solver is an alias of --backend.

examples
--------
  python metrics.py oo.json
  python metrics.py oo.json --backend fci --coupled_energy_method reference --overlap_reference fci
  python metrics.py oo.json --backend davidson --coupled_energy_method perturbation
  python metrics.py oo.json --coupled_energy_method reference --overlap_reference dmrg --bond_dim 250
  python metrics.py oo.json --backend dmrg --bond_dim 250 --penalty 30
"""


def add_dmrg_common_args(parser: argparse.ArgumentParser) -> None:
    """Bond dimension / store / threads used by any ``dmrg`` path."""
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=250,
        help="Block2 DMRG bond dimension (dmrg paths / overlap-K reference)",
    )
    parser.add_argument(
        "--wavefunction_dir",
        default=None,
        help="directory for Block2 MPS files (reuse across runs)",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Block2 OpenMP thread count",
    )


def add_orbital_rotation_arg(parser: argparse.ArgumentParser) -> None:
    """``--orbital_rotation {full,irrep}`` shared by optimize entry points."""
    parser.add_argument(
        "--orbital_rotation",
        choices=ORBITAL_ROTATION_CHOICES,
        default="full",
        metavar="{full,irrep}",
        help=(
            "Orbital-rotation packing: full=SO(n) upper triangle (default); "
            "irrep=only intra-irrep pairs (needs a symmetry-adapted Hamiltonian "
            "from make_pyscf_hamiltonian.py --point_group)."
        ),
    )


def add_greedy_select_args(parser: argparse.ArgumentParser) -> None:
    """``--select greedy|iterative`` pool / cardinality / parity-output flags."""
    parser.add_argument(
        "--select",
        choices=SELECT_CHOICES,
        default="none",
        metavar="{none,greedy,iterative}",
        help=(
            "Symmetry selection before OO: none=use provided parity/--seniority "
            "(default); greedy=one-shot Kruskal over seniority/quartet candidates; "
            "iterative=selection + warm-started OO + external Clifford frame "
            "updates. Both use "
            "the same NC or variance metric as --cost_function."
        ),
    )
    parser.add_argument(
        "--n_sym",
        type=int,
        default=None,
        help="number of independent symmetries to select (--select greedy|iterative)",
    )
    parser.add_argument(
        "--n_singles",
        type=int,
        default=None,
        help=(
            "seniority quota for --select greedy senquart selection "
            "(with --n_quartets; n_sym defaults to their sum)"
        ),
    )
    parser.add_argument(
        "--n_quartets",
        type=int,
        default=None,
        help=(
            "quartet quota for --select greedy senquart selection "
            "(with --n_singles; n_sym defaults to their sum)"
        ),
    )
    parser.add_argument(
        "--candidates",
        choices=CANDIDATE_CHOICES,
        default="senquart",
        metavar="{senquart,seniority}",
        help=(
            "Candidate pool for --select greedy: senquart=seniorities+quartets "
            "(default); seniority=local seniorities only. Ignored for iterative "
            "(always senquart per GF(2) frame)."
        ),
    )
    parser.add_argument(
        "--m_round",
        type=int,
        default=2,
        help=(
            "deprecated for --select iterative (fixed-M ranked scan ignores "
            "m_round; kept for CLI compatibility, must be >= 1 if set)"
        ),
    )
    parser.add_argument(
        "--max_macroiterations",
        type=int,
        default=None,
        help=(
            "max discrete frame updates for --select iterative "
            "(default: max(2*norb, 8))"
        ),
    )
    parser.add_argument(
        "--stable_span_iters",
        type=int,
        default=2,
        help=(
            "for --select iterative: consecutive macros where all five stop "
            "conditions hold before stopping (default: 2)"
        ),
    )
    parser.add_argument(
        "--oo_stop_tol",
        type=float,
        default=1e-6,
        help="iterative OO objective change tolerance |C^t-C^{t-1}|",
    )
    parser.add_argument(
        "--oo_step_tol",
        type=float,
        default=1e-5,
        help="iterative OO step tolerance ||x^t-x^{t-1}||_inf",
    )
    parser.add_argument(
        "--rank_replace_tol",
        type=float,
        default=1e-4,
        help=(
            "iterative ranking: LAS cost-sum improvement below this is "
            "'no meaningful replacement'"
        ),
    )
    parser.add_argument(
        "--disjoint_orbitals",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "for --select greedy with --n_singles/--n_quartets: 1=reject "
            "overlapping orbital supports (default Mixed disjoint); "
            "0=quota Mixed with GF(2) independence only (overlap allowed)"
        ),
    )
    parser.add_argument(
        "--exact_parity",
        default=None,
        help=(
            "optional binary (r x norb) matrix file of exact parity rows E. "
            "Endpoint/checklist STO-3G defaults write report PG rows "
            "(H2O: Q_B1/Q_B2; N2: Q_pix/Q_piy/Q_u). Otherwise default is "
            "all-ones total particle parity. Exact rows seed the GF(2) prior "
            "for both Mixed and Iterative selection and are used for "
            "Clifford tapering / sector dimensions in metrics."
        ),
    )
    parser.add_argument(
        "--exact_sector",
        default=None,
        help=(
            "exact-sector eigenvalue label as an r-bit string (e.g. 0 or 01); "
            "default picks the densest physical tapered sector"
        ),
    )
    parser.add_argument(
        "--iterative_reference",
        choices=("exact_taper", "fci_rotate"),
        default="fci_rotate",
        metavar="{exact_taper,fci_rotate}",
        help=(
            "reference for iterative *ranking* scores: "
            "fci_rotate=frozen rotated FCI/HF vector (default, same as mixed OO); "
            "exact_taper=Route A ApproxGroundState in the exact Clifford sector. "
            "Orbital optimization always uses fci_rotate."
        ),
    )
    parser.add_argument(
        "--parity_output",
        default=None,
        help=(
            "write selected parity matrix here "
            "(default: <outname>_parity.txt or parity_greedy.txt / parity_iterative.txt)"
        ),
    )
    parser.add_argument(
        "--results_csv",
        default=None,
        help=(
            "append a locked summary row when OO finishes "
            "(supports concurrent SLURM array writers)"
        ),
    )


def resolve_select_n_sym(
    *,
    select: str,
    n_sym: int | None,
    n_singles: int | None = None,
    n_quartets: int | None = None,
) -> int | None:
    """Resolve effective ``n_sym`` after optional greedy quotas."""
    if n_singles is None:
        n_singles = _cli_flag_int("--n_singles")
    if n_quartets is None:
        n_quartets = _cli_flag_int("--n_quartets")
    if n_sym is None:
        n_sym = _cli_flag_int("--n_sym")
    if select == "greedy" and n_sym is None and n_singles is not None and n_quartets is not None:
        return int(n_singles) + int(n_quartets)
    return n_sym


def _cli_flag_int(flag: str) -> int | None:
    """Read ``--flag N`` from ``sys.argv`` (survives stale callers that omit kwargs)."""
    argv = sys.argv
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    try:
        return int(argv[index + 1])
    except ValueError:
        return None


def validate_greedy_cli_args(
    *,
    select: str,
    n_sym: int | None,
    cost_function: str,
    parity: str | None = None,
    seniority: bool = False,
    symmetry_manifest: str | None = None,
    m_round: int | None = None,
    n_singles: int | None = None,
    n_quartets: int | None = None,
    candidates: str | None = None,
) -> None:
    """Raise ``ValueError`` when select / parity CLI combinations are invalid.

    Callers typically wrap this with ``parser.error(str(exc))``.
    """
    # Stale optimize_*.py may register --n_singles/--n_quartets but forget to
    # forward them into this helper; recover from argv so quotas still work.
    if n_singles is None:
        n_singles = _cli_flag_int("--n_singles")
    if n_quartets is None:
        n_quartets = _cli_flag_int("--n_quartets")
    if n_sym is None:
        n_sym = _cli_flag_int("--n_sym")

    if select in SELECT_POOL_MODES:
        has_quota = n_singles is not None or n_quartets is not None
        if has_quota:
            if n_singles is None or n_quartets is None:
                raise ValueError("--n_singles and --n_quartets must be set together")
            if int(n_singles) < 0 or int(n_quartets) < 0:
                raise ValueError("--n_singles and --n_quartets must be non-negative")
            if int(n_singles) + int(n_quartets) <= 0:
                raise ValueError("--n_singles + --n_quartets must be positive")
            cand = (candidates or "senquart").lower()
            if select == "greedy" and cand != "senquart":
                raise ValueError(
                    "quota selection (--n_singles/--n_quartets) requires "
                    "--candidates senquart"
                )
            if select == "iterative" and n_sym is None:
                raise ValueError(
                    "--select iterative requires --n_sym "
                    "(quotas apply only to greedy selection)"
                )

        effective_n_sym = resolve_select_n_sym(
            select=select,
            n_sym=n_sym,
            n_singles=n_singles,
            n_quartets=n_quartets,
        )
        if effective_n_sym is None:
            if select == "greedy":
                raise ValueError(
                    "--select greedy requires --n_sym or both --n_singles and --n_quartets"
                )
            raise ValueError(f"--select {select} requires --n_sym")
        if effective_n_sym <= 0:
            raise ValueError(f"--n_sym must be positive, got {effective_n_sym}")
        if (
            has_quota
            and select == "greedy"
            and n_sym is not None
            and int(n_sym) != int(n_singles) + int(n_quartets)
        ):
            raise ValueError(
                f"--n_sym={n_sym} must equal --n_singles+--n_quartets="
                f"{int(n_singles) + int(n_quartets)}"
            )
        if cost_function not in GREEDY_COST_FUNCTIONS:
            raise ValueError(
                f"--select {select} only supports --cost_function NC or variance "
                f"(got {cost_function!r}); sector energy costs are not additive"
            )
        if seniority:
            raise ValueError(
                f"--select {select} is incompatible with --seniority; "
                "use --candidates seniority (greedy) or omit --seniority"
            )
        if parity is not None:
            raise ValueError(
                f"--select {select} builds the parity matrix; omit the parity "
                "positional argument (use --parity_output to save it)"
            )
        if select == "iterative":
            round_val = 2 if m_round is None else int(m_round)
            if round_val < 1:
                raise ValueError(f"--m_round must be >= 1, got {round_val}")
        return

    # select == none: existing supply rules for optimize entry points
    if parity is None and not seniority and symmetry_manifest is None:
        raise ValueError(
            "supply a parity matrix file, --seniority, --symmetry_manifest, "
            "or --select greedy|iterative"
        )


def add_optimize_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` for ``optimize_symmetries.py`` (no ``--backend``)."""
    parser.add_argument(
        "--reference",
        choices=REFERENCE_CHOICES,
        default="fci",
        metavar="{fci,hf,dmrg}",
        help=(
            "REFERENCE STATE and cost engine: "
            "fci=PySCF FCI + ffsim costs (default); "
            "hf=Hartree-Fock + ffsim costs; "
            "dmrg=Block2 MPS + MPS-native NC/variance. "
            "Sector energy costs need fci or hf."
        ),
    )
    add_orbital_rotation_arg(parser)
    add_greedy_select_args(parser)
    add_dmrg_common_args(parser)
    _attach_epilog(parser, OPTIMIZE_EPILOG)


def add_metrics_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--backend`` for ``metrics.py`` (no ``--reference``)."""
    parser.add_argument(
        "--backend",
        "--solver",
        dest="backend",
        choices=METRICS_BACKEND_CHOICES,
        default="fci",
        metavar="{fci,davidson,dmrg}",
        help=(
            "SECTOR SOLVER: fci=eigsh/eigh on each sector block (default); "
            "davidson=PySCF Davidson on the same blocks; "
            "dmrg=Block2 sector-targeted DMRG. "
            "--solver is a deprecated alias of --backend."
        ),
    )
    add_dmrg_common_args(parser)
    _attach_epilog(parser, METRICS_EPILOG)


def optimize_cost_engine(reference: str) -> str:
    """Derived cost engine label for optimize banners / JSON."""
    return "dmrg" if reference == "dmrg" else "statevector"


def print_workflow_banner(script: str, reference: str | None = None, backend: str | None = None, **extra) -> None:
    """Print a short resolved-settings banner so the run mode is obvious."""
    lines = []
    if reference is not None:
        lines.append(
            f"[workflow] reference={reference}  (wavefunction / energy used as truth)"
        )
    if script == "optimize" and reference is not None:
        engine = backend or optimize_cost_engine(reference)
        lines.append(f"[workflow] cost_engine={engine}  (from --reference)")
    elif script == "metrics" and backend is not None:
        lines.append(f"[workflow] backend={backend}  (sector solver)")
    for key, value in extra.items():
        if value is not None:
            lines.append(f"[workflow] {key}={value}")
    if lines:
        print("\n".join(lines), flush=True)


def _attach_epilog(parser: argparse.ArgumentParser, epilog: str) -> None:
    """Append recipes to the parser epilog without clobbering an existing one."""
    existing = parser.epilog or ""
    parser.epilog = (existing + "\n" + epilog).strip()
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
