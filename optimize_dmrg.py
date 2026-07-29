"""Orbital optimization with MPS-native block2 costs (FCIDUMP-friendly).

Same role as ``optimize_symmetries.py --reference dmrg``, but imports no
pyscf/ffsim so it runs on machines that only have block2 (e.g. native
Windows). Output is an ``x_opt`` text file (JSON metadata header + parameter
vector) consumed by ``metrics.py``, ``rotate_fcidump.py`` and
``solve_dmrg.py --U``.

Supports ``--orbital_rotation {full,irrep}`` (default ``full``). Irrep mode
needs a symmetry-adapted FCIDUMP/chk with distinct ORBSYM / point-group labels.

Example::

    python optimize_dmrg.py hamiltonians/sentest_5_d754.FCIDUMP parity.txt \\
        --cost_function NC --bond_dim 200 --maxiter 20
    python optimize_dmrg.py mol.chk parity.txt --orbital_rotation irrep
    python optimize_dmrg.py mol.FCIDUMP --select greedy --n_sym 4 --cost_function NC
    python optimize_dmrg.py mol.FCIDUMP --select greedy --n_singles 3 --n_quartets 2
    python optimize_dmrg.py mol.FCIDUMP --select iterative --n_sym 4 --m_round 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize

from src.dmrg_costs import MultiplyConfig, build_dmrg_orbital_costs
from src.dmrg_solver import DMRGConfig
from src.greedy_selection import (
    default_parity_output_path,
    select_from_pool,
    select_senquart_quota,
    write_parity_matrix,
)
from src.iterative_pool import select_iterative_pool
from src.orbital_rotation import n_params, resolve_orbital_rotation
from src.results_table import append_result_row
from src.workflow_cli import (
    add_greedy_select_args,
    add_orbital_rotation_arg,
    resolve_select_n_sym,
    validate_greedy_cli_args,
)


def callback(intermediate_result):
    print(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        end=" ",
    )
    # SciPy < 1.11 passes the parameter vector; newer versions pass OptimizeResult.
    if hasattr(intermediate_result, "fun"):
        print("{0:4.6f}".format(intermediate_result.fun))
    else:
        print("(iter)")


def _results_row(
    args,
    *,
    status: str,
    final_cost,
    selected_costs,
    parity_output: str,
    outname: str,
    elapsed_s: float,
    message: str = "",
) -> dict:
    return {
        "molecule": args.molpath,
        "select": args.select,
        "cost_function": args.cost_function,
        "n_singles": "" if args.n_singles is None else int(args.n_singles),
        "n_quartets": "" if args.n_quartets is None else int(args.n_quartets),
        "n_sym": "" if args.n_sym is None else int(args.n_sym),
        "m_round": int(args.m_round) if args.select == "iterative" else "",
        "final_cost": final_cost,
        "selected_costs": list(selected_costs) if selected_costs else "",
        "parity_output": parity_output,
        "outname": outname,
        "status": status,
        "elapsed_s": round(float(elapsed_s), 3),
        "job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID", ""),
        "task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "message": message,
    }


def _run_optimize(args) -> None:
    store_dir = args.wavefunction_dir or str(
        Path("wavefunctions") / Path(args.molpath).stem
    )
    selection_meta = None
    iterative_res = None
    selected_costs: tuple[float, ...] = ()
    parity_out = ""
    started = time.time()
    rotation_irreps = None

    if args.select in ("greedy", "iterative"):
        placeholder = np.ones((1, 1), dtype=int)
        costs, result, solver = build_dmrg_orbital_costs(
            args.molpath,
            placeholder,
            store_dir=store_dir,
            config=DMRGConfig(max_bond_dim=args.bond_dim, n_sweeps=args.n_sweeps),
            multiply=MultiplyConfig(
                bond_dim=args.multiply_bond_dim,
                n_sweeps=args.multiply_sweeps,
            ),
            reuse=not args.no_reuse,
            n_threads=args.n_threads,
        )
        rotation_pairs, rotation_irreps = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, solver.n_sites
        )
        costs.pairs = rotation_pairs
        x0 = (
            np.zeros(n_params(solver.n_sites, rotation_pairs))
            if args.x0 is None
            else np.loadtxt(args.x0)
        )

        def _score_row(row: np.ndarray) -> float:
            costs.parity_matrix = np.atleast_2d(np.asarray(row, dtype=int))
            return float(costs.cost_function(args.cost_function)(x0))

        def _score_row_at(
            row: np.ndarray,
            parameters: np.ndarray | None,
        ) -> float:
            if parameters is None:
                raise RuntimeError("iterative scoring requires orbital parameters")
            costs.parity_matrix = np.atleast_2d(np.asarray(row, dtype=int))
            return float(costs.cost_function(args.cost_function)(parameters))

        if args.select == "greedy":
            if args.n_singles is not None and args.n_quartets is not None:
                print(
                    f"[select greedy] quota senquart "
                    f"(n_singles={args.n_singles}, n_quartets={args.n_quartets}, "
                    f"cost={args.cost_function})"
                )
                selection = select_senquart_quota(
                    solver.n_sites,
                    _score_row,
                    int(args.n_singles),
                    int(args.n_quartets),
                )
            else:
                print(
                    f"[select greedy] scoring {args.candidates} pool "
                    f"(n_sym={args.n_sym}, cost={args.cost_function})"
                )
                selection = select_from_pool(
                    solver.n_sites,
                    args.n_sym,
                    _score_row,
                    candidates=args.candidates,
                )
            parity_matrix = selection.parity_matrix
            selected_costs = selection.selected_costs
            selection_meta = selection.metadata(
                candidates=args.candidates,
                cost_function=args.cost_function,
                parity_output="",
            )
        else:
            print(
                f"[select iterative] GF(2) pool extension "
                f"(n_sym={args.n_sym}, m_round={args.m_round}, "
                f"cost={args.cost_function})"
            )
            round_results = []

            def _optimize_pool(
                accumulated_parity: np.ndarray,
                parameters: np.ndarray | None,
                round_index: int,
            ):
                if parameters is None:
                    raise RuntimeError("iterative optimization requires initial x")
                costs.parity_matrix = accumulated_parity
                objective = costs.cost_function(args.cost_function)
                before = float(objective(parameters))
                print(
                    f"[select iterative] round {round_index + 1}: "
                    f"optimizing {len(accumulated_parity)} generators "
                    f"from cost {before:.8g}"
                )
                started_round = time.time()
                opt_result = scipy.optimize.minimize(
                    objective,
                    parameters,
                    method="L-BFGS-B",
                    options={"maxiter": args.maxiter},
                    callback=callback if args.verbose else None,
                )
                round_results.append(opt_result)
                elapsed_round = time.time() - started_round
                print(
                    f"[select iterative] round {round_index + 1}: "
                    f"optimized cost {float(opt_result.fun):.8g}"
                )
                return np.asarray(opt_result.x, dtype=float), {
                    "cost_before": before,
                    "cost_after": float(opt_result.fun),
                    "parameters_before": np.asarray(
                        parameters, dtype=float
                    ).tolist(),
                    "parameters_after": np.asarray(
                        opt_result.x, dtype=float
                    ).tolist(),
                    "converged": bool(opt_result.success),
                    "nit": int(getattr(opt_result, "nit", 0)),
                    "nfev": int(getattr(opt_result, "nfev", 0)),
                    "elapsed": float(elapsed_round),
                    "message": str(opt_result.message),
                }

            selection = select_iterative_pool(
                solver.n_sites,
                args.n_sym,
                _score_row,
                m_round=args.m_round,
                score_row_at=_score_row_at,
                optimize_pool=_optimize_pool,
                initial_parameters=x0,
            )
            iterative_res = round_results[-1]
            x0 = np.asarray(selection.optimized_parameters, dtype=float)
            parity_matrix = selection.parity_matrix
            selected_costs = selection.selected_costs
            selection_meta = selection.metadata(
                cost_function=args.cost_function,
                parity_output="",
                m_round=args.m_round,
            )

        parity_out = args.parity_output or default_parity_output_path(
            args.outname, select=args.select
        )
        parity_out = write_parity_matrix(parity_out, parity_matrix)
        selection_meta["parity_output"] = parity_out
        print(f"[select {args.select}] wrote parity matrix to {parity_out}")
        print(
            f"[select {args.select}] selected_costs="
            f"{np.round(selected_costs, 6).tolist()}"
        )
        costs.parity_matrix = parity_matrix
    else:
        parity_matrix = np.atleast_2d(np.loadtxt(args.parity, dtype=int))
        costs, result, solver = build_dmrg_orbital_costs(
            args.molpath,
            parity_matrix,
            store_dir=store_dir,
            config=DMRGConfig(max_bond_dim=args.bond_dim, n_sweeps=args.n_sweeps),
            multiply=MultiplyConfig(
                bond_dim=args.multiply_bond_dim,
                n_sweeps=args.multiply_sweeps,
            ),
            reuse=not args.no_reuse,
            n_threads=args.n_threads,
        )
        rotation_pairs, rotation_irreps = resolve_orbital_rotation(
            args.orbital_rotation, args.molpath, solver.n_sites
        )
        costs.pairs = rotation_pairs
        x0 = (
            np.zeros(n_params(solver.n_sites, rotation_pairs))
            if args.x0 is None
            else np.loadtxt(args.x0)
        )

    n_free = n_params(solver.n_sites, None)
    n_sym_params = n_params(solver.n_sites, rotation_pairs)
    print(
        f"orbital_rotation={args.orbital_rotation}: "
        f"N_free={n_free}, N_sym={n_sym_params}"
        + (f", reduced={n_free - n_sym_params}" if rotation_pairs is not None else "")
    )
    print("DMRG reference energy: {0:4.6f}".format(result.energy))
    print("wavefunction store: {}".format(result.store_dir))

    f = costs.cost_function(args.cost_function)
    if iterative_res is None:
        print("before optimization: {0:4.6f}".format(f(x0)))
        res = scipy.optimize.minimize(
            f,
            x0,
            method="L-BFGS-B",
            options={"maxiter": args.maxiter},
            callback=callback if args.verbose else None,
        )
        print(res.message)
        print("optimized: {0:4.6f}".format(res.fun))
    else:
        res = iterative_res
        print("iterative final:", res.message)
        print("optimized: {0:4.6f}".format(res.fun))
    print("cost evaluations: {}".format(costs.n_evaluations))

    outname = args.outname or (
        time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_x_opt.txt"
    )
    meta = dict(vars(args))
    meta["orbital_rotation"] = args.orbital_rotation
    if rotation_irreps is not None:
        meta["irreps"] = np.asarray(rotation_irreps, dtype=int).tolist()
    if selection_meta is not None:
        meta.update(selection_meta)
    with open(outname, "a", newline="", encoding="utf-8") as fp:
        fp.write(json.dumps(meta) + "\n")
        np.savetxt(fp, res.x)
    print("wrote", outname)

    if args.results_csv:
        append_result_row(
            args.results_csv,
            _results_row(
                args,
                status="ok",
                final_cost=float(res.fun),
                selected_costs=selected_costs,
                parity_output=parity_out,
                outname=outname,
                elapsed_s=time.time() - started,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize orbital rotations with MPS-native NC/variance costs"
    )
    parser.add_argument("molpath", help="Hamiltonian (.FCIDUMP or .chk)")
    parser.add_argument(
        "parity",
        nargs="?",
        default=None,
        help="parity-matrix path (omit when --select greedy|iterative)",
    )
    parser.add_argument("--cost_function", choices=("NC", "variance"), default="NC")
    parser.add_argument("--x0", default=None, help="initial rotation parameters")
    parser.add_argument("--bond_dim", type=int, default=250)
    parser.add_argument("--n_sweeps", type=int, default=20)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--wavefunction_dir", default=None)
    parser.add_argument("--multiply_bond_dim", type=int, default=None)
    parser.add_argument("--multiply_sweeps", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--no_reuse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outname", default=None)
    add_orbital_rotation_arg(parser)
    add_greedy_select_args(parser)
    args = parser.parse_args()

    try:
        validate_greedy_cli_args(
            select=args.select,
            n_sym=args.n_sym,
            cost_function=args.cost_function,
            parity=args.parity,
            seniority=False,
            symmetry_manifest=None,
            m_round=args.m_round,
            n_singles=args.n_singles,
            n_quartets=args.n_quartets,
            candidates=args.candidates,
        )
    except ValueError as exc:
        parser.error(str(exc))

    args.n_sym = resolve_select_n_sym(
        select=args.select,
        n_sym=args.n_sym,
        n_singles=args.n_singles,
        n_quartets=args.n_quartets,
    )

    started = time.time()
    try:
        _run_optimize(args)
    except Exception as exc:
        if args.results_csv:
            append_result_row(
                args.results_csv,
                _results_row(
                    args,
                    status="failed",
                    final_cost="",
                    selected_costs=(),
                    parity_output="",
                    outname=args.outname or "",
                    elapsed_s=time.time() - started,
                    message=str(exc),
                ),
            )
        raise


if __name__ == "__main__":
    main()
