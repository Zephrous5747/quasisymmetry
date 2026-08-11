# Commit note — pin this to Cursor

Uncommitted work spanning ~2 weeks. `metrics.py` has been destroyed twice
because of this; `src/exact_parity.py` is untracked and has **no** compiled
fallback, so a third incident is unrecoverable.

## Do this first

`.gitignore:14` is `scripts/` and `:12` is `cluster_tests/`. **66 source files
have never been tracked**, including the campaign driver
`scripts/run_endpoint_point.py`. A plain `git add` silently skips them.

Replace the blanket `scripts/` rule with:

```gitignore
# scripts/: track source, ignore generated artefacts
scripts/**/__pycache__/
scripts/**/*.pyc
scripts/**/*.out
scripts/**/*.err
scripts/**/*.log
scripts/**/*.json
scripts/**/*.csv
scripts/**/*.sample
```

Leave `cluster_tests/` ignored — it contains its own nested `.git`.

## Do NOT commit

- **594 files whose only diff is CRLF→LF.** `chemistry.py` and `README.md` diff
  to nothing under `--ignore-all-space`. Adding them buries the real work.
- `results/`, `tables/analysis/data/`, `archive/`, `wavefunctions/` — 3000+
  generated files.
- `archive/metrics.cpython-311.pyc` — keep on disk, do not track.

## Files with real content changes (13)

```
make_pyscf_hamiltonian.py   optimize_dmrg.py   optimize_symmetries.py
metrics.py
src/clifford_sectors.py     src/energy_diagnostics.py
src/greedy_selection.py     src/iterative_pool.py
src/orbital_rotation.py     src/workflow_cli.py
tests/test_coupled_energy.py  tests/test_greedy_selection.py
tests/test_iterative_pool.py
```

## New untracked source (9 + scripts/)

```
src/exact_parity.py         src/exact_taper.py       src/fci_rotation_checks.py
src/parity_rank.py          src/pyscf_chk.py         src/sto3g_exact_symmetries.py
tests/test_exact_parity_taper.py  tests/test_parity_rank.py
tests/test_sto3g_exact_symmetries.py
scripts/**/*.py  scripts/**/*.sh        (after the .gitignore change above)
tables/analysis/section_*.tex  tables/analysis/gen_*.tex
tables/analysis/report_las_sto3g.tex  COMMIT_NOTE.md
```

## Suggested message

```
Rebuild metrics.py exact-tapering path; add Q4 campaigns and report sections

metrics.py: restore the exact/LAS split integration lost when the file was
overwritten. load_clifford_symmetries returns the split with exact generators
ordered first; n_exact counts exact rows surviving at the QUBIT level, since
iota(all-ones) = P_alpha ^ P_beta makes an all-ones row dependent there.
Exact sector defaults to the reference determinant's sector rather than the
densest sector, which is what made K saturate at 261/3584.

Restore --overlap_reference {fci,dmrg} (default fci), absent from HEAD but
documented in workflow_cli and passed by five point scripts. Route to the
Clifford backend whenever the OO JSON carries exact_masks: the determinant
path emits no n_exact and cannot satisfy the audit preconditions.

Add load_oo_input (oo.json is a stream of appended records, optionally with a
trailing bare rotation vector), the reference-weight guard with
--strict_reference_weight, D_max/D_min over the exact-filtered partition, and
the FCI-rotation and effective-parity-rank checks.

Verified against archive/metrics.cpython-311.pyc by bytecode diff (0 unexpected
drift, module scope included) and by reproducing N2 R=1.8: K=18, n_exact=5,
exact_sector=11000, W=1.

Campaigns: two 126-point grids at M=n-r_sp and M=n-r_sp-1, both clean.
Report: Proposition (coset invariance) + matroid corollary; Q4 answered;
budget trade-off section; corrected SO(n) parameterisation (exponential, not
Givens product).

.gitignore: stop ignoring scripts/ wholesale.
```

## Sanity check before pushing

```bash
git diff --cached --stat | tail -5          # expect ~90 files, not ~700
git diff --cached --name-only | grep -c hamiltonians   # expect 0
git ls-files | grep -c "^scripts/"          # expect ~50+
```
