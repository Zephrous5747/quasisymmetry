#!/usr/bin/env bash
# AUDIT 5 (v2) — the money table.
#
# v1 BUG: it joined against tables/{mol}/endpoint_grid.csv, which is append-only
# and interleaves the OLD n_exact=1 campaign with the document-exact one. Last
# row wins != newest era, so it reported the old K (1, 4) and old dim (27, 484).
# v2 reads results/*/bond_*/U_full/*/*/metrics.json directly -- one file per
# point, no era ambiguity.
#
# Columns that matter:
#   supports OK          are the hardcoded orbital indices right at this geometry
#   K / dim              chemical-accuracy K and the retained dimension
#   conv                 did K ever reach chemical accuracy
#   ref_w                reference_weight_sum: fraction of the FCI reference the
#                        retained sector eigenstates actually capture. < 1 means
#                        the taper threw part of the ground state away.
#
# Cheap (RHF only). Run:  bash scripts/audit/audit_05_support_vs_K.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
mkdir -p tables/audit
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"

python3 - <<'PY'
import glob, json, math, re
from pyscf import gto, scf, symm
from src.sto3g_exact_symmetries import STO3G_EXACT_SPATIAL

GEOM = {
  "h2o": [0.958, 1.1293333333, 1.3006666667, 1.472, 1.6433333333,
          1.8146666667, 1.986, 2.1573333333, 2.3286666667, 2.5],
  "n2":  [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
}
EXPECT = {
  "h2o": {"Q_B1": ("B1",), "Q_B2": ("B2",)},
  "n2":  {"Q_pix": ("B3u", "B2g"), "Q_piy": ("B2u", "B3g"),
          "Q_u":   ("B1u", "B2u", "B3u", "Au")},
}


def build(mol, r):
    if mol == "h2o":
        a = math.radians(104.5 / 2.0)
        atom = [["O", (0, 0, 0)],
                ["H", (r * math.sin(a), 0, r * math.cos(a))],
                ["H", (-r * math.sin(a), 0, r * math.cos(a))]]
        return gto.M(atom=atom, basis="sto-3g", symmetry="C2v", verbose=0)
    return gto.M(atom=[["N", (0, 0, 0)], ["N", (0, 0, r)]],
                 basis="sto-3g", symmetry="D2h", verbose=0)


def metrics_for(mol, r, u="full", method="iterative", cost="NC"):
    tag = f"{r:.4f}".replace(".", "p")
    pats = [f"results/{mol}_endpoint_grid/bond_{tag}/U_{u}/{method}/{cost}/metrics.json"]
    # tolerate a slightly different bond rounding on disk
    pats.append(f"results/{mol}_endpoint_grid/bond_{tag[:5]}*/U_{u}/{method}/{cost}/metrics.json")
    for p in pats:
        for f in sorted(glob.glob(p)):
            try:
                return json.load(open(f)), f
            except Exception:                              # noqa: BLE001
                continue
    return None, None


for mol in ("h2o", "n2"):
    hard = {lab: sorted(s) for lab, s in STO3G_EXACT_SPATIAL[mol]}
    print(f"\n{'=' * 108}")
    print(f"{mol.upper()}   hardcoded supports {hard}   "
          f"(source: results/{mol}_endpoint_grid/bond_*/U_full/iterative/NC/metrics.json)")
    print(f"{'=' * 108}")
    print(f"{'R':>9} {'supp OK':>8} {'derived supports':>42} "
          f"{'K':>6} {'dim':>6} {'sect':>5} {'conv':>6} {'ref_w':>7} {'n_ex':>5}")
    for r in GEOM[mol]:
        mf = scf.RHF(build(mol, r)).run()
        m = mf.mol
        orbsym = [str(s) for s in
                  symm.label_orb_symm(m, m.irrep_name, m.symm_orb, mf.mo_coeff)]
        derived = {g: sorted(i for i, s in enumerate(orbsym) if s in irr)
                   for g, irr in EXPECT[mol].items()}
        ok = "YES" if derived == hard else "NO <<"
        d, _f = metrics_for(mol, r)
        if d is None:
            K = dim = sect = conv = refw = nex = "-"
        else:
            K = d.get("K")
            dim = d.get("relevant_sectors_total_dim")
            sect = d.get("relevant_sectors_count")
            conv = "yes" if d.get("converged") else "NO"
            rw = d.get("reference_weight_sum")
            refw = f"{rw:.4f}" if isinstance(rw, (int, float)) else "-"
            nex = d.get("n_exact")
        dshort = "{" + ", ".join(f"{k}:{v}" for k, v in derived.items()) + "}"
        print(f"{r:9.4f} {ok:>8} {dshort:>42} {str(K):>6} {str(dim):>6} "
              f"{str(sect):>5} {str(conv):>6} {refw:>7} {str(nex):>5}")

print("""
How to read it
--------------
  ref_w  < 1.0   the retained (tapered) space does not contain the whole FCI
                 reference -- the missing weight is exactly what no K can
                 recover, so `conv` is NO and K runs up to `dim`.
  n_ex   < r     the off-by-one from audit 2 (document r = 4 for H2O, 5 for N2).
  dim    ~ 2x    the document exact-sector dim (133 / 1824) is the signature of
                 one unpinned point-group generator.
""")
PY
