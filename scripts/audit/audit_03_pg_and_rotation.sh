#!/usr/bin/env bash
# AUDIT 3 (v2) — SUBMIT THIS, do not run it on a login node (v1 hit the CPU cap).
#
#SBATCH --job-name=qs_audit3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=qs_audit3_%j.out
#SBATCH --error=qs_audit3_%j.err
#
# v2 changes: vectorized sector weights, streaming-JSON loader, and the test
# that actually matters -- the FCI weight held by the sector of the HARDCODED
# operators (v1 only tested the orbsym-derived ones, which trivially give 1.0).
#
#   sbatch scripts/audit/audit_03_pg_and_rotation.sh
#   MOL=n2 sbatch scripts/audit/audit_03_pg_and_rotation.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"
export MOL="${MOL:-both}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 - <<'PY'
import os, sys, glob, math
import numpy as np
from pyscf import gto, scf, fci, symm
from pyscf.fci import cistring

sys.path.insert(0, "scripts/audit")
from _qs_json import load_oo                              # noqa: E402
from src.sto3g_exact_symmetries import STO3G_EXACT_SPATIAL, STO3G_NORB

MOL = os.environ.get("MOL", "both")
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
BOND_TAG = {"h2o": lambda r: f"{r:.4f}".replace(".", "p"),
            "n2":  lambda r: f"{r:.4f}".replace(".", "p")}


def build(mol_name, r):
    if mol_name == "h2o":
        a = math.radians(104.5 / 2.0)
        atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (r * math.sin(a), 0.0, r * math.cos(a))],
                ["H", (-r * math.sin(a), 0.0, r * math.cos(a))]]
        pg = "C2v"
    else:
        atom = [["N", (0.0, 0.0, 0.0)], ["N", (0.0, 0.0, r)]]
        pg = "D2h"
    return gto.M(atom=atom, basis="sto-3g", symmetry=pg, verbose=0)


def sector_weights(civec, norb, nelec, supports):
    """Vectorized: weight per sign sector, key = bit j is (Q_j == -1)."""
    names = list(supports)
    masks = [supports[n] for n in names]
    sa = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    sb = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)

    def keys(strings):
        k = np.zeros(len(strings), dtype=np.int64)
        for j, m in enumerate(masks):
            par = np.zeros(len(strings), dtype=np.int64)
            v = strings & m
            while np.any(v):
                par ^= (v & 1)
                v >>= 1
            k |= par << j
        return k

    ka, kb = keys(sa), keys(sb)
    key = ka[:, None] ^ kb[None, :]
    p = np.abs(np.asarray(civec)) ** 2
    tot = np.bincount(key.ravel(), weights=p.ravel(),
                      minlength=1 << len(names))
    return names, tot


def label(names, idx):
    return ",".join(f"{n}={'+' if not (idx >> j) & 1 else '-'}"
                    for j, n in enumerate(names))


def run(mol_name):
    norb = STO3G_NORB[mol_name]
    hard = {lab: sorted(s) for lab, s in STO3G_EXACT_SPATIAL[mol_name]}
    print(f"\n{'#' * 78}\n### {mol_name.upper()}   hardcoded supports: {hard}\n{'#' * 78}")
    for r in GEOM[mol_name]:
        m = build(mol_name, r)
        mf = scf.RHF(m).run()
        orbsym = [str(s) for s in
                  symm.label_orb_symm(m, m.irrep_name, m.symm_orb, mf.mo_coeff)]
        derived = {g: sorted(i for i, s in enumerate(orbsym) if s in irr)
                   for g, irr in EXPECT[mol_name].items()}
        ok = derived == hard
        print(f"\n R={r:.4f}  orbsym={orbsym}")
        print(f"   orbsym-derived supports : {derived}")
        print(f"   hardcoded supports valid: "
              f"{'YES' if ok else '*** NO -- MO ORDER CHANGED ***'}")

        nelec = (m.nelectron // 2, m.nelectron // 2)
        e, civec = fci.FCI(m, mf.mo_coeff).kernel()
        print(f"   E_FCI = {e:.8f}")

        # THE decisive test: are the HARDCODED operators symmetries here?
        sup_hard = {g: sum(1 << p for p in hard[g]) for g in hard}
        names, w = sector_weights(civec, norb, nelec, sup_hard)
        w0 = float(w[0])
        print(f"   FCI weight in all-(+1) sector of the HARDCODED operators: "
              f"{w0:.12f}")
        if w0 < 1.0 - 1e-8:
            print(f"   *** NOT A SYMMETRY at this geometry: "
                  f"{1.0 - w0:.6f} of the ground state lies outside the "
                  f"pinned sector -> K cannot converge ***")
            for idx in np.argsort(-w)[:4]:
                if w[idx] > 1e-10:
                    print(f"        {label(names, int(idx)):40s} {w[idx]:.6f}")

        sup_der = {g: sum(1 << p for p in derived[g]) for g in derived}
        _, wd = sector_weights(civec, norb, nelec, sup_der)
        print(f"   (control) same test with orbsym-derived supports: "
              f"{float(wd[0]):.12f}")

        # Does the run's own orbital rotation break things further?
        tag = BOND_TAG[mol_name](r)
        pats = sorted(glob.glob(
            f"results/{mol_name}_endpoint_grid/bond_{tag}/U_*/*/NC/oo.json"))
        for oo in pats[:2]:
            try:
                data = load_oo(oo)
                from src.orbital_rotation import rotation_from_oo_data
                U = np.asarray(rotation_from_oo_data(data, norb), dtype=float)
                civ = fci.addons.transform_ci_for_orbital_rotation(
                    civec, norb, nelec, U)
            except Exception as exc:                       # noqa: BLE001
                print(f"      [skip rotation] {oo}: {exc}")
                continue
            _, wr = sector_weights(civ, norb, nelec, sup_hard)
            keep = float(wr[0])
            arm = "irrep" if "U_irrep" in oo else "full"
            print(f"      U={arm:5s} {oo}")
            print(f"        weight retained in pinned sector after U: "
                  f"{keep:.6f}   leaked: {1.0 - keep:.6f}")


for name in (["h2o", "n2"] if MOL == "both" else [MOL]):
    run(name)
PY
