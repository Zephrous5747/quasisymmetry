#!/usr/bin/env bash
# Rerun the 7 H2O endpoint points that failed the reference-weight precondition.
#
# WHY THEY FAILED — not a solver problem.
#   All 7 are U=full. Sec.1.1 of the procedure note: point-group parities stay
#   diagonal Z products only if U is block diagonal in the irreps; under a more
#   general U "the simple Clifford tapering described here is unavailable".
#   Full SO(n) OO drifts out of irrep-block form, so Q_B1/Q_B2 stop commuting
#   with H(U) and 2.4%-28.5% of the reference leaves the pinned exact sector.
#   No tightening of MAXITER, tolerances or states_per_sector can recover a
#   reference that is not in the retained space.
#
# So this escalates rather than just retries:
#   TIER 1  full-U, tighter optimiser (more macro, more OO iters, more roots),
#           STRICT off -> records the leakage as data instead of crashing.
#           Expected to remain unconverged; that IS the reportable result.
#   TIER 2  irrep-U, the admissible packing. Expected clean.
# Both tiers are written to separate campaign tags so neither overwrites the
# other's metrics.json.
#
# Usage:
#   bash scripts/trillium_rerun_h2o_failed.sh              # both tiers
#   TIER=1 bash scripts/trillium_rerun_h2o_failed.sh
#   DRY_RUN=1 bash scripts/trillium_rerun_h2o_failed.sh
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
# shellcheck disable=SC1091
source "${REPO}/cluster_tests/_qs_env.sh"
mkdir -p tables/audit

DRY_RUN="${DRY_RUN:-0}"
TIER="${TIER:-both}"

# bond:method:cost — the 7 points that tripped the precondition
POINTS=(
  "1.1293333333:mixed_disjoint:variance"
  "1.3006666667:mixed_disjoint:variance"
  "1.3006666667:iterative:variance"
  "1.4720000000:mixed_disjoint:variance"
  "1.6433333333:mixed_disjoint:variance"
  "1.8146666667:mixed_disjoint:variance"
  "1.9860000000:mixed_disjoint:variance"
)

run_point() {
  local bond="$1" method="$2" cost="$3" packing="$4" tier="$5"
  local extra=()
  if [[ "$tier" == "1" ]]; then
    # Tighter optimiser + a full sector spectrum, and record rather than crash.
    extra+=(--maxiter 400 --max_macroiterations 60 --stable_span_iters 3
            --states_per_sector 1000000)
  else
    extra+=(--maxiter 200 --max_macroiterations 40 --stable_span_iters 3
            --states_per_sector 1000000 --strict_reference_weight)
  fi
  local cmd=(python -u scripts/run_endpoint_point.py
    --molecule h2o --bond "$bond" --method "$method" --cost_function "$cost"
    --basis sto-3g --point_group C2v --orbital_rotation "$packing"
    --m_round 1 --iterative_reference fci_rotate
    --campaign "h2o_fix_tier${tier}"
    --results_csv "tables/audit/h2o_fix_tier${tier}.csv"
    "${extra[@]}")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "${cmd[*]}"
    return 0
  fi
  echo "[tier $tier] h2o R=$bond $method/$cost U=$packing"
  # tier 1 is expected to fail the precondition; do not abort the loop
  "${cmd[@]}" || echo "[tier $tier] point did not converge (recorded)"
}

for spec in "${POINTS[@]}"; do
  IFS=':' read -r bond method cost <<< "$spec"
  [[ "$TIER" == "both" || "$TIER" == "1" ]] && run_point "$bond" "$method" "$cost" full  1
  [[ "$TIER" == "both" || "$TIER" == "2" ]] && run_point "$bond" "$method" "$cost" irrep 2
done

echo
echo "[done] compare the tiers:"
echo "  column -s, -t tables/audit/h2o_fix_tier1.csv | cut -c1-200"
echo "  column -s, -t tables/audit/h2o_fix_tier2.csv | cut -c1-200"
echo "  U=irrep bash scripts/audit/audit_06_verify_rerun.sh | tail -20"
echo
echo "Expected: tier 1 still shows reference_weight_sum < 1 and converged=false"
echo "(that is the full-U result, and it belongs in the note as Table 'leakage');"
echo "tier 2 shows reference_weight_sum = 1 and small K."
