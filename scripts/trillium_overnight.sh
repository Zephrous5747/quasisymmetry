#!/usr/bin/env bash
# Overnight batch. Submits everything outstanding, in dependency order, and
# leaves a single log to read in the morning.
#
#   DRY_RUN=1 bash scripts/trillium_overnight.sh    # show the plan only
#   bash scripts/trillium_overnight.sh              # submit
#
# Jobs, in order:
#   A  verification checks   (~20 min, 1 node)  -- the two flagged in the report
#                                                  plus the W regression
#   B  multi-start N2        (~6 h, 33 tasks)   -- the scatter estimate that
#                                                  Q4(ii) currently lacks
#   C  audits                (after B)          -- summarise both
#
# B does NOT depend on A; they run concurrently. C waits on B.

set -euo pipefail
REPO="${REPO:-$PWD}"
cd "$REPO"

echo "=== overnight plan ==="
echo "  A  scripts/verify/run_outstanding_checks.py   (commutator, rotation, W)"
echo "  B  scripts/trillium_multistart_n2.sh          (33 tasks x 5 starts)"
echo "  C  audit_08 / audit_09 / audit_10             (after B)"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] nothing submitted."
  echo "  check inputs first:"
  echo "    DRY_RUN=1 bash scripts/trillium_multistart_n2.sh"
  exit 0
fi

mkdir -p logs

# ---- A: verification checks -------------------------------------------------
JA=$(sbatch --parsable \
     --job-name=qs_verify --nodes=1 --ntasks=1 --cpus-per-task=8 \
     --time=01:00:00 --output=logs/verify_%j.out --error=logs/verify_%j.err \
     --wrap "source ${REPO}/cluster_tests/_qs_env.sh && \
             python3 -u scripts/verify/run_outstanding_checks.py")
echo "  submitted A (verification)  jobid=$JA"

# ---- B: multi-start ---------------------------------------------------------
mapfile -t POINTS < <(ls -d results/n2_m35_grid/bond_*/U_irrep/*/NC 2>/dev/null | sort)
NB=${#POINTS[@]}
if (( NB == 0 )); then
  echo "  [warn] no m35_grid N2 NC points found -- skipping B and C" >&2
else
  JB=$(sbatch --parsable --export=ALL --array=0-$((NB-1)) \
       scripts/trillium_multistart_n2.sh)
  echo "  submitted B (multi-start, $NB tasks)  jobid=$JB"

  # ---- C: audits, after B completes -----------------------------------------
  JC=$(sbatch --parsable --dependency=afterany:"$JB" \
       --job-name=qs_audit --nodes=1 --ntasks=1 --cpus-per-task=4 \
       --time=00:30:00 --output=logs/audit_%j.out --error=logs/audit_%j.err \
       --wrap "source ${REPO}/cluster_tests/_qs_env.sh && \
               cd ${REPO} && \
               echo '--- audit_08 (M=n-rsp) ---' && \
               BUDGET_H2O=4 BUDGET_N2=6 RESULTS_SUFFIX=q4_grid \
                 bash scripts/audit/audit_08_q4_compare.sh && \
               echo '--- audit_08 (M=n-rsp-1) ---' && \
               BUDGET_H2O=3 BUDGET_N2=5 RESULTS_SUFFIX=m35_grid \
                 bash scripts/audit/audit_08_q4_compare.sh && \
               echo '--- audit_09 arm equivalence ---' && \
               RESULTS_SUFFIX=m35_grid bash scripts/audit/audit_09_arm_equivalence.sh && \
               echo '--- multi-start spread ---' && \
               python3 scripts/analysis/summarise_multistart.py")
  echo "  submitted C (audits, after B)  jobid=$JC"
fi

echo
echo "In the morning:"
echo "  cat logs/verify_*.out          # A: expect all PASS"
echo "  cat logs/audit_*.out           # C: dK vs per-start scatter"
echo "  squeue -u \$USER                # anything still running"
