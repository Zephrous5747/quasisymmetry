#!/usr/bin/env bash
# Archive or delete superseded SLURM .out/.err files in the repo root.
#
# Keeps the newest array per job family by default (highest ArrayJobId), because
# only the newest run corresponds to the current code. Older families are the
# pre-fix campaigns and are only worth keeping as provenance for the retraction.
#
#   bash scripts/clean_job_logs.sh              # dry run, shows what would go
#   bash scripts/clean_job_logs.sh --archive    # tar them into archive/, then delete
#   bash scripts/clean_job_logs.sh --delete     # delete outright
#   KEEP=2 bash scripts/clean_job_logs.sh --archive   # keep the newest 2 arrays
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

MODE="${1:-dry}"
KEEP="${KEEP:-1}"

# family = job name + array id, e.g. qs_exh_2062517
mapfile -t FAMILIES < <(
  ls -1 *.out *.err 2>/dev/null \
    | sed -E 's/_[0-9]+\.(out|err)$//' \
    | sed -E 's/\.(out|err)$//' \
    | sort -u
)

# group by job name (strip the trailing _<arrayid>)
declare -A NEWEST
for fam in "${FAMILIES[@]}"; do
  name="${fam%_*}"
  id="${fam##*_}"
  [[ "$id" =~ ^[0-9]+$ ]] || continue
  cur="${NEWEST[$name]:-0}"
  (( id > cur )) && NEWEST[$name]="$id"
done

KEEP_LIST=()
for name in "${!NEWEST[@]}"; do
  mapfile -t ids < <(printf '%s\n' "${FAMILIES[@]}" \
    | grep -E "^${name}_[0-9]+$" | sed -E "s/^${name}_//" | sort -rn | head -"$KEEP")
  for i in "${ids[@]}"; do KEEP_LIST+=("${name}_${i}"); done
done

echo "keeping (newest $KEEP per job name):"
printf '  %s\n' "${KEEP_LIST[@]}" | sort

DROP=()
for fam in "${FAMILIES[@]}"; do
  skip=0
  for k in "${KEEP_LIST[@]}"; do [[ "$fam" == "$k" ]] && skip=1 && break; done
  (( skip )) || DROP+=("$fam")
done

if (( ${#DROP[@]} == 0 )); then
  echo "nothing to remove"
  exit 0
fi

echo
# NOTE: with `set -euo pipefail`, a pipeline whose first command exits non-zero
# (du on a glob that matches nothing) kills the script. Collect the file list
# first and guard every substitution.
# Test each candidate with -f instead of relying on nullglob: an unmatched
# glob otherwise survives as a literal name and tar dies on "Cannot stat".
files_of() {
  local fam="$1" f
  for f in "${fam}"_*.out "${fam}"_*.err "${fam}".out "${fam}".err; do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
  return 0
}

echo "superseded families:"
TOTAL=0
for fam in "${DROP[@]}"; do
  mapfile -t fl < <(files_of "$fam")
  n=${#fl[@]}
  (( n == 0 )) && continue
  sz=$( { du -ch "${fl[@]}" 2>/dev/null || true; } | tail -1 | cut -f1 )
  TOTAL=$(( TOTAL + n ))
  echo "  $fam  ($n files, ${sz:-?})"
done
echo "  ---- $TOTAL files across ${#DROP[@]} famil(ies)"

case "$MODE" in
  dry)
    echo
    echo "dry run. Re-run with --archive (tar to archive/, then delete) or --delete."
    ;;
  --archive)
    mkdir -p archive
    STAMP="$(date +%Y%m%d_%H%M%S)"
    TAR="archive/job_logs_${STAMP}.tgz"
    ALL=()
    for fam in "${DROP[@]}"; do
      mapfile -t fl < <(files_of "$fam")
      (( ${#fl[@]} )) && ALL+=("${fl[@]}")
    done
    if (( ${#ALL[@]} == 0 )); then
      echo "nothing to archive"; exit 0
    fi
    tar czf "$TAR" "${ALL[@]}"
    echo "[archive] $TAR ($(du -h "$TAR" | cut -f1))"
    for fam in "${DROP[@]}"; do
      mapfile -t fl < <(files_of "$fam")
      (( ${#fl[@]} )) && rm -f "${fl[@]}"
    done
    echo "[clean] removed ${#DROP[@]} superseded famil(ies)"
    ;;
  --delete)
    for fam in "${DROP[@]}"; do
      mapfile -t fl < <(files_of "$fam")
      (( ${#fl[@]} )) && rm -f "${fl[@]}"
    done
    echo "[clean] deleted ${#DROP[@]} superseded famil(ies)"
    ;;
  *)
    echo "unknown mode: $MODE (use --archive or --delete)" >&2
    exit 1
    ;;
esac
