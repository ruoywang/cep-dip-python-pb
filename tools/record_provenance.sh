#!/usr/bin/env bash
# Record code provenance for a compute run, and refuse dirty trees.
#
# Usage:  tools/record_provenance.sh <run-dir>
#
# Writes <run-dir>/run_provenance.txt (+ .diff if dirty). Exits non-zero if
# the working tree is dirty or HEAD is not pushed, so job scripts can gate on:
#
#   $REPO/tools/record_provenance.sh "$RUNDIR" || exit 1

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: record_provenance.sh <run-dir>}"
mkdir -p "$OUT"

cd "$REPO"
HASH=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain | wc -l)
if git branch -r --contains "$HASH" 2>/dev/null | grep -q .; then
  PUSHED=yes
else
  PUSHED=NO
fi

{
  echo "repo:    $REPO"
  echo "commit:  $HASH"
  echo "short:   $SHORT"
  echo "branch:  $BRANCH"
  echo "pushed:  $PUSHED"
  echo "dirty:   $DIRTY file(s)"
  echo "date:    $(date -Is)"
  echo "host:    $(hostname)"
} > "$OUT/run_provenance.txt"

STATUS=0
if [ "$DIRTY" -ne 0 ]; then
  git diff > "$OUT/run_provenance.diff"
  git status --porcelain >> "$OUT/run_provenance.txt"
  echo "record_provenance: REFUSING — $DIRTY uncommitted change(s); commit and push first." >&2
  STATUS=1
fi
if [ "$PUSHED" = "NO" ]; then
  echo "record_provenance: REFUSING — HEAD $SHORT is not on any remote branch; push first." >&2
  STATUS=1
fi
exit $STATUS
