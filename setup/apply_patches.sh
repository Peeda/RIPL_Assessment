#!/usr/bin/env bash
# ==============================================================================
# setup/apply_patches.sh - apply this repo's patches to the ManiSkill checkout.
#
#   source /workspace/ripl/env.sh
#   bash setup/apply_patches.sh          # apply (idempotent)
#   bash setup/apply_patches.sh status   # report only, change nothing
#   bash setup/apply_patches.sh revert   # back to stock upstream
#
# Why this file exists: setup/setup_runpod.sh clones ManiSkill behind a
# [ -d "$ROOT/ManiSkill" ] guard, so a rerun never updates the checkout, and a
# fresh pod re-clones it from scratch. Any change we need inside ManiSkill's
# tree therefore has to live here, in git, as a patch - editing the pod's copy
# by hand is one rebuild away from gone.
#
# Idempotent by construction: a patch that reverse-applies cleanly is already
# in, and is skipped. Safe to run after every setup/setup_runpod.sh.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)      # this script lives in setup/; patches/ is at the root
PATCHES=$ROOT/patches
MODE=${1:-apply}

shopt -s nullglob
files=("$PATCHES"/*.patch)
[ ${#files[@]} -gt 0 ] || { echo "no patches in $PATCHES"; exit 0; }

cd "$MANISKILL_REPO"
echo "ManiSkill: $MANISKILL_REPO @ $(git rev-parse --short HEAD)"

rc=0
for p in "${files[@]}"; do
  name=$(basename "$p")
  if git apply --reverse --check "$p" 2>/dev/null; then
    state=applied
  elif git apply --check "$p" 2>/dev/null; then
    state=pending
  else
    state=CONFLICT
  fi

  case "$MODE:$state" in
    status:*)         echo "  [$state] $name" ;;
    apply:applied)    echo "  [skip]  $name (already applied)" ;;
    apply:pending)    git apply "$p"; echo "  [apply] $name" ;;
    revert:applied)   git apply --reverse "$p"; echo "  [revert] $name" ;;
    revert:pending)   echo "  [skip]  $name (not applied)" ;;
    *:CONFLICT)
      echo "  [FAIL]  $name - applies neither forward nor backward."
      echo "          The checkout moved. Rebase the patch; do not hand-edit"
      echo "          $MANISKILL_REPO and do not skip this."
      rc=1 ;;
  esac
done

# The patched flag is load-bearing enough to confirm rather than assume.
if [ "$MODE" != status ]; then
  echo -n "  pool_feature_map default: "
  grep -m1 -o 'pool_feature_map: bool = [A-Za-z]*' \
    examples/baselines/diffusion_policy/train_rgbd.py 2>/dev/null \
    | sed 's/.*= //' || echo "(absent - stock upstream)"
fi

exit $rc
