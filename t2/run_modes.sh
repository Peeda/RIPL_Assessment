#!/usr/bin/env bash
# ==============================================================================
# t2/run_modes.sh - the three pre-registered confirmation passes.
#
#   source /workspace/ripl/env.sh
#   export CKPT=.../checkpoints/best_eval_success_once.pt
#   tmux new -s t2
#   bash t2/run_modes.sh 2>&1 | tee ~/t2-modes.log
#
# Stages:  size | gap | farb | nearbase | analyze | all   (default: all)
#
# The refined failure axes - face_gap and distance from the Panda base - were
# found by slicing the nominal pass, which is the same way the original
# separation>260mm mode was found and the same reason it needs its own data.
# Re-measuring a region on the rollouts that identified it measures noise. Each
# stage below draws FRESH seeds, from above the nominal range, and repeats each
# one so "this state is hard" separates from "the policy is noisy".
#
# The thresholds are PRE-REGISTERED - written down in notes/t2-failure-modes.md
# before any of these ran. Predictions from the nominal pass, to be checked
# against what comes back (unconditional is 0.713):
#
#   gap       0.640 [0.501, 0.759]   face_gap < 25 mm, reach controlled
#   farb      0.561 [0.410, 0.701]   cubeB far, cubeA comfortable, gap controlled
#   nearbase  0.637 [0.564, 0.704]   dist_min < 520 mm, gap and far-reach controlled
#
# gap and farb are the two T-IV TARGETS, chosen because they fail at different
# stages by different mechanisms: gap fouls cubeB on the way in and fails to get
# cubeA onto it, farb places cleanly and fails to make the stack STAY
# (hold|place 0.657 vs 0.820 reference). nearbase is a third T-II finding, not
# a T-IV target.
#
# farb requires cubeA < 720 mm, not just cubeB far. The both-cubes-far corner is
# where grasping actually breaks down kinematically (0.815 above 740 mm), and no
# bounded residual recovers a target the IK cannot reach - so it is excluded
# rather than left in to depress the result.
#
# The three filters are MUTUALLY EXCLUSIVE - gap and nearbase partition on
# dist_min, and both require dist_max < 760 which farb's dist_B >= 760 excludes.
# Each controls for the others' factor; without that they contaminate each other
# and none of the three numbers means anything.
#
# `size` is free and answers "will these fill?" before an hour is spent finding
# out that they do not. Run it first on any index you have not sized before.
# NOTE farb is the tight one: it hits ~3.4% of eligible seeds, so it needs about
# 12,000 indexed seeds to yield 300 from above the nominal block. Extend the
# index rather than relaxing the 760 mm threshold to fit the index you have.
#
# Each pass writes region_<tag>_seeds.csv (all 300 - this is the fixed
# TARGETED-EVALUATION SET that T-III biases toward and T-IV is scored on; keep
# it) plus the three blocks it was split into, and one rollout CSV per block.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${T2_OUT:-${RIPL_ROOT:-/workspace/ripl}/t2}
SELF=$HERE/$(basename "$0")

# Kept identical across the three so the only thing that differs between them
# is the region. REGION_MIN_SEED floors every pass above the nominal block.
#
# 3 blocks x 100 episodes = the assignment's "100 rollouts and 3 seeds", per
# failure mode. Each block is a DISJOINT set of region seeds run under its own
# policy seed, so the three rates are independent estimates and their spread is
# a real error bar - see do_t1 in run_t2.sh for why reusing one block and
# varying only the policy seed is the wrong reading.
export REGION_BLOCKS=${REGION_BLOCKS:-3}
export REGION_EPISODES=${REGION_EPISODES:-100}
export REGION_SEEDS=${REGION_SEEDS:-$(( REGION_BLOCKS * REGION_EPISODES ))}
export REGION_REPEATS=${REGION_REPEATS:-1}
export REGION_MIN_SEED=${REGION_MIN_SEED:-2200}

GAP_WHERE="face_gap < 0.025 and 0.52 <= dist_min and dist_max < 0.76"
NEAR_WHERE="dist_min < 0.52 and dist_max < 0.76 and face_gap >= 0.05"
FARB_WHERE="dist_B >= 0.76 and dist_A < 0.72 and face_gap >= 0.05"

t0=$(date +%s)
stage() { echo ""; echo "=== $* [$(( $(date +%s) - t0 ))s] ==="; }

pass() {   # tag, where
  local tag=$1 where=$2
  # Never overwrite a finished pass. Rollouts are stochastic, so anything
  # clobbered here is gone - the same rule that cost us the first t1 block.
  if [ -f "$OUT/region_${tag}_seed1.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  skip '$tag' - $OUT/region_${tag}_seed1.csv exists (FORCE=1 to re-run)"
    return
  fi
  stage "pass '$tag': $where"
  REGION_TAG="$tag" REGION_WHERE="$where" bash "$HERE/run_t2.sh" region
}

do_size() {
  stage "sizing all three regions against the index (no rollouts)"
  for spec in "gap|$GAP_WHERE" "farb|$FARB_WHERE" "nearbase|$NEAR_WHERE"; do
    echo ""
    echo "  --- ${spec%%|*}"
    python "$HERE/select_seeds.py" "$OUT/seeds.csv" \
      --min-seed "$REGION_MIN_SEED" -n "$REGION_SEEDS" \
      --where "${spec#*|}" --dry-run
  done
}

case "${1:-all}" in
  size)     do_size ;;
  gap)      pass gap "$GAP_WHERE" ;;
  nearbase) pass nearbase "$NEAR_WHERE" ;;
  farb)     pass farb "$FARB_WHERE" ;;
  analyze)  bash "$HERE/run_t2.sh" analyze ;;
  all)      do_size
            pass gap "$GAP_WHERE"
            pass farb "$FARB_WHERE"
            pass nearbase "$NEAR_WHERE"
            bash "$HERE/run_t2.sh" analyze ;;
  *) sed -n '2,44p' "$SELF"; exit 1 ;;
esac

stage "done"
echo ""
echo "  Figures land in the repo's figures/ but the pod never pushes -"
echo "  rsync them back:  bash setup/transfer.sh info"
