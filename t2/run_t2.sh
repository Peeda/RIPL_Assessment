#!/usr/bin/env bash
# ==============================================================================
# t2/run_t2.sh - T-II failure-mode mining, end to end.
#
#   source /workspace/ripl/env.sh
#   tmux new -s t2
#   bash t2/run_t2.sh index  2>&1 | tee ~/t2-index.log
#   bash t2/run_t2.sh all    2>&1 | tee ~/t2.log
#
# Stages:  index | t1 | mine | region | backend | videos | analyze | plan | all
#          (default: all; backend is not in 'all' - it needs a GPU)
#
#   index    seed -> initial state table. No policy, no GPU, minutes.
#   t1       100 rollouts x 3 seeds - the assignment's T-I deliverable.
#   mine     the nominal pass. The discovery half of T-II.
#   region   fresh seeds from one failure region, repeated. The measurement half.
#   backend  does physx_cuda reproduce physx_cpu? Licenses T-IV's train-on-GPU,
#            evaluate-on-CPU split. Needs a GPU; run it separately.
#   videos   mp4s for the shortlist analyze prints.
#   analyze  curves, taxonomy, Wilson intervals, the video shortlist.
#
# The rollout passes use DISJOINT seed blocks:
#
#   t1       [6000, 6300)           3 blocks of 100, policy seeds 1/2/3
#   nominal  [1000, 1000+NOMINAL)   discovery
#   region   >= 1000+NOMINAL        filtered by REGION_WHERE
#
# Disjointness is not tidiness. Re-measuring on the rollouts that identified
# the region measures noise rather than a failure mode, and a T-I number that
# shares episodes with the T-II substrate is not an independent check on it.
#
# Every REGION_WHERE threshold is PRE-REGISTERED - written down in the plan and
# in notes/t2-failure-modes.md before the pass ran. Say so in the report: a
# threshold fixed in advance is a much stronger claim than one chosen after
# seeing the scatter. The original 80 mm one came from CLAUDE.md's reading of
# UniformPlacementSampler; the refined face_gap and reach thresholds come from
# the nominal pass, which is why they need their own fresh-seed passes and
# cannot be quoted off the episodes that suggested them.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"
: "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"

ENV_ID=${ENV_ID:-StackCube-v1}
CTRL=${CTRL:-pd_ee_delta_pos}
BACKEND=${BACKEND:-physx_cpu}
export ENV_ID CTRL BACKEND

OUT=${T2_OUT:-${RIPL_ROOT:-/workspace/ripl}/t2}
# HERE is t2/, which is also where every python file this driver calls lives.
# ROOT is the repo, which is where figures/ belongs - not under t2/.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# Absolute, because the usage printer at the bottom reads this file back
# AFTER the cd into $OUT, and a relative $0 does not survive that.
SELF=$HERE/$(basename "$0")

# physx_cpu vectorises by SUBPROCESS, so this is a process count, not a batch
# width. It buys nothing above the core count.
NUM_ENVS=${NUM_ENVS:-10}
# 12000, not 8000: the farb region (cubeB far, cubeA comfortable, gap
# controlled) hits ~3.4% of eligible seeds, so 200 of them need ~9,800 seeds
# above the nominal block. Extending the index is policy-free and GPU-free -
# cheaper than relaxing a pre-registered threshold to fit the index on hand.
INDEX_SEEDS=${INDEX_SEEDS:-12000}

# ===========================================================================
# SEED BLOCKS. Two separate disjointness requirements, and the second one was
# missed on the first production run - read both before changing a base.
#
# 1. The passes must be disjoint FROM EACH OTHER. A T-I number sharing episodes
#    with the T-II substrate is not an independent check on it, and a targeted
#    pass re-measuring states the nominal pass reported on measures noise.
#
# 2. Every pass must be disjoint FROM THE DEMONSTRATIONS. Motionplanning demos
#    are generated from CONSECUTIVE EPISODE SEEDS STARTING AT 0
#    (examples/motionplanning/panda/run.py:44-101 - `seed = start_seed` with
#    start_seed=0, then `seed += 1` per attempt). ~990 trajectories were
#    replayed and 800 trained on, so roughly seeds [0, 1000) are TRAINING
#    initial states.
#
#    T1_BASE used to default to 0, which measured T-I on the training set:
#    0.910 on seeds 0-299 against 0.713 on held-out seeds 1000-2199, same
#    checkpoint, a gap uniform across every separation bin. That is a
#    memorisation measurement, not a success rate. DO NOT put any evaluation
#    block below seed 1000.
#
#   t1       [6000, 6300)                3 x 100, the assignment's deliverable
#   nominal  [1000, 1000 + NOMINAL)      the T-II discovery pass
#   region   >= 1000 + NOMINAL, filtered the T-II measurement passes
# ===========================================================================
DEMO_SEED_CEILING=${DEMO_SEED_CEILING:-1000}
T1_BASE=${T1_BASE:-6000}
T1_EPISODES=${T1_EPISODES:-100}    # per seed; the assignment asks for 100 x 3
NOMINAL_BASE=${NOMINAL_BASE:-1000}
NOMINAL=${NOMINAL:-1200}

# The targeted pass selects on an arbitrary condition over the initial state
# (select_seeds.py --where), because the refined failure axes are not centre
# separation. REGION_TAG keeps the outputs from colliding.
#
#   gap       face_gap < 25 mm, reach controlled   the descent fouls cubeB
#   nearbase  dist_min < 520 mm, gap controlled    place phase, grasp is fine
#   reach     dist_max >= 760 mm, gap controlled   at the kinematic boundary
#   near/far  the original separation bands        superseded, still runnable
#
# The three refined filters are MUTUALLY EXCLUSIVE (they partition on dist_min
# and dist_max), so no seed can appear in two passes, and each controls for the
# other modes' factor - without that they contaminate each other and no number
# means anything.
# The assignment asks for per-failure-mode success over 100 rollouts x 3 seeds.
# That is 3 DISJOINT blocks of REGION_EPISODES seeds, each with its own policy
# seed - the same reading do_t1 uses, and for the same reason: reusing one block
# and varying only the policy seed holds the initial states fixed and measures
# DDPM sampling noise, which is a much smaller quantity than a real error bar.
#
# The full REGION_SEEDS list is ALSO the targeted-evaluation set that T-III
# biases toward and T-IV is scored on, so it is written out whole as well as in
# blocks. Keep it fixed once measured.
REGION_BLOCKS=${REGION_BLOCKS:-3}
REGION_EPISODES=${REGION_EPISODES:-100}
REGION_SEEDS=${REGION_SEEDS:-$(( REGION_BLOCKS * REGION_EPISODES ))}
# Repeats are a different instrument - the same state run twice, separating
# "this state is hard" from "the policy is noisy". Already measured on the
# near/far passes (0.74 / 0.67 agreement); 1 here so the 3-seed structure is
# what carries the error bar.
REGION_REPEATS=${REGION_REPEATS:-1}
REGION_TAG=${REGION_TAG:-gap}
REGION_WHERE=${REGION_WHERE:-"face_gap < 0.025 and 0.52 <= dist_min and dist_max < 0.76"}
# Legacy separation-band sugar. Unset by default; setting either one ANDs it
# onto REGION_WHERE, so the original near/far invocations still work verbatim.
REGION_MIN_SEP=${REGION_MIN_SEP:-}
REGION_MAX_SEP=${REGION_MAX_SEP:-}
REGION_MIN_SEED=${REGION_MIN_SEED:-$(( NOMINAL_BASE + NOMINAL ))}
MAX_HOURS=${MAX_HOURS:-99}

# Which checkpoint. Defaults to the rgb run, since that is the T-I deliverable
# and the frozen base for T-IV - failure modes must be characterised on the
# policy T-III and T-IV actually improve.
CKPT=${CKPT:-}
STATE_FLAG=${STATE_FLAG:-}

t0=$(date +%s)
stage() { echo ""; echo "=== $* [$(( $(date +%s) - t0 ))s] ==="; }

need_ckpt() {
  [ -n "$CKPT" ] || { echo "!! set CKPT=/path/to/best_eval_success_once.pt"; exit 1; }
  [ -f "$CKPT" ] || { echo "!! no such checkpoint: $CKPT"; exit 1; }
}

mkdir -p "$OUT"
cd "$OUT"

do_index() {
  stage "seed index - $INDEX_SEEDS seeds, no policy"
  python "$HERE/seed_index.py" "$INDEX_SEEDS" --out "$OUT/seeds.csv"
}

do_t1() {
  need_ckpt
  # The assignment's deliverable: success rate over 100 rollouts x 3 seeds.
  #
  # Each of the three gets its OWN block of 100 episode seeds AND its own
  # policy seed, so the three numbers are independent estimates and their
  # spread is a real error bar. Reusing one seed block and varying only the
  # policy seed would measure DDPM sampling noise while holding the initial
  # states fixed - a much smaller quantity, and not what "3 seeds" means.
  #
  # CLAUDE.md: at high num_eval_envs wandb logs one mean per eval, so T-I's
  # error bars cannot come from the training logs. They come from here.
  if [ "$T1_BASE" -lt "$DEMO_SEED_CEILING" ]; then
    echo "!! T1_BASE=$T1_BASE is inside the demonstration seed range"
    echo "   [0, $DEMO_SEED_CEILING). Those are initial states the policy was"
    echo "   TRAINED on - evaluating there measures memorisation, not success."
    echo "   Use T1_BASE=6000. See the seed-block header in this file."
    exit 1
  fi
  for s in 1 2 3; do
    lo=$(( T1_BASE + (s - 1) * T1_EPISODES ))
    hi=$(( lo + T1_EPISODES ))
    stage "T-I eval $s/3 - seeds $lo:$hi, policy seed $s"
    python "$HERE/mine_rollouts.py" "$CKPT" $STATE_FLAG \
      --seeds "$lo:$hi" --policy-seed "$s" --num-envs "$NUM_ENVS" \
      --out "$OUT/t1_seed$s"
  done
  stage "T-I summary"
  python "$HERE/analyze_rollouts.py" "$OUT"/t1_seed*.csv --figdir "$ROOT/figures"
}

do_mine() {
  need_ckpt
  stage "nominal pass - $NOMINAL episodes, seeds $NOMINAL_BASE:$(( NOMINAL_BASE + NOMINAL ))"
  python "$HERE/mine_rollouts.py" "$CKPT" $STATE_FLAG \
    --seeds "$NOMINAL_BASE:$(( NOMINAL_BASE + NOMINAL ))" \
    --policy-seed 1 --num-envs "$NUM_ENVS" \
    --max-hours "$MAX_HOURS" --out "$OUT/nominal"
}

do_region() {
  need_ckpt
  [ -f "$OUT/seeds.csv" ] || { echo "!! run 'bash t2/run_t2.sh index' first"; exit 1; }

  # Draw region seeds from ABOVE the nominal range so the two passes cannot
  # overlap. Re-measuring on the rollouts that found the region measures noise,
  # not a failure mode - CLAUDE.md is explicit about this.
  stage "selecting seeds - $REGION_TAG - seed >= $REGION_MIN_SEED"
  SEP_ARGS=()
  [ -n "$REGION_MIN_SEP" ] && SEP_ARGS+=(--min-sep "$REGION_MIN_SEP")
  [ -n "$REGION_MAX_SEP" ] && SEP_ARGS+=(--max-sep "$REGION_MAX_SEP")
  python "$HERE/select_seeds.py" "$OUT/seeds.csv" \
    --min-seed "$REGION_MIN_SEED" -n "$REGION_SEEDS" \
    --where "$REGION_WHERE" "${SEP_ARGS[@]}" --split "$REGION_BLOCKS" \
    --out "$OUT/region_${REGION_TAG}_seeds.csv"

  # 100 rollouts x 3 seeds, per failure mode - the assignment's deliverable.
  # Block i gets policy seed i, so the three rates are independent estimates.
  for b in $(seq 1 "$REGION_BLOCKS"); do
    stage "targeted '$REGION_TAG' $b/$REGION_BLOCKS - $REGION_EPISODES episodes, policy seed $b"
    python "$HERE/mine_rollouts.py" "$CKPT" $STATE_FLAG \
      --seed-file "$OUT/region_${REGION_TAG}_seeds_${b}.csv" \
      --repeats "$REGION_REPEATS" \
      --policy-seed "$b" --num-envs "$NUM_ENVS" --trace-stride 1 \
      --max-hours "$MAX_HOURS" --out "$OUT/region_${REGION_TAG}_seed$b"
  done
}

do_backend() {
  need_ckpt
  [ -f "$OUT/nominal.csv" ] || { echo "!! run 'bash t2/run_t2.sh mine' first"; exit 1; }
  [ -f "$OUT/nominal_states.npz" ] || {
    echo "!! $OUT/nominal_states.npz is missing."
    echo "   The paired check replays the CPU pass's exact initial states, so it"
    echo "   needs the sidecar, not just the CSV. Only the CSVs are committed;"
    echo "   pull the npz from the pod that mined them."
    exit 1
  }
  # The CPU arm is nominal.csv, already measured - nothing is re-run on CPU.
  # Only this arm needs the GPU, which is why 'backend' is not part of 'all'.
  stage "backend agreement - replaying nominal on physx_cuda"
  BACKEND=physx_cuda python "$HERE/backend_check.py" "$CKPT" $STATE_FLAG \
    --states "$OUT/nominal_states.npz" --cpu-csv "$OUT/nominal.csv" \
    --num-envs "${GPU_ENVS:-64}" --out "$OUT/backend_check"
}

do_analyze() {
  stage "analysis"
  # nominal + every targeted pass. t1 is a separate deliverable with its own
  # summary, and mixing it in would pool three policy seeds into one rate.
  shopt -s nullglob
  local rollouts=()
  for f in "$OUT"/region_*.csv; do
    # region_*_seeds.csv are seed lists, not rollouts.
    case "$f" in *_seeds.csv|*_seeds_[0-9].csv) ;; *) rollouts+=("$f") ;; esac
  done
  python "$HERE/analyze_rollouts.py" "$OUT/nominal.csv" "${rollouts[@]}" \
    --figdir "$ROOT/figures"
}

do_videos() {
  need_ckpt
  : "${SEEDS:?set SEEDS=1234,5678 - analyze prints the shortlist}"
  stage "videos for seeds $SEEDS"
  python "$HERE/record_seeds.py" "$CKPT" $STATE_FLAG --seeds "$SEEDS" \
    --want "${WANT:-fail}" --attempts "${ATTEMPTS:-5}" --out "$OUT/videos"
}

plan() {
  local t1=$(( 3 * T1_EPISODES ))
  local reg=$(( REGION_SEEDS * REGION_REPEATS ))   # 3 blocks x 100, one policy seed each
  echo ""
  echo "  budget"
  echo "    t1       $t1 episodes   (3 x $T1_EPISODES, seeds $T1_BASE:$(( T1_BASE + 3 * T1_EPISODES )); demos end at $DEMO_SEED_CEILING)"
  echo "    nominal  $NOMINAL episodes   (seeds $NOMINAL_BASE:$(( NOMINAL_BASE + NOMINAL )))"
  echo "    region   $reg episodes   ($REGION_SEEDS x $REGION_REPEATS, tag '$REGION_TAG',"
  echo "                     seed >= $REGION_MIN_SEED, where: $REGION_WHERE)"
  echo "    total    $(( t1 + NOMINAL + reg )) episodes"
  echo ""
  echo "    index    $INDEX_SEEDS seeds. For the exact hit rate of this region"
  echo "             and the index size it implies, without launching anything:"
  echo ""
  echo "      python t2/select_seeds.py $OUT/seeds.csv --min-seed $REGION_MIN_SEED \\"
  echo "             -n $REGION_SEEDS --where \"$REGION_WHERE\" --dry-run"
  echo ""
}

case "${1:-all}" in
  index)   do_index ;;
  t1)      do_t1 ;;
  mine)    do_mine ;;
  region)  do_region ;;
  backend) do_backend ;;
  analyze) do_analyze ;;
  videos)  do_videos ;;
  plan)    plan ;;
  all)     plan; do_index; do_t1; do_mine; do_region; do_analyze ;;
  *) sed -n '2,20p' "$SELF"; exit 1 ;;
esac

stage "done - outputs in $OUT"
ls -la "$OUT" | sed 's/^/  /'
echo ""
echo "  Pull before stopping the pod:  bash setup/transfer.sh info"
