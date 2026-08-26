#!/usr/bin/env bash
# ==============================================================================
# t3/run.sh - the T-III LLM pipeline, end to end.
#
#   source /workspace/ripl/env.sh
#   export ANTHROPIC_API_KEY=...        # or $RIPL_ROOT/anthropic.env, chmod 600
#   bash setup/apply_patches.sh
#   tmux new -s t3
#   MODE=gap VIDEO=~/clip.mp4 bash t3/run.sh all 2>&1 | tee ~/t3.log
#
# Stages (default: all), in dependency order:
#
#   test       spec / loader / gate self-tests. No sim, no GPU, no API key,
#              ~3 s. Runs first because it is the cheapest place to find out
#              the contract is wrong.
#   frames     mp4 -> jpegs. Needs the ffmpeg binary only.
#   prompt     assemble prompts/*.md + the env-source snapshot + the contract
#              into the exact text the model is sent. Read it before generating.
#   generate   the one API call. Needs `anthropic` and a key. Refuses to
#              overwrite an existing generation.
#   lint       layer A over the artifacts. Stdlib, 0.2 s, before any sim.
#   probes     layers B and C - shape, purity, no mutation, and the
#              degenerate-state battery. ~2 min, no checkpoint needed.
#   sampler    layer E - the biased distribution. ~1 min.
#   smoke      5 episodes of align, into $T3_OUT/smoke. ~1 min, and the only
#              way to find out the rollout path works without paying 8 minutes.
#   align      layer D, the centrepiece - 100 real episodes of the frozen
#              policy, scored by the generated reward. ~8 min per mode.
#   calibrate  the same battery against ManiSkill's own 8-stage dense reward,
#              so every AUC is a comparison rather than a bare number.
#   verify     THE GATE. Reads only files. Exits non-zero.
#   report     the figures.
#
# Roughly 25 minutes per mode on an RTX 4090, most of it in `align`.
#
# NOTHING IS REPORTABLE, AND NOTHING GOES INTO T-IV, UNTIL `verify` EXITS 0.
#
# THE INPUT COMES FROM T-II. mp4s are gitignored and never present in a fresh
# clone, so produce one first:
#   SEEDS=$(head -2 t2/results/mode_gap_seed1.csv | tail -1 | cut -d, -f5) \
#     bash t2/run.sh videos
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"
# The API key lives OUTSIDE the repo and outside env.sh, which setup_runpod.sh
# rewrites on every pod build and which gets echoed into build logs.
KEYFILE="${RIPL_ROOT:-/workspace/ripl}/anthropic.env"
# shellcheck source=/dev/null
[ -f "$KEYFILE" ] && source "$KEYFILE"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SELF=$HERE/$(basename "$0")

MODE=${MODE:-gap}
OUT=${T3_OUT:-${RIPL_ROOT:-/workspace/ripl}/t3}
RUN=${T3_RUN:-$ROOT/t3/artifacts/$MODE${GEN:+_gen$GEN}}
export T3_RUN

ENV_ID=${ENV_ID:-StackCube-T3-v1}
CTRL=${CTRL:-pd_ee_delta_pos}
BACKEND=${BACKEND:-physx_cpu}
MAX_EP_STEPS=${MAX_EP_STEPS:-200}
export ENV_ID CTRL BACKEND MAX_EP_STEPS
export T2_RESULTS=${T2_RESULTS:-$ROOT/t2/results}

FRAMES=${FRAMES:-10}
SELECT=${SELECT:-uniform}
EPISODES=${EPISODES:-100}
NUM_ENVS=${NUM_ENVS:-10}
DRAWS=${DRAWS:-4096}
export STATE_FLAG=${STATE_FLAG:-}

t0=$(date +%s)
stage() { echo ""; echo "######## $* [$(( $(date +%s) - t0 ))s]"; }

on_exit() {
  local rc=$?
  [ "$rc" -eq 0 ] && return 0
  echo ""
  echo "######## FAILED at stage '${STAGE:-?}' (exit $rc) after $(( $(date +%s) - t0 ))s"
  echo ""
  echo "  Anything already in $OUT and $RUN is NOT reproducible - the API call"
  echo "  is sampled and the rollouts are stochastic. Pull before you stop or"
  echo "  terminate the pod:"
  echo "      bash setup/transfer.sh info"
  echo ""
  echo "  A FAILED VERIFY IS A RESULT, not an accident: keep the generation, it"
  echo "  is the report's account of how an LLM-written reward goes wrong."
  exit "$rc"
}
trap on_exit EXIT

REPO_CKPT=$ROOT/checkpoints/stackcube_rgb_spatial_800demos.pt

find_ckpt() {
  if [ -n "${CKPT:-}" ]; then echo "  ckpt        $CKPT  (from \$CKPT)"; return 0; fi
  if [ -f "$REPO_CKPT" ]; then
    export CKPT="$REPO_CKPT"
    echo "  ckpt        $CKPT  (committed in-repo)"
    return 0
  fi
  echo "!! no checkpoint. Expected $REPO_CKPT, or set CKPT=..."
  exit 1
}

do_preflight() {
  : "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"
  echo "  mode        $MODE"
  echo "  run dir     $RUN"
  echo "  out         $OUT"
  mkdir -p "$OUT" "$RUN"
}

need_key() {
  if [ -z "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    echo "!! ANTHROPIC_API_KEY is not set."
    echo "   export it, or put it in $KEYFILE (chmod 600) which this script sources."
    echo "   Do NOT put it in env.sh - setup_runpod.sh rewrites that file."
    exit 1
  fi
  echo "  api key     set (...${ANTHROPIC_API_KEY: -4})"
}

# --- stages ----------------------------------------------------------------

do_test() {
  stage "self-tests - the contract, layer A, and the gate. No sim, no key"
  python3 "$HERE/test_spec.py"
  python3 "$HERE/test_loader.py"
  python3 "$HERE/test_verify.py"
}

do_frames() {
  stage "frames - $FRAMES by '$SELECT'"
  if [ -z "${VIDEO:-}" ] && [ ! -d "$RUN/frames" ]; then
    echo "!! set VIDEO=/path/to/clip.mp4"
    echo ""
    echo "   T-III's input is a clip of a real failure, and T-II produces it:"
    echo "     SEEDS=<a seed from $T2_RESULTS/mode_${MODE}_seed1.csv> \\"
    echo "       WANT=fail bash t2/run.sh videos"
    echo "   mp4s are gitignored, so a fresh clone never has one."
    exit 1
  fi
  python3 "$HERE/assemble.py" --mode "$MODE" --out "$RUN" \
    ${VIDEO:+--video "$VIDEO"} --frames "$FRAMES" --select "$SELECT" \
    --seed "${VIDEO_SEED:-unknown}" --outcome "${OUTCOME:-failure}" \
    ${WITH_STATS:+--with-stats} ${ALLOW_DRIFT:+--allow-source-drift}
}

do_prompt() { do_frames; }

do_generate() {
  stage "generate - one API call"
  need_key
  [ -f "$RUN/blocks.json" ] || { echo "!! run 'bash t3/run.sh prompt' first"; exit 1; }
  python3 "$HERE/generate.py" --run "$RUN" \
    --model "${T3_MODEL:-claude-opus-5}" ${FORCE:+--force}
}

do_lint() {
  stage "lint - layer A, no simulator"
  python3 "$HERE/loader.py" "$RUN"
}

do_probes() {
  stage "probes - layers B and C"
  [ -f "$HERE/fixtures/grasp_hover_states.npz" ] || {
    find_ckpt
    echo "  capturing the grasped-state fixture (needed for P7_held)"
    python "$HERE/probes.py" --make-fixture "$CKPT" $STATE_FLAG
  }
  python "$HERE/probes.py" --run "$RUN" --mode "$MODE" --out "$OUT" $STATE_FLAG
}

do_sampler() {
  stage "sampler - layer E, $DRAWS draws"
  python "$HERE/sampler_check.py" --run "$RUN" --mode "$MODE" --out "$OUT" \
    --draws "$DRAWS"
}

# A separate directory on purpose: sharing $OUT would leave a 5-episode
# align CSV that the real run's resume logic would accept as finished.
do_smoke() {
  stage "smoke - 5 episodes through the full rollout path, into $OUT/smoke"
  find_ckpt
  rm -rf "$OUT/smoke"; mkdir -p "$OUT/smoke"
  python "$HERE/align.py" --run "$RUN" --mode "$MODE" --out "$OUT/smoke" \
    --ckpt "$CKPT" --episodes 5 --control-episodes 2 --num-envs 5 $STATE_FLAG
  echo ""
  echo "  The rollout path works: env registered in every worker, seeds honoured,"
  echo "  reward_mode=dense returning the generated reward, metrics read back."
  echo "  (verify.py will reject this shape - 5 episodes, not $EPISODES. That is"
  echo "   the point; it is a smoke test, not a measurement.)"
}

do_align() {
  stage "align - layer D, $EPISODES episodes on the T-II evaluation seeds"
  find_ckpt
  python "$HERE/align.py" --run "$RUN" --mode "$MODE" --out "$OUT" \
    --ckpt "$CKPT" --episodes "$EPISODES" --num-envs "$NUM_ENVS" $STATE_FLAG
}

# The calibration arm. Without it every AUC in the report is uncalibrated, and
# "the LLM's reward is worse than the one it was shown" is a finding that cannot
# be made. Runs the identical battery on ManiSkill's own dense reward.
do_calibrate() {
  stage "calibrate - the same battery on ManiSkill's 8-stage dense reward"
  find_ckpt
  python "$HERE/probes.py" --run "$RUN" --mode "$MODE" --out "$OUT" \
    --reward "$HERE/fixtures/stock_reward.py" --label stock $STATE_FLAG
  python "$HERE/align.py" --mode "$MODE" --out "$OUT" --ckpt "$CKPT" \
    --reward "$HERE/fixtures/stock_reward.py" --label stock \
    --episodes "$EPISODES" --num-envs "$NUM_ENVS" $STATE_FLAG
}

do_verify() {
  stage "verify - the gate"
  python3 "$HERE/verify.py" "$OUT" --mode "$MODE" --run "$RUN" \
    --t2-results "$T2_RESULTS"
}

do_report() {
  stage "report - figures"
  python "$HERE/report.py" --out "$OUT" --mode "$MODE" \
    --figdir "$ROOT/figures" --index "$T2_RESULTS/seeds.csv"
}

STAGE=${1:-all}
case "${1:-all}" in
  test)      do_test ;;
  frames)    do_preflight; do_frames ;;
  prompt)    do_preflight; do_prompt ;;
  generate)  do_preflight; do_generate ;;
  lint)      do_lint ;;
  probes)    do_preflight; do_probes ;;
  sampler)   do_preflight; do_sampler ;;
  smoke)     do_preflight; do_smoke ;;
  align)     do_preflight; do_align ;;
  calibrate) do_preflight; do_calibrate ;;
  verify)    do_verify ;;
  report)    do_report ;;
  all)       do_test; do_preflight; do_frames; do_generate; do_lint
             do_probes; do_sampler; do_smoke; do_align; do_calibrate
             do_verify; do_report ;;
  # trap off first: an unknown stage is a typo, and telling someone to rescue
  # data because they mistyped is how a real warning stops being read.
  *) trap - EXIT; sed -n '2,44p' "$SELF"; exit 1 ;;
esac

stage "done - artifacts in $RUN, measurements in $OUT"
echo ""
echo "  PULL BEFORE STOPPING THE POD - the pod never pushes:"
echo "      bash setup/transfer.sh info"
