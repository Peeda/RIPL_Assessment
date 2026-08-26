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
#   test       spec/loader self-tests. No sim, no GPU, no API key, ~1 s. First
#              because it is the cheapest place to find the contract is wrong.
#   prompt     mp4 -> frames, then assemble prompts/*.md + the env-source
#              snapshot + the contract into the exact text the model is sent.
#              Needs the ffmpeg binary only. READ prompt.txt before generating.
#   generate   the one API call. Needs `anthropic` and a key. Refuses to
#              overwrite an existing generation.
#   lint       the static check over the artifacts. Stdlib, 0.2 s.
#   check      the three measurements: sampler | reward | align. ~10 min,
#              almost all of it in align.
#   calibrate  the same battery on ManiSkill's own 8-stage dense reward, so the
#              AUC is a comparison rather than a bare number. Optional, ~8 min.
#   summary    read the measurements, print OK/WARN. ALWAYS EXITS 0 - a
#              threshold missed is a finding for the report, not a refusal.
#   report     the two figures.
#
# THE INPUT COMES FROM T-II. mp4s are gitignored and never present in a fresh
# clone, so produce one first:
#   SEEDS=$(head -2 t2/results/mode_gap_seed1.csv | tail -1 | cut -d, -f5) \
#     WANT=fail bash t2/run.sh videos
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
  stage "self-tests - the contract and the checker. No sim, no key"
  python3 "$HERE/test_t3.py"
}

do_prompt() {
  stage "prompt - $FRAMES frames, evenly spaced"
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
    ${VIDEO:+--video "$VIDEO"} --frames "$FRAMES" \
    --seed "${VIDEO_SEED:-unknown}" --outcome "${OUTCOME:-failure}" \
    ${WITH_STATS:+--with-stats}
}

do_generate() {
  stage "generate - one API call"
  need_key
  [ -f "$RUN/blocks.json" ] || { echo "!! run 'bash t3/run.sh prompt' first"; exit 1; }
  python3 "$HERE/generate.py" --run "$RUN" \
    --model "${T3_MODEL:-claude-opus-5}" ${FORCE:+--force}
}

do_lint() {
  stage "lint - the static check, no simulator"
  python3 "$HERE/loader.py" "$RUN"
}

do_check() {
  stage "check - sampler, reward, align"
  find_ckpt
  python "$HERE/check.py" sampler --run "$RUN" --mode "$MODE" --out "$OUT" \
    --draws "$DRAWS"
  python "$HERE/check.py" reward --run "$RUN" --mode "$MODE" --out "$OUT" \
    $STATE_FLAG
  python "$HERE/check.py" align --run "$RUN" --mode "$MODE" --out "$OUT" \
    --ckpt "$CKPT" --episodes "$EPISODES" --num-envs "$NUM_ENVS" $STATE_FLAG
}

# The calibration arm. Without it every AUC in the report is uncalibrated. With
# it, both outcomes are reportable - "the LLM beat the reward it was shown at
# the failing stage" is a result, and "it did not" is a better one.
do_calibrate() {
  stage "calibrate - the same battery on ManiSkill's 8-stage dense reward"
  find_ckpt
  python "$HERE/check.py" reward --run "$RUN" --mode "$MODE" --out "$OUT" \
    --reward "$HERE/fixtures/stock_reward.py" --label stock $STATE_FLAG
  python "$HERE/check.py" align --mode "$MODE" --out "$OUT" --ckpt "$CKPT" \
    --reward "$HERE/fixtures/stock_reward.py" --label stock \
    --episodes "$EPISODES" --num-envs "$NUM_ENVS" $STATE_FLAG
}

do_summary() {
  stage "summary - the measurements, read back"
  python3 "$HERE/summary.py" "$OUT" --mode "$MODE" --run "$RUN"
}

do_report() {
  stage "report - figures"
  python "$HERE/report.py" --out "$OUT" --mode "$MODE" \
    --figdir "$ROOT/figures" --index "$T2_RESULTS/seeds.csv"
}

STAGE=${1:-all}
case "${1:-all}" in
  test)      do_test ;;
  prompt|frames) do_preflight; do_prompt ;;
  generate)  do_preflight; do_generate ;;
  lint)      do_lint ;;
  check)     do_preflight; do_check ;;
  calibrate) do_preflight; do_calibrate ;;
  summary)   do_summary ;;
  report)    do_report ;;
  all)       do_test; do_preflight; do_prompt; do_generate; do_lint
             do_check; do_summary; do_report ;;
  # trap off first: an unknown stage is a typo, and telling someone to rescue
  # data because they mistyped is how a real warning stops being read.
  *) trap - EXIT; sed -n '2,32p' "$SELF"; exit 1 ;;
esac

stage "done - artifacts in $RUN, measurements in $OUT"
echo ""
echo "  PULL BEFORE STOPPING THE POD - the pod never pushes:"
echo "      bash setup/transfer.sh info"
