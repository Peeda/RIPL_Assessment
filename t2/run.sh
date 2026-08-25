#!/usr/bin/env bash
# ==============================================================================
# t2/run.sh - the T-II failure-mode evaluation, end to end.
#
#   source /workspace/ripl/env.sh
#   bash setup/apply_patches.sh          # setup_runpod.sh re-clones and wipes it
#   tmux new -s t2
#   bash t2/run.sh all 2>&1 | tee ~/t2.log
#
# Stages (default: all), in dependency order:
#
#   test     geometry + verifier self-tests. No sim, no GPU, ~3 s. Runs first
#            because it is the cheapest place to find out the definition of a
#            failure mode is wrong.
#   index    seed -> initial state table. No policy, no GPU, ~5 min.
#   check    prove the rollouts are driven by THESE weights, by racing them
#            against untrained / random / zero actions. ~4 min.
#   eval     the deliverable: per-mode success over 100 rollouts x 3 seeds,
#            for both modes plus a nominal reference arm. ~27 min.
#   verify   assert the pass is what it claims. Exits non-zero if not.
#   report   the tables and figures.
#   videos   mp4s for named seeds:  SEEDS=1234,5678 bash t2/run.sh videos
#
# Roughly 40 minutes on an RTX 4090, most of it in `eval`.
#
# Every stage skips work already done, so re-running after an interruption
# resumes rather than restarting - and eval never overwrites a finished block,
# because rollouts are stochastic and anything clobbered is gone. FORCE=1
# overrides that, deliberately.
#
# THE RUN IS ONLY REPORTABLE IF `verify` EXITS 0. It re-derives the geometry
# from the logged poses, joins them against an independently built seed index,
# and checks the 100 x 3 shape - see t2/README.md.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${T2_OUT:-${RIPL_ROOT:-/workspace/ripl}/t2}
SELF=$HERE/$(basename "$0")

ENV_ID=${ENV_ID:-StackCube-v1}
CTRL=${CTRL:-pd_ee_delta_pos}
BACKEND=${BACKEND:-physx_cpu}
export ENV_ID CTRL BACKEND

# physx_cpu vectorises by SUBPROCESS, so this is a process count, not a batch
# width. It buys nothing above the core count.
NUM_ENVS=${NUM_ENVS:-10}
# 25000 because the 'farb' region hits ~3.4% of eligible seeds and eval_modes
# draws only from seeds >= 10000, so 300 of them need ~9k eligible. `index`
# prints exactly how many each mode found and what it would take - read it.
INDEX_SEEDS=${INDEX_SEEDS:-25000}
EPISODES=${EPISODES:-100}      # per block; the assignment asks for 100 x 3
BLOCKS=${BLOCKS:-3}
# Exported because both `check` and `eval` need to agree on state vs rgb, and
# set -u makes a bare expansion of an unset var fatal.
export STATE_FLAG=${STATE_FLAG:-}

t0=$(date +%s)
stage() { echo ""; echo "######## $* [$(( $(date +%s) - t0 ))s]"; }

# ---------------------------------------------------------------------------
# The checkpoint is DERIVED, not pasted. t1/run_pipeline.sh builds the run
# directory out of env id, obs mode, encoder variant and demo count, so the
# path is a function of the run that produced it - and a pasted path is how you
# end up reporting numbers from the pooled-encoder arm by accident.
# ---------------------------------------------------------------------------
find_ckpt() {
  [ -n "${CKPT:-}" ] && return 0
  local dp="${MANISKILL_REPO:?source env.sh first}/examples/baselines/diffusion_policy"
  local want="$dp/runs/diffusion_policy-$ENV_ID-rgb-${VARIANT:-spatial}-${NUM_DEMOS:-800}_motionplanning_demos-1/checkpoints/best_eval_success_once.pt"
  if [ -f "$want" ]; then export CKPT="$want"; return 0; fi
  echo "!! could not find the expected checkpoint:"
  echo "     $want"
  echo ""
  echo "   Candidates on this pod:"
  find "$dp/runs" -name "best_eval_success_once.pt" 2>/dev/null | sed 's/^/     /' || true
  echo ""
  echo "   Set CKPT=... explicitly, or NUM_DEMOS/VARIANT if the run was named"
  echo "   differently. The T-I deliverable and T-IV's frozen base is the"
  echo "   SPATIAL encoder at 800 demos - not the pooled arm."
  exit 1
}

do_preflight() {
  : "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"
  find_ckpt
  echo "  CKPT        $CKPT"
  echo "  OUT         $OUT"
  # patches/0001 makes pool_feature_map an Args field. On a stock checkout a
  # spatial checkpoint cannot be loaded at all, and the error surfaces deep
  # inside load_state_dict as a shape mismatch. Catch it here instead.
  if ! python - <<'PY'
import dataclasses, os, sys
sys.path.insert(0, f"{os.environ['MANISKILL_REPO']}/examples/baselines/diffusion_policy")
import train_rgbd as T
sys.exit(0 if any(f.name == "pool_feature_map" for f in dataclasses.fields(T.Args)) else 1)
PY
  then
    echo "!! ManiSkill is stock - patches/0001 is not applied, so the spatial"
    echo "   encoder checkpoint cannot be loaded. Run:"
    echo "     bash setup/apply_patches.sh"
    exit 1
  fi
  echo "  patches     applied (pool_feature_map is an Args field)"
  mkdir -p "$OUT"
}

# --- stages ----------------------------------------------------------------

do_test() {
  stage "self-tests - geometry and the verifier, no sim"
  python3 "$HERE/test_geometry.py"
  python3 "$HERE/test_verify.py"
}

do_index() {
  stage "seed index - $INDEX_SEEDS seeds, no policy, no GPU"
  local have
  have=$(( $(wc -l < "$OUT/seeds.csv" 2>/dev/null || echo 1) - 1 ))
  if [ "$have" -ge "$INDEX_SEEDS" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  already $have seeds indexed, skipping (FORCE=1 to rebuild)"
    return
  fi
  python "$HERE/seed_index.py" "$INDEX_SEEDS" --out "$OUT/seeds.csv"
}

do_check() {
  # verify.py can show the episodes are the right seeds in the right region
  # without the policy ever being consulted. This is the gap it cannot close
  # offline: that the ACTIONS came from these weights.
  stage "policy check - trained weights vs untrained / random / zero"
  python "$HERE/policy_check.py" "$CKPT" $STATE_FLAG --episodes "${PC_EPISODES:-30}"
}

do_eval() {
  stage "evaluation - $EPISODES rollouts x $BLOCKS seeds, per mode"
  [ -f "$OUT/seeds.csv" ] || { echo "!! run 'bash t2/run.sh index' first"; exit 1; }
  python "$HERE/eval_modes.py" "$CKPT" $STATE_FLAG \
    --modes ${MODES:-nominal gap farb} \
    --index "$OUT/seeds.csv" --out "$OUT" \
    --episodes "$EPISODES" --blocks "$BLOCKS" --num-envs "$NUM_ENVS"
}

do_verify() { stage "verify"; python3 "$HERE/verify.py" "$OUT"; }

do_report() {
  stage "report - tables and figures"
  local disc=()
  # The 1,200-episode discovery pass, if it is on this machine. It is the
  # evidence that the modes are REGIONS rather than post-hoc labels, and it is
  # committed - so this works off-pod too.
  for d in "$OUT/nominal.csv" "$ROOT/t2/results/nominal.csv"; do
    [ -f "$d" ] && { disc=(--discovery "$d"); break; }
  done
  python "$HERE/report.py" --eval "$OUT" "${disc[@]}" --figdir "$ROOT/figures"
}

do_videos() {
  : "${SEEDS:?set SEEDS=1234,5678 - pick them from the mode CSVs}"
  stage "videos for seeds $SEEDS"
  python "$HERE/record_seeds.py" "$CKPT" $STATE_FLAG --seeds "$SEEDS" \
    --want "${WANT:-fail}" --attempts "${ATTEMPTS:-5}" --out "$OUT/videos"
}

case "${1:-all}" in
  test)    do_test ;;
  index)   do_preflight; do_index ;;
  check)   do_preflight; do_check ;;
  eval)    do_preflight; do_eval ;;
  verify)  do_verify ;;
  report)  do_report ;;
  videos)  do_preflight; do_videos ;;
  all)     do_test; do_preflight; do_index; do_check; do_eval
           do_verify; do_report ;;
  *) sed -n '2,33p' "$SELF"; exit 1 ;;
esac

stage "done - outputs in $OUT, figures in $ROOT/figures"
echo ""
echo "  PULL BEFORE STOPPING THE POD - the pod never pushes, so the figures"
echo "  and manifests do not come back through git:"
echo "      bash setup/transfer.sh info"
