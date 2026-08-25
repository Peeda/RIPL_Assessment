#!/usr/bin/env bash
# ==============================================================================
# t2/run_all.sh - everything T-II still needs, in order, in one command.
#
#   source /workspace/ripl/env.sh
#   tmux new -s t2
#   bash t2/run_all.sh 2>&1 | tee ~/t2-all.log
#
# Stages, in dependency order (default: all):
#
#   preflight  env, patches, checkpoint. Fails fast rather than at minute 40.
#   index      seed -> initial state table, 12000 seeds. No policy, no GPU.
#   size       hit rate per region. Free, and it is the PRE-REGISTRATION
#              receipt: it timestamps the thresholds before any rollout.
#   policy     proves the rollouts are driven by the trained weights, by
#              racing them against untrained / random / zero actions. ~4 min.
#   modes      the three per-failure-mode evaluations. 100 rollouts x 3 seeds
#              each - the assignment's deliverable. ~27 min.
#   feasible   is the far region the policy's fault or the task's? The
#              motion-planner oracle is the load-bearing half and is a
#              metadata join; the reach map is an optional figure and is
#              allowed to fail without taking the run with it.
#   backend    does physx_cuda reproduce physx_cpu? Licenses T-IV training on
#              GPU while evaluating on CPU. Needs a GPU.
#   analyze    curves, taxonomy, Wilson intervals, figures, video shortlist.
#   verify     asserts the pass is what it claims. Exits non-zero if not.
#
# Every stage skips work that is already done, so re-running after an
# interruption resumes rather than restarting - and never overwrites a finished
# rollout pass, because rollouts are stochastic and anything clobbered is gone.
# FORCE=1 overrides that, deliberately.
#
# Total: roughly an hour on an RTX 4090, most of it in `modes`.
# ==============================================================================
set -euo pipefail

ENVSH="${RIPL_ROOT:-/workspace/ripl}/env.sh"
# shellcheck source=/dev/null
[ -f "$ENVSH" ] && source "$ENVSH"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${T2_OUT:-${RIPL_ROOT:-/workspace/ripl}/t2}
SELF=$HERE/$(basename "$0")

export INDEX_SEEDS=${INDEX_SEEDS:-12000}
# Exported, not just defaulted: run_all invokes policy_check.py directly AND
# delegates to run_t2.sh, and both have to agree on whether this is the state
# or the rgb policy. set -u turns a bare $STATE_FLAG into a hard failure.
export STATE_FLAG=${STATE_FLAG:-}

t0=$(date +%s)
stage() { echo ""; echo "############ $* [$(( $(date +%s) - t0 ))s]"; }

# ---------------------------------------------------------------------------
# The checkpoint is DERIVED, not pasted. t1/run_pipeline.sh builds the run
# directory out of env id, obs mode, encoder variant and demo count, so the
# path is a function of the run that produced it - and a pasted path is how you
# end up reporting numbers from the pooled-encoder arm by accident.
# ---------------------------------------------------------------------------
find_ckpt() {
  [ -n "${CKPT:-}" ] && return 0
  local dp="${MANISKILL_REPO:?source env.sh first}/examples/baselines/diffusion_policy"
  local env_id=${ENV_ID:-StackCube-v1}
  local want="$dp/runs/diffusion_policy-$env_id-rgb-${VARIANT:-spatial}-${NUM_DEMOS:-800}_motionplanning_demos-1/checkpoints/best_eval_success_once.pt"
  if [ -f "$want" ]; then
    export CKPT="$want"
    return 0
  fi
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
  stage "preflight"
  : "${MANISKILL_REPO:?source /workspace/ripl/env.sh first}"
  find_ckpt
  echo "  CKPT        $CKPT"
  echo "  OUT         $OUT"
  echo "  INDEX_SEEDS $INDEX_SEEDS"
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

do_index() {
  # 12000 because the farb region hits ~3.4% of eligible seeds and needs 300 of
  # them from above the nominal block. Deterministic, so re-running rewrites
  # identical rows for the seeds already covered and existing seed lists stay
  # valid.
  local n
  n=$(wc -l < "$OUT/seeds.csv" 2>/dev/null || echo 0)
  if [ "$n" -gt "$INDEX_SEEDS" ] && [ "${FORCE:-0}" != "1" ]; then
    stage "index - already $((n - 1)) seeds, skipping"
    return
  fi
  stage "index - $INDEX_SEEDS seeds"
  bash "$HERE/run_t2.sh" index
}

do_size() {
  stage "size - pre-registration receipt, no rollouts"
  bash "$HERE/run_modes.sh" size
}

do_modes() {
  stage "modes - three per-mode evals, 100 rollouts x 3 seeds each"
  bash "$HERE/run_modes.sh" gap
  bash "$HERE/run_modes.sh" farb
  bash "$HERE/run_modes.sh" nearbase
}

do_feasible() {
  stage "feasibility - is the far region the policy's fault or the task's?"
  local json
  json=$(ls "${MS_ASSET_DIR:-}/demos/${ENV_ID:-StackCube-v1}/motionplanning/"*.json 2>/dev/null | head -1 || true)
  if [ -n "$json" ]; then
    python "$HERE/demo_feasibility.py" "$json" --index "$OUT/seeds.csv" \
      --out "$OUT/demo_feasibility.csv" || echo "  (feasibility join declined - see message above)"
  else
    echo "  !! no demo trajectory.json under \$MS_ASSET_DIR; skipping the"
    echo "     motion-planner oracle. The reach map below still runs."
  fi
  if [ -f "$OUT/reach_map.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  reach map already computed, skipping"
  else
    # Non-fatal on purpose. demo_feasibility.py above already settles the
    # feasibility question, and settles it with a better instrument - the
    # motion planner is the actual task solver, not an IK approximation of
    # one. This adds a figure and a manipulability curve; neither is worth
    # taking the run down for.
    python "$HERE/reach_map.py" --out "$OUT/reach_map.csv" --figdir "$ROOT/figures" \
      || echo "  !! reach_map failed - optional, continuing. See the message above."
  fi
}

do_backend() {
  if [ ! -f "$OUT/nominal_states.npz" ]; then
    stage "backend - SKIPPED"
    echo "  $OUT/nominal_states.npz is missing. The paired check replays the"
    echo "  nominal pass's exact initial states, so it needs that sidecar -"
    echo "  only the CSVs are committed. Run this on the pod that mined them,"
    echo "  or rsync the npz across."
    return
  fi
  stage "backend - physx_cuda vs physx_cpu, paired by injected initial state"
  bash "$HERE/run_t2.sh" backend
}

do_policy() {
  # Proves the actions come from THESE weights. verify.py can show the episodes
  # are the right seeds in the right region without the policy ever being
  # consulted, so this is the gap it cannot close offline. ~4 min.
  stage "policy check - trained weights vs untrained / random / zero"
  python "$HERE/policy_check.py" "$CKPT" $STATE_FLAG --episodes "${PC_EPISODES:-30}"
}

do_analyze() { stage "analyze";  bash "$HERE/run_t2.sh" analyze; }
do_verify()  { stage "verify";   python "$HERE/verify.py" "$OUT"; }

case "${1:-all}" in
  preflight) do_preflight ;;
  index)     do_preflight; do_index ;;
  size)      do_preflight; do_size ;;
  modes)     do_preflight; do_modes ;;
  feasible)  do_preflight; do_feasible ;;
  backend)   do_preflight; do_backend ;;
  policy)    do_preflight; do_policy ;;
  analyze)   do_analyze ;;
  verify)    do_verify ;;
  # Deliverables BEFORE the optional diagnostics. analyze and verify are the
  # T-II outputs and take about a minute between them; feasible and backend are
  # slower, and backend is really a T-IV prerequisite. Running them first meant
  # a crash in an optional figure generator cost the required outputs.
  all)       do_preflight; do_index; do_size; do_policy; do_modes
             do_analyze; do_verify; do_feasible; do_backend ;;
  *) sed -n '2,27p' "$SELF"; exit 1 ;;
esac

stage "done"
echo ""
echo "  Outputs in $OUT; figures in $ROOT/figures."
echo "  PULL BEFORE STOPPING THE POD - the pod never pushes, so figures and"
echo "  the .npz sidecars do not come back through git:"
echo "      bash setup/transfer.sh info"
